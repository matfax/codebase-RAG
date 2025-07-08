import gc
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

from ..utils.cache_key_generator import KeyType, get_cache_key_generator
from ..utils.cache_utils import deserialize_from_cache, serialize_for_cache
from .cache_service import CacheError, CacheOperationError, get_cache_service


@dataclass
class BatchInsertionStats:
    """Statistics for batch insertion operations."""

    total_points: int = 0
    successful_insertions: int = 0
    failed_insertions: int = 0
    batch_count: int = 0
    total_duration: float = 0.0
    average_batch_size: float = 0.0
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_points <= 0:
            return 0.0
        return (self.successful_insertions / self.total_points) * 100

    @property
    def insertions_per_second(self) -> float:
        """Calculate insertions per second rate."""
        if self.total_duration <= 0:
            return 0.0
        return self.successful_insertions / self.total_duration


class QdrantService:
    def __init__(self, host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Configuration
        self.default_batch_size = int(os.getenv("QDRANT_BATCH_SIZE", "500"))
        self.max_retries = int(os.getenv("QDRANT_MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("QDRANT_RETRY_DELAY", "1.0"))

        # Cache integration
        self.cache_service = None
        self.cache_key_generator = None
        self._cache_enabled = os.getenv("QDRANT_CACHE_ENABLED", "true").lower() == "true"
        self._cache_ttl = int(os.getenv("QDRANT_CACHE_TTL", "300"))  # 5 minutes default
        self._connection_cache_ttl = int(os.getenv("QDRANT_CONNECTION_CACHE_TTL", "60"))  # 1 minute default

        self._setup_logging()

    async def _initialize_cache(self) -> None:
        """Initialize cache service if not already initialized."""
        if self._cache_enabled and self.cache_service is None:
            try:
                self.cache_service = await get_cache_service()
                self.cache_key_generator = get_cache_key_generator()
                self.logger.info("Cache service initialized for QdrantService")
            except Exception as e:
                self.logger.warning(f"Failed to initialize cache service: {e}")
                self._cache_enabled = False

    async def _get_from_cache(self, key: str) -> Any:
        """Get value from cache with error handling."""
        if not self._cache_enabled or not self.cache_service:
            return None

        try:
            cached_data = await self.cache_service.get(key)
            if cached_data is not None:
                return deserialize_from_cache(cached_data)
        except Exception as e:
            self.logger.warning(f"Cache get failed for key {key}: {e}")
        return None

    async def _set_in_cache(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set value in cache with error handling."""
        if not self._cache_enabled or not self.cache_service:
            return

        try:
            serialized_data = serialize_for_cache(value)
            await self.cache_service.set(key, serialized_data, ttl or self._cache_ttl)
        except Exception as e:
            self.logger.warning(f"Cache set failed for key {key}: {e}")

    async def _delete_from_cache(self, key: str) -> None:
        """Delete value from cache with error handling."""
        if not self._cache_enabled or not self.cache_service:
            return

        try:
            await self.cache_service.delete(key)
        except Exception as e:
            self.logger.warning(f"Cache delete failed for key {key}: {e}")

    def _generate_cache_key(self, key_type: str, identifier: str, **kwargs) -> str:
        """Generate cache key for Qdrant operations."""
        if not self.cache_key_generator:
            # Fallback to simple key generation if cache key generator is not available
            return f"qdrant:{key_type}:{identifier}"

        try:
            return self.cache_key_generator.generate_key(
                key_type=KeyType.METADATA,
                namespace=f"qdrant_{key_type}",
                project_id="global",
                content={"identifier": identifier, **kwargs},
            )
        except Exception as e:
            self.logger.warning(f"Failed to generate cache key: {e}")
            return f"qdrant:{key_type}:{identifier}"

    def _setup_logging(self) -> None:
        """Setup logging configuration for Qdrant service."""
        if not self.logger.handlers:
            log_level = os.getenv("LOG_LEVEL", "INFO").upper()
            self.logger.setLevel(getattr(logging, log_level, logging.INFO))

            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            self.logger.propagate = False

    def calculate_optimal_batch_size(self, points: list[PointStruct], base_batch_size: int | None = None) -> int:
        """
        Calculate optimal batch size based on point dimensions and metadata size.

        Args:
            points: List of points to analyze
            base_batch_size: Base batch size to start from (uses default if None)

        Returns:
            Optimized batch size
        """
        if not points:
            return base_batch_size or self.default_batch_size

        if base_batch_size is None:
            base_batch_size = self.default_batch_size

        # Estimate memory usage per point
        sample_point = points[0]

        # Vector dimension (primary memory consumer)
        vector_size = len(sample_point.vector) if sample_point.vector else 768  # Default dimension

        # Estimate payload size (rough approximation)
        payload_size = len(str(sample_point.payload)) if sample_point.payload else 100

        # Estimate memory per point in bytes
        # Float32 vectors: 4 bytes per dimension + payload overhead
        estimated_memory_per_point = (vector_size * 4) + payload_size + 100  # 100 bytes overhead

        # Target maximum memory per batch (default: 50MB)
        max_memory_per_batch = int(os.getenv("QDRANT_MAX_BATCH_MEMORY_MB", "50")) * 1024 * 1024

        # Calculate optimal batch size based on memory
        memory_based_batch_size = max_memory_per_batch // estimated_memory_per_point

        # Use the smaller of memory-based and configured batch size
        optimal_size = min(memory_based_batch_size, base_batch_size)

        # Ensure minimum batch size of 1
        optimal_size = max(1, optimal_size)

        # Log optimization if size was adjusted
        if optimal_size != base_batch_size:
            self.logger.info(
                f"Optimized batch size: {base_batch_size} -> {optimal_size} "
                f"(vector_dim: {vector_size}, estimated_memory_per_point: {estimated_memory_per_point} bytes)"
            )

        return optimal_size

    def batch_upsert_with_retry(
        self,
        collection_name: str,
        points: list[PointStruct],
        batch_size: int | None = None,
        enable_optimization: bool = True,
    ) -> BatchInsertionStats:
        """
        Perform batch upsert with automatic retry, optimization, and comprehensive error handling.

        Args:
            collection_name: Name of the collection to insert into
            points: List of points to insert
            batch_size: Override default batch size (optional)
            enable_optimization: Whether to optimize batch size based on point characteristics

        Returns:
            BatchInsertionStats with detailed insertion statistics
        """
        stats = BatchInsertionStats()
        stats.total_points = len(points)
        start_time = time.time()

        if not points:
            self.logger.warning("No points provided for batch upsert")
            return stats

        # Determine optimal batch size
        if enable_optimization:
            effective_batch_size = self.calculate_optimal_batch_size(points, batch_size)
        else:
            effective_batch_size = batch_size or self.default_batch_size

        stats.average_batch_size = effective_batch_size

        self.logger.info(f"Starting batch upsert for {len(points)} points to {collection_name} " f"(batch_size: {effective_batch_size})")

        # Process points in batches
        for i in range(0, len(points), effective_batch_size):
            batch_points = points[i : i + effective_batch_size]
            batch_num = (i // effective_batch_size) + 1
            stats.batch_count += 1

            # Attempt batch insertion with retry
            success = self._insert_batch_with_retry(collection_name, batch_points, batch_num, stats)

            if success:
                stats.successful_insertions += len(batch_points)
            else:
                # Try individual point insertion for failed batch
                individual_successes = self._retry_individual_points(collection_name, batch_points, batch_num, stats)
                stats.successful_insertions += individual_successes
                stats.failed_insertions += len(batch_points) - individual_successes

            # Memory cleanup after each batch
            del batch_points
            gc.collect()

        stats.total_duration = time.time() - start_time

        # Log final statistics
        self.logger.info(
            f"Batch upsert complete: {stats.successful_insertions}/{stats.total_points} points inserted "
            f"({stats.success_rate:.1f}% success rate, {stats.insertions_per_second:.1f} points/sec)"
        )

        if stats.errors:
            self.logger.warning(f"Encountered {len(stats.errors)} errors during batch insertion")

        return stats

    def _insert_batch_with_retry(
        self,
        collection_name: str,
        batch_points: list[PointStruct],
        batch_num: int,
        stats: BatchInsertionStats,
    ) -> bool:
        """
        Insert a single batch with retry logic.

        Returns:
            True if batch insertion succeeded, False otherwise
        """
        for attempt in range(self.max_retries + 1):
            try:
                batch_start = time.time()
                self.client.upsert(collection_name=collection_name, points=batch_points)
                batch_duration = time.time() - batch_start

                self.logger.debug(
                    f"Batch {batch_num} inserted successfully: {len(batch_points)} points "
                    f"(attempt {attempt + 1}, duration: {batch_duration:.2f}s)"
                )
                return True

            except Exception as e:
                error_msg = f"Batch {batch_num} attempt {attempt + 1} failed: {e}"

                if attempt < self.max_retries:
                    self.logger.warning(f"{error_msg}. Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    self.logger.error(f"{error_msg}. All retry attempts exhausted.")
                    stats.errors.append(error_msg)

        return False

    def _retry_individual_points(
        self,
        collection_name: str,
        batch_points: list[PointStruct],
        batch_num: int,
        stats: BatchInsertionStats,
    ) -> int:
        """
        Retry failed batch by inserting individual points.

        Returns:
            Number of successfully inserted individual points
        """
        self.logger.warning(f"Retrying batch {batch_num} with individual point insertion " f"({len(batch_points)} points)")

        successful_individual = 0

        for point_idx, point in enumerate(batch_points):
            try:
                self.client.upsert(collection_name=collection_name, points=[point])
                successful_individual += 1

            except Exception as e:
                error_msg = f"Individual point insertion failed " f"(batch {batch_num}, point {point_idx + 1}): {e}"
                self.logger.error(error_msg)
                stats.errors.append(error_msg)

        self.logger.info(f"Individual retry for batch {batch_num}: " f"{successful_individual}/{len(batch_points)} points succeeded")

        return successful_individual

    async def get_collection_info(self, collection_name: str, use_cache: bool = True) -> dict[str, Any]:
        """
        Get detailed information about a collection.

        Args:
            collection_name: Name of the collection
            use_cache: Whether to use cache for this operation

        Returns:
            Dictionary with collection statistics and configuration
        """
        # Initialize cache if needed
        await self._initialize_cache()

        # Generate cache key
        cache_key = self._generate_cache_key("collection_info", collection_name)

        # Try to get from cache first
        if use_cache:
            cached_info = await self._get_from_cache(cache_key)
            if cached_info is not None:
                self.logger.debug(f"Collection info cache hit for {collection_name}")
                return cached_info

        try:
            collection_info = self.client.get_collection(collection_name)
            point_count = self.client.count(collection_name).count

            result = {
                "name": collection_name,
                "points_count": point_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.value,
                "status": collection_info.status.value,
                "optimizer_status": collection_info.optimizer_status,
                "indexed_vectors_count": getattr(collection_info, "indexed_vectors_count", 0),
                "cached_at": time.time(),
            }

            # Cache the result
            if use_cache:
                await self._set_in_cache(cache_key, result)
                self.logger.debug(f"Cached collection info for {collection_name}")

            return result

        except Exception as e:
            self.logger.error(f"Failed to get collection info for {collection_name}: {e}")

            # Try to return cached data as fallback if available
            if use_cache:
                cached_info = await self._get_from_cache(cache_key)
                if cached_info is not None:
                    self.logger.warning(f"Returning cached collection info for {collection_name} due to error")
                    cached_info["error"] = str(e)
                    cached_info["fallback"] = True
                    return cached_info

            return {"error": str(e), "collection_name": collection_name}

    async def collection_exists(self, collection_name: str, use_cache: bool = True) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection to check
            use_cache: Whether to use cache for this operation

        Returns:
            True if collection exists, False otherwise
        """
        # Initialize cache if needed
        await self._initialize_cache()

        # Generate cache key
        cache_key = self._generate_cache_key("collection_exists", collection_name)

        # Try to get from cache first
        if use_cache:
            cached_exists = await self._get_from_cache(cache_key)
            if cached_exists is not None:
                self.logger.debug(f"Collection existence cache hit for {collection_name}: {cached_exists}")
                return cached_exists

        try:
            collections = self.client.get_collections()
            existing_names = [col.name for col in collections.collections]
            exists = collection_name in existing_names

            # Cache the result with shorter TTL as existence can change
            if use_cache:
                await self._set_in_cache(cache_key, exists, ttl=self._connection_cache_ttl)
                self.logger.debug(f"Cached collection existence for {collection_name}: {exists}")

            return exists

        except Exception as e:
            self.logger.error(f"Error checking if collection exists: {e}")

            # Try to return cached data as fallback if available
            if use_cache:
                cached_exists = await self._get_from_cache(cache_key)
                if cached_exists is not None:
                    self.logger.warning(f"Returning cached existence for {collection_name} due to error: {cached_exists}")
                    return cached_exists

            return False

    def create_metadata_collection(self, collection_name: str) -> bool:
        """
        Create a collection optimized for metadata storage.

        This creates a collection with minimal vector configuration
        since metadata collections primarily use payload functionality.

        Args:
            collection_name: Name of the metadata collection

        Returns:
            True if collection was created successfully
        """
        try:
            if self.collection_exists(collection_name):
                self.logger.info(f"Metadata collection '{collection_name}' already exists")
                return True

            from qdrant_client.http.models import Distance, VectorParams

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1,  # Minimal vector size for metadata collections
                    distance=Distance.COSINE,
                ),
            )

            self.logger.info(f"Created metadata collection: {collection_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create metadata collection '{collection_name}': {e}")
            return False

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if collection was deleted successfully
        """
        try:
            if not self.collection_exists(collection_name):
                self.logger.info(f"Collection '{collection_name}' does not exist")
                return True

            self.client.delete_collection(collection_name)
            self.logger.info(f"Deleted collection: {collection_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False

    async def list_collections(self, use_cache: bool = True) -> list[dict[str, Any]]:
        """
        List all collections with basic information.

        Args:
            use_cache: Whether to use cache for this operation

        Returns:
            List of dictionaries with collection information
        """
        # Initialize cache if needed
        await self._initialize_cache()

        # Generate cache key for collections list
        cache_key = self._generate_cache_key("collections_list", "all")

        # Try to get from cache first
        if use_cache:
            cached_collections = await self._get_from_cache(cache_key)
            if cached_collections is not None:
                self.logger.debug("Collections list cache hit")
                return cached_collections

        try:
            collections = self.client.get_collections()
            result = []

            for collection in collections.collections:
                try:
                    info = await self.get_collection_info(collection.name, use_cache=use_cache)
                    result.append(info)
                except Exception as e:
                    self.logger.warning(f"Failed to get info for collection '{collection.name}': {e}")
                    result.append({"name": collection.name, "error": str(e)})

            # Cache the result with shorter TTL as collection list can change
            if use_cache:
                await self._set_in_cache(cache_key, result, ttl=self._connection_cache_ttl)
                self.logger.debug("Cached collections list")

            return result

        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}")

            # Try to return cached data as fallback if available
            if use_cache:
                cached_collections = await self._get_from_cache(cache_key)
                if cached_collections is not None:
                    self.logger.warning("Returning cached collections list due to error")
                    return cached_collections

            return []

    async def get_batch_collection_info(self, collection_names: list[str], use_cache: bool = True) -> dict[str, dict[str, Any]]:
        """
        Get information for multiple collections in batch.

        Args:
            collection_names: List of collection names
            use_cache: Whether to use cache for this operation

        Returns:
            Dictionary mapping collection names to their information
        """
        # Initialize cache if needed
        await self._initialize_cache()

        result = {}
        uncached_collections = []

        # Check cache for each collection
        if use_cache:
            for collection_name in collection_names:
                cache_key = self._generate_cache_key("collection_info", collection_name)
                cached_info = await self._get_from_cache(cache_key)
                if cached_info is not None:
                    result[collection_name] = cached_info
                    self.logger.debug(f"Batch collection info cache hit for {collection_name}")
                else:
                    uncached_collections.append(collection_name)
        else:
            uncached_collections = collection_names

        # Fetch uncached collections
        for collection_name in uncached_collections:
            try:
                info = await self.get_collection_info(collection_name, use_cache=use_cache)
                result[collection_name] = info
            except Exception as e:
                self.logger.warning(f"Failed to get batch info for collection '{collection_name}': {e}")
                result[collection_name] = {"name": collection_name, "error": str(e)}

        return result

    async def check_batch_collection_exists(self, collection_names: list[str], use_cache: bool = True) -> dict[str, bool]:
        """
        Check existence for multiple collections in batch.

        Args:
            collection_names: List of collection names
            use_cache: Whether to use cache for this operation

        Returns:
            Dictionary mapping collection names to their existence status
        """
        # Initialize cache if needed
        await self._initialize_cache()

        result = {}
        uncached_collections = []

        # Check cache for each collection
        if use_cache:
            for collection_name in collection_names:
                cache_key = self._generate_cache_key("collection_exists", collection_name)
                cached_exists = await self._get_from_cache(cache_key)
                if cached_exists is not None:
                    result[collection_name] = cached_exists
                    self.logger.debug(f"Batch collection existence cache hit for {collection_name}")
                else:
                    uncached_collections.append(collection_name)
        else:
            uncached_collections = collection_names

        # Fetch uncached collections
        for collection_name in uncached_collections:
            try:
                exists = await self.collection_exists(collection_name, use_cache=use_cache)
                result[collection_name] = exists
            except Exception as e:
                self.logger.warning(f"Failed to check batch existence for collection '{collection_name}': {e}")
                result[collection_name] = False

        return result

    async def get_database_health(self, use_cache: bool = True) -> dict[str, Any]:
        """
        Get database health status with cache fallback.

        Args:
            use_cache: Whether to use cache for this operation

        Returns:
            Dictionary with database health information
        """
        # Initialize cache if needed
        await self._initialize_cache()

        # Generate cache key
        cache_key = self._generate_cache_key("database_health", "status")

        # Try to get from cache first
        if use_cache:
            cached_health = await self._get_from_cache(cache_key)
            if cached_health is not None:
                self.logger.debug("Database health cache hit")
                return cached_health

        try:
            # Test database connection and get basic info
            start_time = time.time()
            collections = self.client.get_collections()
            response_time = time.time() - start_time

            health_status = {
                "status": "healthy",
                "response_time": response_time,
                "collections_count": len(collections.collections),
                "timestamp": time.time(),
                "connection_test": "passed",
            }

            # Cache the result with shorter TTL for health status
            if use_cache:
                await self._set_in_cache(cache_key, health_status, ttl=self._connection_cache_ttl)
                self.logger.debug("Cached database health status")

            return health_status

        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")

            # Try to return cached health data as fallback
            if use_cache:
                cached_health = await self._get_from_cache(cache_key)
                if cached_health is not None:
                    self.logger.warning("Returning cached database health due to connection failure")
                    cached_health["status"] = "unhealthy"
                    cached_health["error"] = str(e)
                    cached_health["fallback"] = True
                    return cached_health

            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time(),
                "connection_test": "failed",
            }

    async def invalidate_collection_cache(self, collection_name: str) -> None:
        """
        Invalidate cache entries for a specific collection.

        Args:
            collection_name: Name of the collection to invalidate
        """
        if not self._cache_enabled or not self.cache_service:
            return

        try:
            # Invalidate all cache entries related to this collection
            cache_keys = [
                self._generate_cache_key("collection_info", collection_name),
                self._generate_cache_key("collection_exists", collection_name),
            ]

            for cache_key in cache_keys:
                await self._delete_from_cache(cache_key)

            # Also invalidate collections list cache
            list_cache_key = self._generate_cache_key("collections_list", "all")
            await self._delete_from_cache(list_cache_key)

            self.logger.debug(f"Invalidated cache for collection: {collection_name}")

        except Exception as e:
            self.logger.warning(f"Failed to invalidate cache for collection {collection_name}: {e}")

    async def invalidate_all_cache(self) -> None:
        """
        Invalidate all Qdrant-related cache entries.
        """
        if not self._cache_enabled or not self.cache_service:
            return

        try:
            # This would ideally use a pattern-based cache invalidation
            # For now, we'll invalidate common cache keys
            await self._delete_from_cache(self._generate_cache_key("collections_list", "all"))
            await self._delete_from_cache(self._generate_cache_key("database_health", "status"))

            self.logger.info("Invalidated all Qdrant cache entries")

        except Exception as e:
            self.logger.warning(f"Failed to invalidate Qdrant cache: {e}")

    async def get_collections_by_pattern(self, pattern: str, use_cache: bool = True) -> list[str]:
        """
        Get collection names matching a pattern.

        Args:
            pattern: Pattern to match (simple string contains)
            use_cache: Whether to use cache for this operation

        Returns:
            List of matching collection names
        """
        # Initialize cache if needed
        await self._initialize_cache()

        # Generate cache key for pattern search
        cache_key = self._generate_cache_key("collections_pattern", pattern)

        # Try to get from cache first
        if use_cache:
            cached_matches = await self._get_from_cache(cache_key)
            if cached_matches is not None:
                self.logger.debug(f"Collections pattern cache hit for pattern: {pattern}")
                return cached_matches

        try:
            collections = self.client.get_collections()
            matching = [col.name for col in collections.collections if pattern in col.name]

            # Cache the result with shorter TTL as collection list can change
            if use_cache:
                await self._set_in_cache(cache_key, matching, ttl=self._connection_cache_ttl)
                self.logger.debug(f"Cached collections pattern for: {pattern}")

            return matching

        except Exception as e:
            self.logger.error(f"Failed to get collections by pattern '{pattern}': {e}")

            # Try to return cached data as fallback if available
            if use_cache:
                cached_matches = await self._get_from_cache(cache_key)
                if cached_matches is not None:
                    self.logger.warning(f"Returning cached pattern matches for '{pattern}' due to error")
                    return cached_matches

            return []

    async def get_collection_schema(self, collection_name: str, use_cache: bool = True) -> dict[str, Any]:
        """
        Get detailed schema information for a collection.

        Args:
            collection_name: Name of the collection
            use_cache: Whether to use cache for this operation

        Returns:
            Dictionary with collection schema information
        """
        # Initialize cache if needed
        await self._initialize_cache()

        # Generate cache key
        cache_key = self._generate_cache_key("collection_schema", collection_name)

        # Try to get from cache first
        if use_cache:
            cached_schema = await self._get_from_cache(cache_key)
            if cached_schema is not None:
                self.logger.debug(f"Collection schema cache hit for {collection_name}")
                return cached_schema

        try:
            collection_info = self.client.get_collection(collection_name)

            schema_info = {
                "name": collection_name,
                "vector_config": {
                    "size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance.value,
                },
                "optimizer_config": collection_info.config.optimizer_config,
                "wal_config": collection_info.config.wal_config,
                "quantization_config": getattr(collection_info.config, "quantization_config", None),
                "hnsw_config": getattr(collection_info.config.params.vectors, "hnsw_config", None),
                "status": collection_info.status.value,
                "cached_at": time.time(),
            }

            # Cache with longer TTL as schema rarely changes
            if use_cache:
                await self._set_in_cache(cache_key, schema_info, ttl=self._cache_ttl * 2)
                self.logger.debug(f"Cached collection schema for {collection_name}")

            return schema_info

        except Exception as e:
            self.logger.error(f"Failed to get collection schema for {collection_name}: {e}")

            # Try to return cached data as fallback if available
            if use_cache:
                cached_schema = await self._get_from_cache(cache_key)
                if cached_schema is not None:
                    self.logger.warning(f"Returning cached schema for {collection_name} due to error")
                    cached_schema["error"] = str(e)
                    cached_schema["fallback"] = True
                    return cached_schema

            return {"error": str(e), "collection_name": collection_name}

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        score_threshold: float | None = None,
        use_cache: bool = True,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Search for similar vectors with caching support.

        Args:
            collection_name: Name of the collection
            query_vector: Query vector for similarity search
            limit: Maximum number of results to return
            score_threshold: Minimum score threshold for results
            use_cache: Whether to use cache for this operation
            **kwargs: Additional search parameters

        Returns:
            List of search results with metadata
        """
        # Initialize cache if needed
        await self._initialize_cache()

        # Generate cache key for search query
        search_params = {
            "collection": collection_name,
            "vector": query_vector,
            "limit": limit,
            "score_threshold": score_threshold,
            **kwargs,
        }
        cache_key = self._generate_cache_key("vector_search", f"{collection_name}_{limit}", **search_params)

        # Try to get from cache first
        if use_cache:
            cached_results = await self._get_from_cache(cache_key)
            if cached_results is not None:
                self.logger.debug(f"Vector search cache hit for collection {collection_name}")
                return cached_results

        try:
            from qdrant_client.http.models import SearchRequest

            # Perform vector search
            search_result = self.client.search(
                collection_name=collection_name, query_vector=query_vector, limit=limit, score_threshold=score_threshold, **kwargs
            )

            # Format results
            results = []
            for scored_point in search_result:
                result = {
                    "id": scored_point.id,
                    "score": scored_point.score,
                    "payload": scored_point.payload,
                    "vector": getattr(scored_point, "vector", None),
                }
                results.append(result)

            # Add metadata
            search_metadata = {
                "results": results,
                "query_params": {
                    "collection": collection_name,
                    "limit": limit,
                    "score_threshold": score_threshold,
                },
                "result_count": len(results),
                "cached_at": time.time(),
            }

            # Cache the results with shorter TTL as data can change
            if use_cache:
                await self._set_in_cache(cache_key, search_metadata, ttl=self._connection_cache_ttl)
                self.logger.debug(f"Cached vector search results for collection {collection_name}")

            return search_metadata

        except Exception as e:
            self.logger.error(f"Vector search failed for collection {collection_name}: {e}")

            # Try to return cached data as fallback if available
            if use_cache:
                cached_results = await self._get_from_cache(cache_key)
                if cached_results is not None:
                    self.logger.warning(f"Returning cached search results for {collection_name} due to error")
                    cached_results["error"] = str(e)
                    cached_results["fallback"] = True
                    return cached_results

            return {
                "results": [],
                "error": str(e),
                "collection": collection_name,
                "result_count": 0,
            }

    async def get_database_config(self, use_cache: bool = True) -> dict[str, Any]:
        """
        Get database configuration with caching.

        Args:
            use_cache: Whether to use cache for this operation

        Returns:
            Dictionary with database configuration
        """
        # Initialize cache if needed
        await self._initialize_cache()

        # Generate cache key
        cache_key = self._generate_cache_key("database_config", "settings")

        # Try to get from cache first
        if use_cache:
            cached_config = await self._get_from_cache(cache_key)
            if cached_config is not None:
                self.logger.debug("Database config cache hit")
                return cached_config

        try:
            # Get basic database info
            collections = self.client.get_collections()

            # Build configuration info
            config_info = {
                "host": getattr(self.client, "host", "unknown"),
                "port": getattr(self.client, "port", "unknown"),
                "collections_count": len(collections.collections),
                "default_batch_size": self.default_batch_size,
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
                "cache_enabled": self._cache_enabled,
                "cache_ttl": self._cache_ttl,
                "cached_at": time.time(),
            }

            # Cache with longer TTL as config rarely changes
            if use_cache:
                await self._set_in_cache(cache_key, config_info, ttl=self._cache_ttl * 2)
                self.logger.debug("Cached database config")

            return config_info

        except Exception as e:
            self.logger.error(f"Failed to get database config: {e}")

            # Try to return cached data as fallback if available
            if use_cache:
                cached_config = await self._get_from_cache(cache_key)
                if cached_config is not None:
                    self.logger.warning("Returning cached database config due to error")
                    cached_config["error"] = str(e)
                    cached_config["fallback"] = True
                    return cached_config

            return {
                "error": str(e),
                "cache_enabled": self._cache_enabled,
                "cached_at": time.time(),
            }

    async def get_connection_pool_stats(self, use_cache: bool = True) -> dict[str, Any]:
        """
        Get connection pool statistics with caching.

        Args:
            use_cache: Whether to use cache for this operation

        Returns:
            Dictionary with connection pool statistics
        """
        # Initialize cache if needed
        await self._initialize_cache()

        # Generate cache key
        cache_key = self._generate_cache_key("connection_pool", "stats")

        # Try to get from cache first
        if use_cache:
            cached_stats = await self._get_from_cache(cache_key)
            if cached_stats is not None:
                self.logger.debug("Connection pool stats cache hit")
                return cached_stats

        try:
            # Get basic connection information
            pool_stats = {
                "host": getattr(self.client, "host", "localhost"),
                "port": getattr(self.client, "port", 6333),
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
                "cache_enabled": self._cache_enabled,
                "cache_ttl": self._cache_ttl,
                "connection_cache_ttl": self._connection_cache_ttl,
                "last_health_check": time.time(),
            }

            # Add cache service info if available
            if self.cache_service:
                cache_health = await self.cache_service.get_health()
                pool_stats["cache_status"] = cache_health.status.value
                pool_stats["redis_connected"] = cache_health.redis_connected
                if cache_health.redis_ping_time:
                    pool_stats["redis_ping_time"] = cache_health.redis_ping_time

            pool_stats["cached_at"] = time.time()

            # Cache with shorter TTL as stats can change frequently
            if use_cache:
                await self._set_in_cache(cache_key, pool_stats, ttl=self._connection_cache_ttl)
                self.logger.debug("Cached connection pool stats")

            return pool_stats

        except Exception as e:
            self.logger.error(f"Failed to get connection pool stats: {e}")

            # Try to return cached data as fallback if available
            if use_cache:
                cached_stats = await self._get_from_cache(cache_key)
                if cached_stats is not None:
                    self.logger.warning("Returning cached connection pool stats due to error")
                    cached_stats["error"] = str(e)
                    cached_stats["fallback"] = True
                    return cached_stats

            return {
                "error": str(e),
                "cache_enabled": self._cache_enabled,
                "cached_at": time.time(),
            }

    async def get_comprehensive_health_status(self, use_cache: bool = True) -> dict[str, Any]:
        """
        Get comprehensive health status including database and cache.

        Args:
            use_cache: Whether to use cache for this operation

        Returns:
            Dictionary with comprehensive health information
        """
        # Initialize cache if needed
        await self._initialize_cache()

        # Generate cache key
        cache_key = self._generate_cache_key("comprehensive_health", "full_status")

        # Try to get from cache first (with shorter TTL for health checks)
        if use_cache:
            cached_health = await self._get_from_cache(cache_key)
            if cached_health is not None:
                self.logger.debug("Comprehensive health status cache hit")
                return cached_health

        health_status = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "components": {},
        }

        # Database health
        try:
            db_health = await self.get_database_health(use_cache=False)  # Fresh check for health
            health_status["components"]["database"] = db_health
        except Exception as e:
            health_status["components"]["database"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["overall_status"] = "degraded"

        # Cache service health
        try:
            if self.cache_service:
                cache_health = await self.cache_service.get_health()
                health_status["components"]["cache"] = {
                    "status": cache_health.status.value,
                    "redis_connected": cache_health.redis_connected,
                    "redis_ping_time": cache_health.redis_ping_time,
                    "memory_usage": cache_health.memory_usage,
                }
            else:
                health_status["components"]["cache"] = {
                    "status": "disabled",
                    "message": "Cache service not initialized",
                }
        except Exception as e:
            health_status["components"]["cache"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            if health_status["overall_status"] == "healthy":
                health_status["overall_status"] = "degraded"

        # Connection pool stats
        try:
            pool_stats = await self.get_connection_pool_stats(use_cache=False)
            health_status["components"]["connection_pool"] = pool_stats
        except Exception as e:
            health_status["components"]["connection_pool"] = {
                "status": "unhealthy",
                "error": str(e),
            }

        # Update overall status based on component health
        if any(comp.get("status") == "unhealthy" for comp in health_status["components"].values()):
            health_status["overall_status"] = "unhealthy"

        # Cache the comprehensive health status with very short TTL
        if use_cache:
            await self._set_in_cache(cache_key, health_status, ttl=30)  # 30 seconds TTL
            self.logger.debug("Cached comprehensive health status")

        return health_status

    def clear_collection(self, collection_name: str) -> bool:
        """
        Clear all points from a collection without deleting the collection.

        Args:
            collection_name: Name of the collection to clear

        Returns:
            True if collection was cleared successfully
        """
        try:
            if not self.collection_exists(collection_name):
                self.logger.warning(f"Collection '{collection_name}' does not exist")
                return False

            # Delete all points by scrolling and deleting in batches
            while True:
                response = self.client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    with_payload=False,
                    with_vectors=False,
                )

                points = response[0]
                if not points:
                    break

                point_ids = [point.id for point in points]
                self.client.delete(collection_name=collection_name, points_selector=point_ids)

            self.logger.info(f"Cleared all points from collection: {collection_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to clear collection '{collection_name}': {e}")
            return False

    def get_metadata_collections(self, project_name: str | None = None) -> list[str]:
        """
        Get metadata collection names, optionally filtered by project.

        Args:
            project_name: Optional project name to filter by

        Returns:
            List of metadata collection names
        """
        metadata_suffix = "_file_metadata"

        if project_name:
            # Look for specific project metadata collection
            pattern = f"project_{project_name}{metadata_suffix}"
            return self.get_collections_by_pattern(pattern)
        else:
            # Get all metadata collections
            return self.get_collections_by_pattern(metadata_suffix)

    def delete_points_by_file_paths(self, collection_name: str, file_paths: list[str]) -> bool:
        """
        Delete points from collection based on file paths.

        Args:
            collection_name: Name of the collection
            file_paths: List of file paths to delete

        Returns:
            True if deletion was successful
        """
        if not file_paths:
            return True

        try:
            if not self.collection_exists(collection_name):
                self.logger.warning(f"Collection '{collection_name}' does not exist")
                return False

            deleted_count = 0

            for file_path in file_paths:
                # Search for points with this file path
                search_result = self.client.scroll(
                    collection_name=collection_name,
                    scroll_filter={"must": [{"key": "file_path", "match": {"value": file_path}}]},
                    limit=100,  # Should be few points per file
                    with_payload=False,
                    with_vectors=False,
                )

                # Collect and delete point IDs
                points_to_delete = [point.id for point in search_result[0]]

                if points_to_delete:
                    self.client.delete(
                        collection_name=collection_name,
                        points_selector=points_to_delete,
                    )
                    deleted_count += len(points_to_delete)
                    self.logger.debug(f"Deleted {len(points_to_delete)} points for file: {file_path}")

            self.logger.info(f"Deleted {deleted_count} points for {len(file_paths)} files from collection '{collection_name}'")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting points by file paths: {e}")
            return False
