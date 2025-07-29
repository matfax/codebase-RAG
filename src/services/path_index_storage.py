"""
Path Index Storage and Retrieval Service for Wave 2.0 Task 2.7 - Fast Path Lookup.

This service implements high-performance indexing structures for path storage and retrieval.
It provides multiple index types, fast lookup capabilities, and storage optimization for
efficient path-based operations in the PathRAG system.
"""

import asyncio
import json
import logging
import pickle
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from ..models.relational_path import (
    AnyPath,
    DataFlowPath,
    DependencyPath,
    ExecutionPath,
    PathNode,
    PathType,
    RelationalPathCollection,
)


class IndexType(Enum):
    """Types of indexes available for path storage."""

    BREADCRUMB_INDEX = "breadcrumb"  # Index by breadcrumb paths
    TYPE_INDEX = "type"  # Index by path type
    IMPORTANCE_INDEX = "importance"  # Index by importance score
    COMPLEXITY_INDEX = "complexity"  # Index by complexity metrics
    HYBRID_INDEX = "hybrid"  # Multi-dimensional index
    SEMANTIC_INDEX = "semantic"  # Semantic similarity index


class StorageFormat(Enum):
    """Storage formats for path data."""

    JSON = "json"  # Human-readable JSON format
    PICKLE = "pickle"  # Python pickle format (faster)
    COMPRESSED = "compressed"  # Compressed pickle format
    HYBRID = "hybrid"  # Metadata in JSON, data in pickle


class QueryOperator(Enum):
    """Query operators for path retrieval."""

    EQUALS = "eq"  # Exact match
    CONTAINS = "contains"  # Contains substring
    STARTS_WITH = "starts_with"  # Starts with prefix
    GREATER_THAN = "gt"  # Greater than (for numeric)
    LESS_THAN = "lt"  # Less than (for numeric)
    IN_RANGE = "range"  # Within range (for numeric)
    SIMILAR_TO = "similar"  # Semantic similarity


@dataclass
class IndexEntry:
    """Entry in a path index."""

    # Core identification
    path_id: str  # Path identifier
    index_key: str  # Key used for indexing

    # Path metadata
    path_type: PathType  # Type of path
    node_count: int  # Number of nodes
    importance_score: float = 0.0  # Importance score
    complexity_score: float = 0.0  # Complexity score

    # Storage information
    storage_location: str = ""  # Where path data is stored
    storage_format: StorageFormat = StorageFormat.JSON
    data_size: int = 0  # Size of stored data

    # Index metadata
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class QueryFilter:
    """Filter for path queries."""

    # Filter criteria
    field: str  # Field to filter on
    operator: QueryOperator  # Query operator
    value: Any  # Value to compare against
    secondary_value: Any | None = None  # Secondary value (for ranges)

    # Filter options
    case_sensitive: bool = True  # Case sensitivity for string operations
    fuzzy_threshold: float = 0.8  # Threshold for fuzzy matching

    def matches(self, entry: IndexEntry, path_data: dict[str, Any] | None = None) -> bool:
        """Check if an index entry matches this filter."""
        try:
            # Get the field value
            field_value = self._get_field_value(entry, path_data)

            if field_value is None:
                return False

            # Apply operator
            if self.operator == QueryOperator.EQUALS:
                if isinstance(field_value, str) and isinstance(self.value, str):
                    if self.case_sensitive:
                        return field_value == self.value
                    else:
                        return field_value.lower() == self.value.lower()
                return field_value == self.value

            elif self.operator == QueryOperator.CONTAINS:
                if isinstance(field_value, str) and isinstance(self.value, str):
                    if self.case_sensitive:
                        return self.value in field_value
                    else:
                        return self.value.lower() in field_value.lower()
                return False

            elif self.operator == QueryOperator.STARTS_WITH:
                if isinstance(field_value, str) and isinstance(self.value, str):
                    if self.case_sensitive:
                        return field_value.startswith(self.value)
                    else:
                        return field_value.lower().startswith(self.value.lower())
                return False

            elif self.operator == QueryOperator.GREATER_THAN:
                return field_value > self.value

            elif self.operator == QueryOperator.LESS_THAN:
                return field_value < self.value

            elif self.operator == QueryOperator.IN_RANGE:
                if self.secondary_value is None:
                    return False
                min_val = min(self.value, self.secondary_value)
                max_val = max(self.value, self.secondary_value)
                return min_val <= field_value <= max_val

            elif self.operator == QueryOperator.SIMILAR_TO:
                # Simplified similarity check
                if isinstance(field_value, str) and isinstance(self.value, str):
                    similarity = self._calculate_string_similarity(field_value, self.value)
                    return similarity >= self.fuzzy_threshold
                return False

            return False

        except Exception:
            return False

    def _get_field_value(self, entry: IndexEntry, path_data: dict[str, Any] | None) -> Any:
        """Get the value of the specified field."""
        # Direct entry fields
        if hasattr(entry, self.field):
            return getattr(entry, self.field)

        # Path data fields
        if path_data and self.field in path_data:
            return path_data[self.field]

        return None

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings."""
        if not str1 or not str2:
            return 0.0

        # Simple Levenshtein-based similarity
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 1.0

        distance = self._levenshtein_distance(str1.lower(), str2.lower())
        return 1.0 - (distance / max_len)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


@dataclass
class QueryResult:
    """Result of a path query operation."""

    # Query results
    matching_entries: list[IndexEntry]  # Matching index entries
    total_matches: int  # Total number of matches

    # Query metadata
    query_time_ms: float  # Time taken for query
    index_types_used: list[IndexType]  # Index types utilized
    cache_hits: int = 0  # Number of cache hits

    # Performance metrics
    entries_scanned: int = 0  # Number of entries scanned
    index_efficiency: float = 1.0  # Query efficiency (0-1)

    def get_path_ids(self) -> list[str]:
        """Get list of matching path IDs."""
        return [entry.path_id for entry in self.matching_entries]

    def is_efficient_query(self) -> bool:
        """Check if query was efficient."""
        return self.index_efficiency > 0.5 and self.query_time_ms < 1000


@dataclass
class StorageStats:
    """Statistics about path storage."""

    # Storage metrics
    total_paths_stored: int = 0  # Total paths in storage
    total_storage_size: int = 0  # Total storage size in bytes
    index_storage_size: int = 0  # Size of index data

    # Index statistics
    indexes_created: int = 0  # Number of indexes
    index_hit_rate: float = 0.0  # Index hit rate
    average_query_time_ms: float = 0.0  # Average query time

    # Performance metrics
    storage_operations: int = 0  # Total storage operations
    retrieval_operations: int = 0  # Total retrieval operations
    cache_efficiency: float = 0.0  # Cache efficiency

    # Storage breakdown by type
    storage_by_type: dict[PathType, int] = field(default_factory=dict)
    storage_by_format: dict[StorageFormat, int] = field(default_factory=dict)


class PathIndexStorage:
    """
    High-performance path index storage and retrieval service that provides
    fast lookup capabilities through multiple indexing strategies and
    optimized storage formats.

    Key features:
    - Multiple index types (breadcrumb, type, importance, hybrid)
    - Configurable storage formats (JSON, pickle, compressed)
    - Fast query processing with multiple operators
    - LRU caching for frequently accessed paths
    - Storage optimization and compression
    - Performance monitoring and statistics
    """

    def __init__(
        self,
        storage_directory: str,
        default_storage_format: StorageFormat = StorageFormat.HYBRID,
        enable_caching: bool = True,
        cache_size: int = 1000,
    ):
        """
        Initialize the path index storage service.

        Args:
            storage_directory: Directory for storing path data and indexes
            default_storage_format: Default format for storing paths
            enable_caching: Whether to enable LRU caching
            cache_size: Maximum number of paths to cache
        """
        self.logger = logging.getLogger(__name__)
        self.storage_dir = Path(storage_directory)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.default_format = default_storage_format
        self.enable_caching = enable_caching
        self.cache_size = cache_size

        # Index storage
        self._indexes: dict[IndexType, dict[str, list[IndexEntry]]] = {index_type: defaultdict(list) for index_type in IndexType}

        # Path cache (LRU)
        self._path_cache: dict[str, AnyPath] = {}
        self._cache_access_order: list[str] = []

        # Metadata storage
        self._path_metadata: dict[str, dict[str, Any]] = {}

        # Performance tracking
        self._stats = StorageStats()

        # Load existing indexes
        asyncio.create_task(self._load_existing_indexes())

    async def store_path_collection(self, collection: RelationalPathCollection, storage_format: StorageFormat | None = None) -> bool:
        """
        Store a complete path collection with indexing.

        Args:
            collection: Path collection to store
            storage_format: Optional storage format override

        Returns:
            True if storage successful, False otherwise
        """
        try:
            self.logger.info(f"Storing path collection: {collection.collection_name}")

            format_to_use = storage_format or self.default_format

            # Store all paths from collection
            all_paths = collection.execution_paths + collection.data_flow_paths + collection.dependency_paths

            if not all_paths:
                self.logger.warning("No paths to store in collection")
                return True

            # Store individual paths
            storage_tasks = []
            for path in all_paths:
                task = self.store_path(path, format_to_use)
                storage_tasks.append(task)

            # Execute storage operations
            results = await asyncio.gather(*storage_tasks, return_exceptions=True)

            # Check results
            successful_stores = sum(1 for result in results if result is True)

            if successful_stores == len(all_paths):
                # Store collection metadata
                await self._store_collection_metadata(collection)

                self.logger.info(f"Successfully stored {successful_stores} paths from collection {collection.collection_name}")
                return True
            else:
                failed_stores = len(all_paths) - successful_stores
                self.logger.warning(f"Partially successful: {successful_stores} stored, {failed_stores} failed")
                return False

        except Exception as e:
            self.logger.error(f"Failed to store path collection: {str(e)}")
            return False

    async def store_path(self, path: AnyPath, storage_format: StorageFormat | None = None) -> bool:
        """
        Store a single path with indexing.

        Args:
            path: Path to store
            storage_format: Optional storage format override

        Returns:
            True if storage successful, False otherwise
        """
        try:
            format_to_use = storage_format or self.default_format

            # Store path data
            storage_location = await self._store_path_data(path, format_to_use)
            if not storage_location:
                return False

            # Create index entries
            await self._create_index_entries(path, storage_location, format_to_use)

            # Update cache if enabled
            if self.enable_caching:
                self._update_cache(path.path_id, path)

            # Update statistics
            self._stats.total_paths_stored += 1
            self._stats.storage_operations += 1
            self._stats.storage_by_type[path.path_type] = self._stats.storage_by_type.get(path.path_type, 0) + 1
            self._stats.storage_by_format[format_to_use] = self._stats.storage_by_format.get(format_to_use, 0) + 1

            return True

        except Exception as e:
            self.logger.error(f"Failed to store path {path.path_id}: {str(e)}")
            return False

    async def retrieve_path(self, path_id: str) -> AnyPath | None:
        """
        Retrieve a single path by ID.

        Args:
            path_id: ID of path to retrieve

        Returns:
            Path object if found, None otherwise
        """
        try:
            # Check cache first
            if self.enable_caching and path_id in self._path_cache:
                self._update_cache_access(path_id)
                return self._path_cache[path_id]

            # Find path in indexes
            storage_location = await self._find_path_storage_location(path_id)
            if not storage_location:
                return None

            # Load path data
            path = await self._load_path_data(storage_location)
            if path:
                # Update cache
                if self.enable_caching:
                    self._update_cache(path_id, path)

                # Update access statistics
                await self._update_path_access_stats(path_id)

                self._stats.retrieval_operations += 1

            return path

        except Exception as e:
            self.logger.error(f"Failed to retrieve path {path_id}: {str(e)}")
            return None

    async def query_paths(
        self, filters: list[QueryFilter], limit: int | None = None, offset: int = 0, order_by: str | None = None
    ) -> QueryResult:
        """
        Query paths using filters.

        Args:
            filters: List of query filters
            limit: Maximum number of results
            offset: Number of results to skip
            order_by: Field to order results by

        Returns:
            QueryResult with matching paths
        """
        start_time = time.time()

        try:
            # Determine best index to use
            index_type = self._select_optimal_index(filters)

            # Get candidate entries
            candidate_entries = await self._get_candidate_entries(filters, index_type)

            # Apply filters
            matching_entries = []
            entries_scanned = 0

            for entry in candidate_entries:
                entries_scanned += 1

                # Load path data if needed for complex filters
                path_data = None
                if self._requires_path_data(filters):
                    path_data = await self._load_path_metadata(entry.path_id)

                # Check all filters
                matches_all = True
                for filter_obj in filters:
                    if not filter_obj.matches(entry, path_data):
                        matches_all = False
                        break

                if matches_all:
                    matching_entries.append(entry)

            # Sort results if requested
            if order_by:
                matching_entries = self._sort_entries(matching_entries, order_by)

            # Apply pagination
            total_matches = len(matching_entries)
            if offset > 0:
                matching_entries = matching_entries[offset:]
            if limit is not None:
                matching_entries = matching_entries[:limit]

            # Calculate performance metrics
            query_time_ms = (time.time() - start_time) * 1000
            index_efficiency = 1.0 - (entries_scanned / max(1, len(candidate_entries)))

            # Update statistics
            self._stats.average_query_time_ms = (self._stats.average_query_time_ms + query_time_ms) / 2.0

            result = QueryResult(
                matching_entries=matching_entries,
                total_matches=total_matches,
                query_time_ms=query_time_ms,
                index_types_used=[index_type],
                entries_scanned=entries_scanned,
                index_efficiency=index_efficiency,
            )

            self.logger.debug(f"Query completed: {len(matching_entries)} matches in {query_time_ms:.2f}ms")

            return result

        except Exception as e:
            self.logger.error(f"Query failed: {str(e)}")
            return QueryResult(matching_entries=[], total_matches=0, query_time_ms=(time.time() - start_time) * 1000, index_types_used=[])

    async def get_paths_by_type(self, path_type: PathType) -> list[AnyPath]:
        """
        Get all paths of a specific type.

        Args:
            path_type: Type of paths to retrieve

        Returns:
            List of paths of the specified type
        """
        filters = [QueryFilter("path_type", QueryOperator.EQUALS, path_type)]
        result = await self.query_paths(filters)

        # Load actual path objects
        paths = []
        for entry in result.matching_entries:
            path = await self.retrieve_path(entry.path_id)
            if path:
                paths.append(path)

        return paths

    async def get_paths_by_importance(self, min_importance: float, max_importance: float = 1.0) -> list[AnyPath]:
        """
        Get paths within an importance range.

        Args:
            min_importance: Minimum importance score
            max_importance: Maximum importance score

        Returns:
            List of paths within the importance range
        """
        filters = [QueryFilter("importance_score", QueryOperator.IN_RANGE, min_importance, max_importance)]

        result = await self.query_paths(filters, order_by="importance_score")

        # Load actual path objects
        paths = []
        for entry in result.matching_entries:
            path = await self.retrieve_path(entry.path_id)
            if path:
                paths.append(path)

        return paths

    async def search_paths_by_breadcrumb(self, breadcrumb_pattern: str, fuzzy_match: bool = False) -> list[AnyPath]:
        """
        Search paths by breadcrumb pattern.

        Args:
            breadcrumb_pattern: Pattern to search for
            fuzzy_match: Whether to use fuzzy matching

        Returns:
            List of matching paths
        """
        if fuzzy_match:
            operator = QueryOperator.SIMILAR_TO
        else:
            operator = QueryOperator.CONTAINS

        filters = [QueryFilter("index_key", operator, breadcrumb_pattern)]
        result = await self.query_paths(filters)

        # Load actual path objects
        paths = []
        for entry in result.matching_entries:
            path = await self.retrieve_path(entry.path_id)
            if path:
                paths.append(path)

        return paths

    async def delete_path(self, path_id: str) -> bool:
        """
        Delete a path from storage and indexes.

        Args:
            path_id: ID of path to delete

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            # Find and remove from indexes
            await self._remove_from_indexes(path_id)

            # Remove from cache
            if path_id in self._path_cache:
                del self._path_cache[path_id]
                if path_id in self._cache_access_order:
                    self._cache_access_order.remove(path_id)

            # Remove metadata
            if path_id in self._path_metadata:
                del self._path_metadata[path_id]

            # Remove storage file
            storage_location = await self._find_path_storage_location(path_id)
            if storage_location:
                storage_path = Path(storage_location)
                if storage_path.exists():
                    storage_path.unlink()

            self._stats.total_paths_stored = max(0, self._stats.total_paths_stored - 1)

            return True

        except Exception as e:
            self.logger.error(f"Failed to delete path {path_id}: {str(e)}")
            return False

    async def optimize_indexes(self) -> bool:
        """
        Optimize all indexes for better performance.

        Returns:
            True if optimization successful, False otherwise
        """
        try:
            self.logger.info("Starting index optimization")

            # Rebuild indexes from scratch
            old_indexes = self._indexes.copy()
            self._indexes = {index_type: defaultdict(list) for index_type in IndexType}

            # Rebuild from stored paths
            storage_files = list(self.storage_dir.glob("*.json")) + list(self.storage_dir.glob("*.pkl"))

            for storage_file in storage_files:
                try:
                    path = await self._load_path_data(str(storage_file))
                    if path:
                        await self._create_index_entries(path, str(storage_file), self.default_format)
                except Exception as e:
                    self.logger.warning(f"Failed to reindex {storage_file}: {str(e)}")

            # Save optimized indexes
            await self._save_indexes()

            self.logger.info("Index optimization completed")
            return True

        except Exception as e:
            self.logger.error(f"Index optimization failed: {str(e)}")
            # Restore old indexes
            self._indexes = old_indexes
            return False

    async def get_storage_stats(self) -> StorageStats:
        """
        Get current storage statistics.

        Returns:
            StorageStats object with current statistics
        """
        # Update storage size statistics
        storage_size = 0
        for storage_file in self.storage_dir.iterdir():
            if storage_file.is_file():
                storage_size += storage_file.stat().st_size

        self._stats.total_storage_size = storage_size

        # Calculate cache efficiency
        if self._cache_access_order:
            cache_hits = len(self._path_cache)
            total_accesses = len(self._cache_access_order)
            self._stats.cache_efficiency = cache_hits / total_accesses

        return self._stats

    async def _store_path_data(self, path: AnyPath, storage_format: StorageFormat) -> str | None:
        """Store path data in the specified format."""
        try:
            filename = f"{path.path_id}"

            if storage_format == StorageFormat.JSON:
                file_path = self.storage_dir / f"{filename}.json"
                path_dict = self._path_to_dict(path)

                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(path_dict, f, indent=2, default=str)

                return str(file_path)

            elif storage_format == StorageFormat.PICKLE:
                file_path = self.storage_dir / f"{filename}.pkl"

                with open(file_path, "wb") as f:
                    pickle.dump(path, f, protocol=pickle.HIGHEST_PROTOCOL)

                return str(file_path)

            elif storage_format == StorageFormat.COMPRESSED:
                import gzip

                file_path = self.storage_dir / f"{filename}.pkl.gz"

                with gzip.open(file_path, "wb") as f:
                    pickle.dump(path, f, protocol=pickle.HIGHEST_PROTOCOL)

                return str(file_path)

            elif storage_format == StorageFormat.HYBRID:
                # Metadata in JSON, full data in pickle
                json_path = self.storage_dir / f"{filename}_meta.json"
                pickle_path = self.storage_dir / f"{filename}.pkl"

                # Store metadata
                metadata = {
                    "path_id": path.path_id,
                    "path_type": path.path_type.value,
                    "node_count": len(path.nodes),
                    "data_file": str(pickle_path),
                }

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

                # Store full data
                with open(pickle_path, "wb") as f:
                    pickle.dump(path, f, protocol=pickle.HIGHEST_PROTOCOL)

                return str(pickle_path)

            return None

        except Exception as e:
            self.logger.error(f"Failed to store path data: {str(e)}")
            return None

    async def _load_path_data(self, storage_location: str) -> AnyPath | None:
        """Load path data from storage location."""
        try:
            storage_path = Path(storage_location)

            if not storage_path.exists():
                return None

            if storage_location.endswith(".json"):
                with open(storage_path, encoding="utf-8") as f:
                    path_dict = json.load(f)
                return self._dict_to_path(path_dict)

            elif storage_location.endswith(".pkl"):
                with open(storage_path, "rb") as f:
                    return pickle.load(f)

            elif storage_location.endswith(".pkl.gz"):
                import gzip

                with gzip.open(storage_path, "rb") as f:
                    return pickle.load(f)

            return None

        except Exception as e:
            self.logger.error(f"Failed to load path data from {storage_location}: {str(e)}")
            return None

    async def _create_index_entries(self, path: AnyPath, storage_location: str, storage_format: StorageFormat):
        """Create index entries for a path."""
        # Get path metadata
        importance_score = self._get_path_importance(path)
        complexity_score = self._get_path_complexity(path)

        storage_path = Path(storage_location)
        data_size = storage_path.stat().st_size if storage_path.exists() else 0

        # Create entries for different index types

        # Breadcrumb index
        for node in path.nodes:
            if node.breadcrumb:
                entry = IndexEntry(
                    path_id=path.path_id,
                    index_key=node.breadcrumb,
                    path_type=path.path_type,
                    node_count=len(path.nodes),
                    importance_score=importance_score,
                    complexity_score=complexity_score,
                    storage_location=storage_location,
                    storage_format=storage_format,
                    data_size=data_size,
                )

                self._indexes[IndexType.BREADCRUMB_INDEX][node.breadcrumb].append(entry)

        # Type index
        type_entry = IndexEntry(
            path_id=path.path_id,
            index_key=path.path_type.value,
            path_type=path.path_type,
            node_count=len(path.nodes),
            importance_score=importance_score,
            complexity_score=complexity_score,
            storage_location=storage_location,
            storage_format=storage_format,
            data_size=data_size,
        )

        self._indexes[IndexType.TYPE_INDEX][path.path_type.value].append(type_entry)

        # Importance index (bucketized)
        importance_bucket = self._get_importance_bucket(importance_score)
        importance_entry = IndexEntry(
            path_id=path.path_id,
            index_key=importance_bucket,
            path_type=path.path_type,
            node_count=len(path.nodes),
            importance_score=importance_score,
            complexity_score=complexity_score,
            storage_location=storage_location,
            storage_format=storage_format,
            data_size=data_size,
        )

        self._indexes[IndexType.IMPORTANCE_INDEX][importance_bucket].append(importance_entry)

        # Store path metadata for complex queries
        self._path_metadata[path.path_id] = {
            "path_type": path.path_type.value,
            "node_count": len(path.nodes),
            "importance_score": importance_score,
            "complexity_score": complexity_score,
            "breadcrumbs": [node.breadcrumb for node in path.nodes if node.breadcrumb],
        }

    def _get_path_importance(self, path: AnyPath) -> float:
        """Get importance score for a path."""
        if isinstance(path, ExecutionPath):
            return getattr(path, "criticality_score", 0.5)
        elif isinstance(path, DataFlowPath):
            return getattr(path, "data_quality_score", 0.5)
        elif isinstance(path, DependencyPath):
            return getattr(path, "stability_score", 0.5)
        else:
            if path.nodes:
                return sum(node.importance_score for node in path.nodes) / len(path.nodes)
            return 0.5

    def _get_path_complexity(self, path: AnyPath) -> float:
        """Get complexity score for a path."""
        if hasattr(path, "complexity_score"):
            return getattr(path, "complexity_score", 0.0)
        else:
            # Estimate complexity from path structure
            return min(1.0, len(path.nodes) * 0.1)

    def _get_importance_bucket(self, importance_score: float) -> str:
        """Get importance bucket for indexing."""
        if importance_score >= 0.8:
            return "critical"
        elif importance_score >= 0.6:
            return "high"
        elif importance_score >= 0.4:
            return "medium"
        elif importance_score >= 0.2:
            return "low"
        else:
            return "minimal"

    def _path_to_dict(self, path: AnyPath) -> dict[str, Any]:
        """Convert path to dictionary for JSON storage."""
        return asdict(path)

    def _dict_to_path(self, path_dict: dict[str, Any]) -> AnyPath | None:
        """Convert dictionary to path object."""
        try:
            path_type = PathType(path_dict.get("path_type", "execution_path"))

            if path_type == PathType.EXECUTION_PATH:
                return ExecutionPath(**path_dict)
            elif path_type == PathType.DATA_FLOW:
                return DataFlowPath(**path_dict)
            elif path_type == PathType.DEPENDENCY_PATH:
                return DependencyPath(**path_dict)

            return None

        except Exception as e:
            self.logger.error(f"Failed to convert dict to path: {str(e)}")
            return None

    def _update_cache(self, path_id: str, path: AnyPath):
        """Update LRU cache with path."""
        if not self.enable_caching:
            return

        # Remove if already exists
        if path_id in self._path_cache:
            self._cache_access_order.remove(path_id)

        # Add to cache
        self._path_cache[path_id] = path
        self._cache_access_order.append(path_id)

        # Evict if over limit
        while len(self._path_cache) > self.cache_size:
            oldest_id = self._cache_access_order.pop(0)
            del self._path_cache[oldest_id]

    def _update_cache_access(self, path_id: str):
        """Update cache access order."""
        if path_id in self._cache_access_order:
            self._cache_access_order.remove(path_id)
            self._cache_access_order.append(path_id)

    async def _find_path_storage_location(self, path_id: str) -> str | None:
        """Find storage location for a path ID."""
        # Check all indexes for the path
        for index_type, index_data in self._indexes.items():
            for key, entries in index_data.items():
                for entry in entries:
                    if entry.path_id == path_id:
                        return entry.storage_location

        return None

    async def _load_path_metadata(self, path_id: str) -> dict[str, Any] | None:
        """Load path metadata for complex queries."""
        return self._path_metadata.get(path_id)

    def _select_optimal_index(self, filters: list[QueryFilter]) -> IndexType:
        """Select the most optimal index for the given filters."""
        # Simple heuristic: choose index based on first filter
        if not filters:
            return IndexType.TYPE_INDEX

        first_filter = filters[0]

        if first_filter.field == "path_type":
            return IndexType.TYPE_INDEX
        elif first_filter.field == "importance_score":
            return IndexType.IMPORTANCE_INDEX
        elif first_filter.field in ["index_key", "breadcrumb"]:
            return IndexType.BREADCRUMB_INDEX
        else:
            return IndexType.HYBRID_INDEX

    async def _get_candidate_entries(self, filters: list[QueryFilter], index_type: IndexType) -> list[IndexEntry]:
        """Get candidate entries from the specified index."""
        candidates = []

        if not filters:
            # Return all entries from the index
            for entries_list in self._indexes[index_type].values():
                candidates.extend(entries_list)
            return candidates

        # Get candidates based on first filter
        first_filter = filters[0]

        if first_filter.operator == QueryOperator.EQUALS:
            # Direct lookup
            key = str(first_filter.value)
            candidates.extend(self._indexes[index_type].get(key, []))

        elif first_filter.operator in {QueryOperator.CONTAINS, QueryOperator.STARTS_WITH}:
            # Scan all keys
            search_value = str(first_filter.value).lower()

            for key, entries in self._indexes[index_type].items():
                key_lower = key.lower()

                if first_filter.operator == QueryOperator.CONTAINS:
                    if search_value in key_lower:
                        candidates.extend(entries)
                elif first_filter.operator == QueryOperator.STARTS_WITH:
                    if key_lower.startswith(search_value):
                        candidates.extend(entries)

        else:
            # For other operators, return all entries for filtering
            for entries_list in self._indexes[index_type].values():
                candidates.extend(entries_list)

        return candidates

    def _requires_path_data(self, filters: list[QueryFilter]) -> bool:
        """Check if any filter requires loading full path data."""
        index_entry_fields = {
            "path_id",
            "index_key",
            "path_type",
            "node_count",
            "importance_score",
            "complexity_score",
            "storage_location",
            "storage_format",
            "data_size",
            "created_at",
            "last_accessed",
            "access_count",
        }

        for filter_obj in filters:
            if filter_obj.field not in index_entry_fields:
                return True

        return False

    def _sort_entries(self, entries: list[IndexEntry], order_by: str) -> list[IndexEntry]:
        """Sort entries by the specified field."""
        try:
            reverse = False
            field = order_by

            # Handle descending order
            if order_by.startswith("-"):
                reverse = True
                field = order_by[1:]

            if hasattr(IndexEntry, field):
                return sorted(entries, key=lambda e: getattr(e, field, 0), reverse=reverse)

            return entries

        except Exception:
            return entries

    async def _update_path_access_stats(self, path_id: str):
        """Update access statistics for a path."""
        # Find and update index entries
        for index_type, index_data in self._indexes.items():
            for key, entries in index_data.items():
                for entry in entries:
                    if entry.path_id == path_id:
                        entry.update_access()

    async def _remove_from_indexes(self, path_id: str):
        """Remove path from all indexes."""
        for index_type, index_data in self._indexes.items():
            keys_to_clean = []

            for key, entries in index_data.items():
                # Remove matching entries
                entries[:] = [e for e in entries if e.path_id != path_id]

                # Mark empty keys for removal
                if not entries:
                    keys_to_clean.append(key)

            # Remove empty keys
            for key in keys_to_clean:
                del index_data[key]

    async def _store_collection_metadata(self, collection: RelationalPathCollection):
        """Store collection metadata."""
        metadata_file = self.storage_dir / "collections.json"

        # Load existing metadata
        collections_metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, encoding="utf-8") as f:
                    collections_metadata = json.load(f)
            except Exception:
                pass

        # Add new collection
        collections_metadata[collection.collection_id] = {
            "collection_name": collection.collection_name,
            "total_paths": collection.get_total_path_count(),
            "execution_paths": len(collection.execution_paths),
            "data_flow_paths": len(collection.data_flow_paths),
            "dependency_paths": len(collection.dependency_paths),
            "created_at": time.time(),
        }

        # Save updated metadata
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(collections_metadata, f, indent=2)

    async def _load_existing_indexes(self):
        """Load existing indexes from storage."""
        try:
            index_file = self.storage_dir / "indexes.pkl"
            if index_file.exists():
                with open(index_file, "rb") as f:
                    self._indexes = pickle.load(f)

                self.logger.info("Loaded existing indexes")

            # Load metadata
            metadata_file = self.storage_dir / "metadata.pkl"
            if metadata_file.exists():
                with open(metadata_file, "rb") as f:
                    self._path_metadata = pickle.load(f)

                self.logger.info("Loaded existing metadata")

        except Exception as e:
            self.logger.warning(f"Failed to load existing indexes: {str(e)}")

    async def _save_indexes(self):
        """Save indexes to storage."""
        try:
            # Save indexes
            index_file = self.storage_dir / "indexes.pkl"
            with open(index_file, "wb") as f:
                pickle.dump(self._indexes, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save metadata
            metadata_file = self.storage_dir / "metadata.pkl"
            with open(metadata_file, "wb") as f:
                pickle.dump(self._path_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

            self.logger.debug("Saved indexes and metadata")

        except Exception as e:
            self.logger.error(f"Failed to save indexes: {str(e)}")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            # Save indexes before cleanup
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._save_indexes())
            else:
                loop.run_until_complete(self._save_indexes())
        except Exception:
            pass


# Factory function
def create_path_index_storage(
    storage_directory: str, storage_format: StorageFormat = StorageFormat.HYBRID, enable_caching: bool = True, cache_size: int = 1000
) -> PathIndexStorage:
    """
    Factory function to create a PathIndexStorage instance.

    Args:
        storage_directory: Directory for storing path data
        storage_format: Default storage format
        enable_caching: Whether to enable caching
        cache_size: Maximum cache size

    Returns:
        Configured PathIndexStorage instance
    """
    return PathIndexStorage(storage_directory, storage_format, enable_caching, cache_size)
