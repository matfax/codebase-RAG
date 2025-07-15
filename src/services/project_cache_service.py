"""
Project context cache service for the Codebase RAG MCP Server.

This module provides comprehensive caching for project metadata, detection results,
collection mappings, project statistics, and file filtering results to improve
performance by avoiding repeated project analysis operations.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from config.cache_config import CacheConfig, get_global_cache_config
from utils.cache_key_generator import CacheKeyGenerator, KeyType, get_cache_key_generator

from .cache_service import BaseCacheService, CacheOperationError, get_cache_service


@dataclass
class ProjectMetadata:
    """Project metadata structure for caching."""

    project_name: str
    project_type: str
    directory: str
    project_id: str
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    file_count: int = 0
    total_size_mb: float = 0.0
    language_breakdown: dict[str, int] = field(default_factory=dict)
    relevant_files: list[str] = field(default_factory=list)
    excluded_files: int = 0
    exclusion_rate: float = 0.0
    indexing_complexity: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    git_info: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class ProjectDetectionResult:
    """Project detection result structure for caching."""

    directory: str
    project_id: str
    project_name: str
    project_type: str
    confidence: float = 1.0
    detection_method: str = "file_based"
    cached_at: float = field(default_factory=time.time)
    ttl_seconds: int = 3600  # 1 hour default TTL


@dataclass
class CollectionMapping:
    """Collection mapping structure for caching."""

    project_id: str
    collections: dict[str, str] = field(default_factory=dict)  # collection_type -> collection_name
    collection_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)
    exists_in_qdrant: bool = False


@dataclass
class FileFilteringResult:
    """File filtering result structure for caching."""

    directory: str
    total_examined: int = 0
    included: int = 0
    excluded_by_extension: int = 0
    excluded_by_pattern: int = 0
    excluded_by_gitignore: int = 0
    excluded_by_size: int = 0
    excluded_by_binary: int = 0
    excluded_directories: int = 0
    relevant_files: list[str] = field(default_factory=list)
    configuration: dict[str, Any] = field(default_factory=dict)
    processed_at: float = field(default_factory=time.time)
    processing_time_ms: float = 0.0


class ProjectCacheService:
    """
    Project context cache service with comprehensive project data caching.

    This service provides caching for:
    - Project metadata (name, type, structure, statistics)
    - Project detection results (which project a directory belongs to)
    - Collection mapping (which collections exist for projects)
    - File filtering results (which files match project patterns)
    - Project-wide cache invalidation when project structure changes
    """

    def __init__(self, config: CacheConfig | None = None):
        """
        Initialize the project cache service.

        Args:
            config: Cache configuration instance
        """
        self.config = config or get_global_cache_config()
        self.logger = logging.getLogger(__name__)

        # Cache service and key generator will be initialized async
        self._cache_service: BaseCacheService | None = None
        self._key_generator: CacheKeyGenerator | None = None

        # Default TTL values for different cache types
        self.default_ttls = {
            "project_metadata": 7200,  # 2 hours
            "project_detection": 3600,  # 1 hour
            "collection_mapping": 1800,  # 30 minutes
            "file_filtering": 1800,  # 30 minutes
            "project_statistics": 3600,  # 1 hour
        }

        # Cache key prefixes
        self.key_prefixes = {
            "metadata": "project_meta",
            "detection": "project_detect",
            "collections": "project_collections",
            "filtering": "project_filtering",
            "statistics": "project_stats",
        }

    async def initialize(self) -> None:
        """Initialize the project cache service."""
        try:
            self._cache_service = await get_cache_service()
            self._key_generator = get_cache_key_generator()
            self.logger.info("Project cache service initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize project cache service: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the project cache service."""
        self.logger.info("Project cache service shutdown")

    # Project Metadata Caching

    async def cache_project_metadata(self, project_path: str, metadata: ProjectMetadata, ttl: int | None = None) -> bool:
        """
        Cache project metadata.

        Args:
            project_path: Path to the project
            metadata: Project metadata to cache
            ttl: Time to live in seconds (optional)

        Returns:
            bool: True if cached successfully
        """
        try:
            cache_key = self._generate_project_metadata_key(project_path)
            ttl = ttl or self.default_ttls["project_metadata"]

            # Serialize metadata to dict for caching
            metadata_dict = {
                "project_name": metadata.project_name,
                "project_type": metadata.project_type,
                "directory": metadata.directory,
                "project_id": metadata.project_id,
                "created_at": metadata.created_at,
                "last_updated": metadata.last_updated,
                "file_count": metadata.file_count,
                "total_size_mb": metadata.total_size_mb,
                "language_breakdown": metadata.language_breakdown,
                "relevant_files": metadata.relevant_files,
                "excluded_files": metadata.excluded_files,
                "exclusion_rate": metadata.exclusion_rate,
                "indexing_complexity": metadata.indexing_complexity,
                "recommendations": metadata.recommendations,
                "git_info": metadata.git_info,
                "error": metadata.error,
            }

            await self._cache_service.set(cache_key, metadata_dict, ttl)
            self.logger.debug(f"Cached project metadata for: {project_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cache project metadata for {project_path}: {e}")
            return False

    async def get_project_metadata(self, project_path: str) -> ProjectMetadata | None:
        """
        Get cached project metadata.

        Args:
            project_path: Path to the project

        Returns:
            ProjectMetadata: Cached metadata if found, None otherwise
        """
        try:
            cache_key = self._generate_project_metadata_key(project_path)
            metadata_dict = await self._cache_service.get(cache_key)

            if metadata_dict:
                # Deserialize from dict to ProjectMetadata
                return ProjectMetadata(
                    project_name=metadata_dict["project_name"],
                    project_type=metadata_dict["project_type"],
                    directory=metadata_dict["directory"],
                    project_id=metadata_dict["project_id"],
                    created_at=metadata_dict.get("created_at", time.time()),
                    last_updated=metadata_dict.get("last_updated", time.time()),
                    file_count=metadata_dict.get("file_count", 0),
                    total_size_mb=metadata_dict.get("total_size_mb", 0.0),
                    language_breakdown=metadata_dict.get("language_breakdown", {}),
                    relevant_files=metadata_dict.get("relevant_files", []),
                    excluded_files=metadata_dict.get("excluded_files", 0),
                    exclusion_rate=metadata_dict.get("exclusion_rate", 0.0),
                    indexing_complexity=metadata_dict.get("indexing_complexity", {}),
                    recommendations=metadata_dict.get("recommendations", []),
                    git_info=metadata_dict.get("git_info"),
                    error=metadata_dict.get("error"),
                )
            return None

        except Exception as e:
            self.logger.error(f"Failed to get project metadata for {project_path}: {e}")
            return None

    # Project Detection Result Caching

    async def cache_project_detection(self, directory: str, detection_result: ProjectDetectionResult, ttl: int | None = None) -> bool:
        """
        Cache project detection result.

        Args:
            directory: Directory path
            detection_result: Detection result to cache
            ttl: Time to live in seconds (optional)

        Returns:
            bool: True if cached successfully
        """
        try:
            cache_key = self._generate_project_detection_key(directory)
            ttl = ttl or self.default_ttls["project_detection"]

            # Serialize detection result to dict
            detection_dict = {
                "directory": detection_result.directory,
                "project_id": detection_result.project_id,
                "project_name": detection_result.project_name,
                "project_type": detection_result.project_type,
                "confidence": detection_result.confidence,
                "detection_method": detection_result.detection_method,
                "cached_at": detection_result.cached_at,
                "ttl_seconds": detection_result.ttl_seconds,
            }

            await self._cache_service.set(cache_key, detection_dict, ttl)
            self.logger.debug(f"Cached project detection for: {directory}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cache project detection for {directory}: {e}")
            return False

    async def get_project_detection(self, directory: str) -> ProjectDetectionResult | None:
        """
        Get cached project detection result.

        Args:
            directory: Directory path

        Returns:
            ProjectDetectionResult: Cached detection result if found, None otherwise
        """
        try:
            cache_key = self._generate_project_detection_key(directory)
            detection_dict = await self._cache_service.get(cache_key)

            if detection_dict:
                # Check if cached result is still valid (hasn't exceeded its own TTL)
                cached_at = detection_dict.get("cached_at", 0)
                ttl_seconds = detection_dict.get("ttl_seconds", 3600)

                if time.time() - cached_at > ttl_seconds:
                    # Cache entry expired, remove it
                    await self._cache_service.delete(cache_key)
                    return None

                return ProjectDetectionResult(
                    directory=detection_dict["directory"],
                    project_id=detection_dict["project_id"],
                    project_name=detection_dict["project_name"],
                    project_type=detection_dict["project_type"],
                    confidence=detection_dict.get("confidence", 1.0),
                    detection_method=detection_dict.get("detection_method", "file_based"),
                    cached_at=cached_at,
                    ttl_seconds=ttl_seconds,
                )
            return None

        except Exception as e:
            self.logger.error(f"Failed to get project detection for {directory}: {e}")
            return None

    # Collection Mapping Caching

    async def cache_collection_mapping(self, project_id: str, mapping: CollectionMapping, ttl: int | None = None) -> bool:
        """
        Cache collection mapping for a project.

        Args:
            project_id: Project identifier
            mapping: Collection mapping to cache
            ttl: Time to live in seconds (optional)

        Returns:
            bool: True if cached successfully
        """
        try:
            cache_key = self._generate_collection_mapping_key(project_id)
            ttl = ttl or self.default_ttls["collection_mapping"]

            # Serialize mapping to dict
            mapping_dict = {
                "project_id": mapping.project_id,
                "collections": mapping.collections,
                "collection_stats": mapping.collection_stats,
                "last_updated": mapping.last_updated,
                "exists_in_qdrant": mapping.exists_in_qdrant,
            }

            await self._cache_service.set(cache_key, mapping_dict, ttl)
            self.logger.debug(f"Cached collection mapping for project: {project_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cache collection mapping for {project_id}: {e}")
            return False

    async def get_collection_mapping(self, project_id: str) -> CollectionMapping | None:
        """
        Get cached collection mapping for a project.

        Args:
            project_id: Project identifier

        Returns:
            CollectionMapping: Cached mapping if found, None otherwise
        """
        try:
            cache_key = self._generate_collection_mapping_key(project_id)
            mapping_dict = await self._cache_service.get(cache_key)

            if mapping_dict:
                return CollectionMapping(
                    project_id=mapping_dict["project_id"],
                    collections=mapping_dict.get("collections", {}),
                    collection_stats=mapping_dict.get("collection_stats", {}),
                    last_updated=mapping_dict.get("last_updated", time.time()),
                    exists_in_qdrant=mapping_dict.get("exists_in_qdrant", False),
                )
            return None

        except Exception as e:
            self.logger.error(f"Failed to get collection mapping for {project_id}: {e}")
            return None

    # File Filtering Result Caching

    async def cache_file_filtering_result(self, directory: str, result: FileFilteringResult, ttl: int | None = None) -> bool:
        """
        Cache file filtering result.

        Args:
            directory: Directory path
            result: File filtering result to cache
            ttl: Time to live in seconds (optional)

        Returns:
            bool: True if cached successfully
        """
        try:
            cache_key = self._generate_file_filtering_key(directory)
            ttl = ttl or self.default_ttls["file_filtering"]

            # Serialize result to dict
            result_dict = {
                "directory": result.directory,
                "total_examined": result.total_examined,
                "included": result.included,
                "excluded_by_extension": result.excluded_by_extension,
                "excluded_by_pattern": result.excluded_by_pattern,
                "excluded_by_gitignore": result.excluded_by_gitignore,
                "excluded_by_size": result.excluded_by_size,
                "excluded_by_binary": result.excluded_by_binary,
                "excluded_directories": result.excluded_directories,
                "relevant_files": result.relevant_files,
                "configuration": result.configuration,
                "processed_at": result.processed_at,
                "processing_time_ms": result.processing_time_ms,
            }

            await self._cache_service.set(cache_key, result_dict, ttl)
            self.logger.debug(f"Cached file filtering result for: {directory}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cache file filtering result for {directory}: {e}")
            return False

    async def get_file_filtering_result(self, directory: str) -> FileFilteringResult | None:
        """
        Get cached file filtering result.

        Args:
            directory: Directory path

        Returns:
            FileFilteringResult: Cached result if found, None otherwise
        """
        try:
            cache_key = self._generate_file_filtering_key(directory)
            result_dict = await self._cache_service.get(cache_key)

            if result_dict:
                return FileFilteringResult(
                    directory=result_dict["directory"],
                    total_examined=result_dict.get("total_examined", 0),
                    included=result_dict.get("included", 0),
                    excluded_by_extension=result_dict.get("excluded_by_extension", 0),
                    excluded_by_pattern=result_dict.get("excluded_by_pattern", 0),
                    excluded_by_gitignore=result_dict.get("excluded_by_gitignore", 0),
                    excluded_by_size=result_dict.get("excluded_by_size", 0),
                    excluded_by_binary=result_dict.get("excluded_by_binary", 0),
                    excluded_directories=result_dict.get("excluded_directories", 0),
                    relevant_files=result_dict.get("relevant_files", []),
                    configuration=result_dict.get("configuration", {}),
                    processed_at=result_dict.get("processed_at", time.time()),
                    processing_time_ms=result_dict.get("processing_time_ms", 0.0),
                )
            return None

        except Exception as e:
            self.logger.error(f"Failed to get file filtering result for {directory}: {e}")
            return None

    # Project Statistics Caching

    async def cache_project_statistics(self, project_id: str, statistics: dict[str, Any], ttl: int | None = None) -> bool:
        """
        Cache project statistics.

        Args:
            project_id: Project identifier
            statistics: Project statistics to cache
            ttl: Time to live in seconds (optional)

        Returns:
            bool: True if cached successfully
        """
        try:
            cache_key = self._generate_project_statistics_key(project_id)
            ttl = ttl or self.default_ttls["project_statistics"]

            # Add timestamp to statistics
            stats_with_timestamp = {
                **statistics,
                "cached_at": time.time(),
                "project_id": project_id,
            }

            await self._cache_service.set(cache_key, stats_with_timestamp, ttl)
            self.logger.debug(f"Cached project statistics for: {project_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to cache project statistics for {project_id}: {e}")
            return False

    async def get_project_statistics(self, project_id: str) -> dict[str, Any] | None:
        """
        Get cached project statistics.

        Args:
            project_id: Project identifier

        Returns:
            Dict[str, Any]: Cached statistics if found, None otherwise
        """
        try:
            cache_key = self._generate_project_statistics_key(project_id)
            return await self._cache_service.get(cache_key)

        except Exception as e:
            self.logger.error(f"Failed to get project statistics for {project_id}: {e}")
            return None

    # Project-wide Cache Invalidation

    async def invalidate_project_cache(self, project_id: str) -> bool:
        """
        Invalidate all cache entries for a specific project.

        Args:
            project_id: Project identifier

        Returns:
            bool: True if invalidation was successful
        """
        try:
            # Get all keys for the project
            keys_to_invalidate = self._key_generator.invalidate_keys_by_project(project_id)

            if keys_to_invalidate:
                # Delete cache entries in batch
                results = await self._cache_service.delete_batch(keys_to_invalidate)
                success_count = sum(1 for success in results.values() if success)

                self.logger.info(f"Invalidated {success_count}/{len(keys_to_invalidate)} cache entries for project: {project_id}")

                return success_count > 0
            else:
                self.logger.debug(f"No cache entries found for project: {project_id}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to invalidate project cache for {project_id}: {e}")
            return False

    async def invalidate_directory_cache(self, directory: str) -> bool:
        """
        Invalidate cache entries related to a specific directory.

        Args:
            directory: Directory path

        Returns:
            bool: True if invalidation was successful
        """
        try:
            directory_path = str(Path(directory).resolve())

            # Generate keys that might be associated with this directory
            keys_to_check = [
                self._generate_project_metadata_key(directory_path),
                self._generate_project_detection_key(directory_path),
                self._generate_file_filtering_key(directory_path),
            ]

            # Delete existing keys
            results = await self._cache_service.delete_batch(keys_to_check)
            success_count = sum(1 for success in results.values() if success)

            self.logger.info(f"Invalidated {success_count} cache entries for directory: {directory_path}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to invalidate directory cache for {directory}: {e}")
            return False

    async def invalidate_all_project_caches(self) -> bool:
        """
        Invalidate all project-related cache entries.

        Returns:
            bool: True if invalidation was successful
        """
        try:
            # Get all project-related keys
            patterns = [f"*{prefix}*" for prefix in self.key_prefixes.values()]
            all_keys = []

            for pattern in patterns:
                keys = self._key_generator.invalidate_keys_by_pattern(pattern)
                all_keys.extend(keys)

            if all_keys:
                # Remove duplicates
                unique_keys = list(set(all_keys))

                # Delete in batches
                batch_size = 100
                total_deleted = 0

                for i in range(0, len(unique_keys), batch_size):
                    batch = unique_keys[i : i + batch_size]
                    results = await self._cache_service.delete_batch(batch)
                    total_deleted += sum(1 for success in results.values() if success)

                self.logger.info(f"Invalidated {total_deleted} project cache entries")
                return total_deleted > 0
            else:
                self.logger.debug("No project cache entries found to invalidate")
                return True

        except Exception as e:
            self.logger.error(f"Failed to invalidate all project caches: {e}")
            return False

    # Cache Health and Statistics

    async def get_cache_health(self) -> dict[str, Any]:
        """
        Get health information for the project cache service.

        Returns:
            Dict[str, Any]: Cache health information
        """
        try:
            cache_health = await self._cache_service.get_health()
            key_stats = self._key_generator.get_key_statistics()

            # Filter project-related key statistics
            project_key_count = 0
            for key_type in ["project", "metadata"]:
                project_key_count += key_stats.get("type_distribution", {}).get(key_type, 0)

            return {
                "service_status": "healthy" if cache_health.status.value == "healthy" else "unhealthy",
                "cache_backend_status": cache_health.status.value,
                "redis_connected": cache_health.redis_connected,
                "project_keys_count": project_key_count,
                "total_keys_count": key_stats.get("total_keys", 0),
                "collision_rate": key_stats.get("collision_rate", 0),
                "default_ttls": self.default_ttls,
                "last_check": time.time(),
            }

        except Exception as e:
            self.logger.error(f"Failed to get cache health: {e}")
            return {
                "service_status": "error",
                "error": str(e),
                "last_check": time.time(),
            }

    # Private helper methods for key generation

    def _generate_project_metadata_key(self, project_path: str) -> str:
        """Generate cache key for project metadata."""
        return self._key_generator.generate_project_key(
            project_path=project_path,
            project_metadata={"type": "metadata"},
            additional_params={"cache_type": self.key_prefixes["metadata"]},
        )

    def _generate_project_detection_key(self, directory: str) -> str:
        """Generate cache key for project detection."""
        return self._key_generator.generate_key(
            key_type=KeyType.PROJECT,
            namespace=self.key_prefixes["detection"],
            project_id=self._extract_project_id_from_path(directory),
            content={"directory": directory, "type": "detection"},
        )

    def _generate_collection_mapping_key(self, project_id: str) -> str:
        """Generate cache key for collection mapping."""
        return self._key_generator.generate_key(
            key_type=KeyType.PROJECT,
            namespace=self.key_prefixes["collections"],
            project_id=project_id,
            content={"project_id": project_id, "type": "collections"},
        )

    def _generate_file_filtering_key(self, directory: str) -> str:
        """Generate cache key for file filtering result."""
        return self._key_generator.generate_key(
            key_type=KeyType.PROJECT,
            namespace=self.key_prefixes["filtering"],
            project_id=self._extract_project_id_from_path(directory),
            content={"directory": directory, "type": "filtering"},
        )

    def _generate_project_statistics_key(self, project_id: str) -> str:
        """Generate cache key for project statistics."""
        return self._key_generator.generate_key(
            key_type=KeyType.PROJECT,
            namespace=self.key_prefixes["statistics"],
            project_id=project_id,
            content={"project_id": project_id, "type": "statistics"},
        )

    def _extract_project_id_from_path(self, path: str) -> str:
        """Extract project ID from a file path."""
        return Path(path).name.replace(" ", "_").replace("-", "_")


# Global project cache service instance
_project_cache_service: ProjectCacheService | None = None


async def get_project_cache_service() -> ProjectCacheService:
    """
    Get the global project cache service instance.

    Returns:
        ProjectCacheService: The global project cache service instance
    """
    global _project_cache_service
    if _project_cache_service is None:
        _project_cache_service = ProjectCacheService()
        await _project_cache_service.initialize()
    return _project_cache_service


async def shutdown_project_cache_service() -> None:
    """Shutdown the global project cache service."""
    global _project_cache_service
    if _project_cache_service:
        await _project_cache_service.shutdown()
        _project_cache_service = None
