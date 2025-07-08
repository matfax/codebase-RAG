"""
Cache invalidation service for the Codebase RAG MCP Server.

This module provides comprehensive cache invalidation functionality with support for:
- File change detection integration
- Project-specific invalidation strategies
- Partial invalidation for incremental updates
- Manual cache invalidation tools
- Cascade invalidation for dependent caches
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ..config.cache_config import CacheConfig, get_global_cache_config
from ..models.file_metadata import FileMetadata
from ..services.cache_service import BaseCacheService, get_cache_service
from ..services.change_detector_service import ChangeDetectionResult, ChangeDetectorService, ChangeType
from ..services.embedding_cache_service import EmbeddingCacheService
from ..services.file_cache_service import FileCacheService
from ..services.file_metadata_service import FileMetadataService
from ..services.project_cache_service import ProjectCacheService
from ..services.search_cache_service import SearchCacheService
from ..utils.cache_key_generator import CacheKeyGenerator


class InvalidationReason(Enum):
    """Reasons for cache invalidation."""

    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    FILE_ADDED = "file_added"
    PROJECT_CHANGED = "project_changed"
    MANUAL_INVALIDATION = "manual_invalidation"
    DEPENDENCY_CHANGED = "dependency_changed"
    SYSTEM_UPGRADE = "system_upgrade"
    CACHE_CORRUPTION = "cache_corruption"
    TTL_EXPIRED = "ttl_expired"


class InvalidationStrategy(Enum):
    """Cache invalidation strategies."""

    IMMEDIATE = "immediate"  # Invalidate immediately
    LAZY = "lazy"  # Invalidate on next access
    BATCH = "batch"  # Batch invalidations for efficiency
    SCHEDULED = "scheduled"  # Schedule for later invalidation


@dataclass
class InvalidationEvent:
    """Represents a cache invalidation event."""

    event_id: str
    reason: InvalidationReason
    timestamp: datetime
    affected_keys: list[str]
    affected_files: list[str] = field(default_factory=list)
    project_name: str | None = None
    cascade_level: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "event_id": self.event_id,
            "reason": self.reason.value,
            "timestamp": self.timestamp.isoformat(),
            "affected_keys": self.affected_keys,
            "affected_files": self.affected_files,
            "project_name": self.project_name,
            "cascade_level": self.cascade_level,
            "metadata": self.metadata,
        }


@dataclass
class InvalidationStats:
    """Statistics for cache invalidation operations."""

    total_invalidations: int = 0
    file_based_invalidations: int = 0
    manual_invalidations: int = 0
    cascade_invalidations: int = 0
    keys_invalidated: int = 0
    avg_invalidation_time: float = 0.0
    last_invalidation: datetime | None = None

    def update(self, keys_count: int, duration: float, reason: InvalidationReason) -> None:
        """Update statistics with new invalidation data."""
        self.total_invalidations += 1
        self.keys_invalidated += keys_count
        self.last_invalidation = datetime.now()

        # Update averages
        if self.total_invalidations > 0:
            self.avg_invalidation_time = (self.avg_invalidation_time * (self.total_invalidations - 1) + duration) / self.total_invalidations

        # Update reason-specific counts
        if reason == InvalidationReason.FILE_MODIFIED:
            self.file_based_invalidations += 1
        elif reason == InvalidationReason.MANUAL_INVALIDATION:
            self.manual_invalidations += 1
        elif reason == InvalidationReason.DEPENDENCY_CHANGED:
            self.cascade_invalidations += 1


class CacheInvalidationService:
    """
    Comprehensive cache invalidation service.

    This service manages cache invalidation across all cache types with support for:
    - File-based invalidation triggered by file system changes
    - Project-specific invalidation strategies
    - Cascade invalidation for dependent caches
    - Manual invalidation tools
    - Batch and scheduled invalidation for performance
    """

    def __init__(self, config: CacheConfig | None = None):
        """Initialize the cache invalidation service."""
        self.config = config or get_global_cache_config()
        self.logger = logging.getLogger(__name__)
        self.key_generator = CacheKeyGenerator()

        # Cache services
        self._cache_services: dict[str, BaseCacheService] = {}
        self._embedding_cache: EmbeddingCacheService | None = None
        self._search_cache: SearchCacheService | None = None
        self._project_cache: ProjectCacheService | None = None
        self._file_cache: FileCacheService | None = None

        # File change detection integration
        self._change_detector: ChangeDetectorService | None = None
        self._file_metadata_service: FileMetadataService | None = None

        # Invalidation management
        self._invalidation_queue: asyncio.Queue = asyncio.Queue()
        self._invalidation_task: asyncio.Task | None = None
        self._dependency_map: dict[str, set[str]] = {}
        self._stats = InvalidationStats()

        # File monitoring
        self._file_metadata_cache: dict[str, FileMetadata] = {}
        self._monitored_files: set[str] = set()
        self._project_files: dict[str, set[str]] = {}  # Track files per project

        # Event logging
        self._event_log: list[InvalidationEvent] = []
        self._max_event_log_size = 1000

    async def initialize(self) -> None:
        """Initialize the cache invalidation service."""
        try:
            # Initialize cache services
            await self._initialize_cache_services()

            # Start invalidation worker
            self._invalidation_task = asyncio.create_task(self._invalidation_worker())

            self.logger.info("Cache invalidation service initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize cache invalidation service: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the cache invalidation service."""
        try:
            # Stop invalidation worker
            if self._invalidation_task:
                self._invalidation_task.cancel()
                try:
                    await self._invalidation_task
                except asyncio.CancelledError:
                    pass

            # Process any remaining invalidations
            await self._process_pending_invalidations()

            self.logger.info("Cache invalidation service shutdown successfully")

        except Exception as e:
            self.logger.error(f"Error during cache invalidation service shutdown: {e}")

    async def _initialize_cache_services(self) -> None:
        """Initialize all cache services for invalidation."""
        # Get main cache service
        main_cache = await get_cache_service()
        self._cache_services["main"] = main_cache

        # Initialize file change detection services
        self._file_metadata_service = FileMetadataService()
        self._change_detector = ChangeDetectorService(self._file_metadata_service)

        # Initialize specialized cache services
        self._embedding_cache = EmbeddingCacheService(self.config)
        self._search_cache = SearchCacheService(self.config)
        self._project_cache = ProjectCacheService(self.config)
        self._file_cache = FileCacheService(self.config)

        await self._embedding_cache.initialize()
        await self._search_cache.initialize()
        await self._project_cache.initialize()
        await self._file_cache.initialize()

        # Register cache services
        self._cache_services["embedding"] = self._embedding_cache
        self._cache_services["search"] = self._search_cache
        self._cache_services["project"] = self._project_cache
        self._cache_services["file"] = self._file_cache

    async def invalidate_file_caches(
        self, file_path: str, reason: InvalidationReason = InvalidationReason.FILE_MODIFIED, cascade: bool = True
    ) -> InvalidationEvent:
        """
        Invalidate all caches related to a specific file.

        Args:
            file_path: Path to the file that changed
            reason: Reason for invalidation
            cascade: Whether to cascade invalidation to dependent caches

        Returns:
            InvalidationEvent: Details of the invalidation event
        """
        start_time = time.time()
        affected_keys = []

        try:
            # Generate cache keys for the file
            file_keys = await self._generate_file_cache_keys(file_path)
            affected_keys.extend(file_keys)

            # Invalidate file-specific caches
            await self._invalidate_keys_in_service("file", file_keys)

            # Invalidate embedding caches for chunks from this file
            embedding_keys = await self._generate_embedding_cache_keys(file_path)
            affected_keys.extend(embedding_keys)
            await self._invalidate_keys_in_service("embedding", embedding_keys)

            # Invalidate search caches that might contain results from this file
            search_keys = await self._generate_search_cache_keys(file_path)
            affected_keys.extend(search_keys)
            await self._invalidate_keys_in_service("search", search_keys)

            # Cascade invalidation to dependent caches
            if cascade:
                dependent_keys = await self._get_dependent_keys(file_keys)
                affected_keys.extend(dependent_keys)
                await self._invalidate_dependent_keys(dependent_keys)

            # Create invalidation event
            event = InvalidationEvent(
                event_id=f"file_{int(time.time() * 1000)}",
                reason=reason,
                timestamp=datetime.now(),
                affected_keys=affected_keys,
                affected_files=[file_path],
                metadata={"file_path": file_path, "cascade": cascade},
            )

            # Update statistics
            duration = time.time() - start_time
            self._stats.update(len(affected_keys), duration, reason)

            # Log event
            self._log_invalidation_event(event)

            self.logger.info(f"Invalidated {len(affected_keys)} cache entries for file: {file_path}")

            return event

        except Exception as e:
            self.logger.error(f"Failed to invalidate file caches for {file_path}: {e}")
            raise

    async def invalidate_project_caches(
        self, project_name: str, reason: InvalidationReason = InvalidationReason.PROJECT_CHANGED
    ) -> InvalidationEvent:
        """
        Invalidate all caches related to a specific project.

        Args:
            project_name: Name of the project
            reason: Reason for invalidation

        Returns:
            InvalidationEvent: Details of the invalidation event
        """
        start_time = time.time()
        affected_keys = []

        try:
            # Get all cache keys for the project
            project_keys = await self._generate_project_cache_keys(project_name)
            affected_keys.extend(project_keys)

            # Invalidate across all cache services
            for service_name, cache_service in self._cache_services.items():
                service_keys = [key for key in project_keys if key.startswith(f"{service_name}:")]
                if service_keys:
                    await self._invalidate_keys_in_service(service_name, service_keys)

            # Create invalidation event
            event = InvalidationEvent(
                event_id=f"project_{int(time.time() * 1000)}",
                reason=reason,
                timestamp=datetime.now(),
                affected_keys=affected_keys,
                project_name=project_name,
                metadata={"project_name": project_name},
            )

            # Update statistics
            duration = time.time() - start_time
            self._stats.update(len(affected_keys), duration, reason)

            # Log event
            self._log_invalidation_event(event)

            self.logger.info(f"Invalidated {len(affected_keys)} cache entries for project: {project_name}")

            return event

        except Exception as e:
            self.logger.error(f"Failed to invalidate project caches for {project_name}: {e}")
            raise

    async def invalidate_keys(
        self, keys: list[str], reason: InvalidationReason = InvalidationReason.MANUAL_INVALIDATION, cascade: bool = False
    ) -> InvalidationEvent:
        """
        Invalidate specific cache keys.

        Args:
            keys: List of cache keys to invalidate
            reason: Reason for invalidation
            cascade: Whether to cascade invalidation to dependent caches

        Returns:
            InvalidationEvent: Details of the invalidation event
        """
        start_time = time.time()
        affected_keys = list(keys)

        try:
            # Group keys by cache service
            service_keys = {}
            for key in keys:
                service_name = self._extract_service_name(key)
                if service_name not in service_keys:
                    service_keys[service_name] = []
                service_keys[service_name].append(key)

            # Invalidate keys in each service
            for service_name, service_key_list in service_keys.items():
                await self._invalidate_keys_in_service(service_name, service_key_list)

            # Handle cascade invalidation
            if cascade:
                dependent_keys = await self._get_dependent_keys(keys)
                affected_keys.extend(dependent_keys)
                await self._invalidate_dependent_keys(dependent_keys)

            # Create invalidation event
            event = InvalidationEvent(
                event_id=f"manual_{int(time.time() * 1000)}",
                reason=reason,
                timestamp=datetime.now(),
                affected_keys=affected_keys,
                metadata={"manual": True, "cascade": cascade},
            )

            # Update statistics
            duration = time.time() - start_time
            self._stats.update(len(affected_keys), duration, reason)

            # Log event
            self._log_invalidation_event(event)

            self.logger.info(f"Invalidated {len(affected_keys)} cache entries manually")

            return event

        except Exception as e:
            self.logger.error(f"Failed to invalidate keys manually: {e}")
            raise

    async def invalidate_pattern(
        self, pattern: str, reason: InvalidationReason = InvalidationReason.MANUAL_INVALIDATION
    ) -> InvalidationEvent:
        """
        Invalidate cache keys matching a pattern.

        Args:
            pattern: Pattern to match cache keys (supports wildcards)
            reason: Reason for invalidation

        Returns:
            InvalidationEvent: Details of the invalidation event
        """
        start_time = time.time()
        affected_keys = []

        try:
            # Find matching keys in all cache services
            for service_name, cache_service in self._cache_services.items():
                matching_keys = await self._find_matching_keys(cache_service, pattern)
                if matching_keys:
                    affected_keys.extend(matching_keys)
                    await self._invalidate_keys_in_service(service_name, matching_keys)

            # Create invalidation event
            event = InvalidationEvent(
                event_id=f"pattern_{int(time.time() * 1000)}",
                reason=reason,
                timestamp=datetime.now(),
                affected_keys=affected_keys,
                metadata={"pattern": pattern},
            )

            # Update statistics
            duration = time.time() - start_time
            self._stats.update(len(affected_keys), duration, reason)

            # Log event
            self._log_invalidation_event(event)

            self.logger.info(f"Invalidated {len(affected_keys)} cache entries matching pattern: {pattern}")

            return event

        except Exception as e:
            self.logger.error(f"Failed to invalidate pattern {pattern}: {e}")
            raise

    async def clear_all_caches(self, reason: InvalidationReason = InvalidationReason.MANUAL_INVALIDATION) -> InvalidationEvent:
        """
        Clear all caches across all services.

        Args:
            reason: Reason for clearing

        Returns:
            InvalidationEvent: Details of the invalidation event
        """
        start_time = time.time()

        try:
            # Clear all cache services
            for service_name, cache_service in self._cache_services.items():
                await cache_service.clear()

            # Create invalidation event
            event = InvalidationEvent(
                event_id=f"clear_all_{int(time.time() * 1000)}",
                reason=reason,
                timestamp=datetime.now(),
                affected_keys=["*"],
                metadata={"clear_all": True},
            )

            # Update statistics
            duration = time.time() - start_time
            self._stats.update(1, duration, reason)  # Use 1 as placeholder for "all"

            # Log event
            self._log_invalidation_event(event)

            self.logger.info("Cleared all caches")

            return event

        except Exception as e:
            self.logger.error(f"Failed to clear all caches: {e}")
            raise

    def register_file_dependency(self, file_path: str, dependent_keys: list[str]) -> None:
        """
        Register cache keys that depend on a file.

        Args:
            file_path: Path to the file
            dependent_keys: List of cache keys that depend on this file
        """
        if file_path not in self._dependency_map:
            self._dependency_map[file_path] = set()
        self._dependency_map[file_path].update(dependent_keys)

    def get_invalidation_stats(self) -> InvalidationStats:
        """Get cache invalidation statistics."""
        return self._stats

    async def detect_and_invalidate_changes(
        self, project_name: str, current_files: list[str], project_root: str | None = None
    ) -> tuple[ChangeDetectionResult, list[InvalidationEvent]]:
        """
        Detect file changes and automatically invalidate affected caches.

        Args:
            project_name: Name of the project
            current_files: List of current file paths
            project_root: Optional project root for relative path calculation

        Returns:
            Tuple of (ChangeDetectionResult, list of InvalidationEvents)
        """
        if not self._change_detector:
            raise RuntimeError("Change detector not initialized")

        try:
            # Detect changes
            changes = self._change_detector.detect_changes(project_name, current_files, project_root)
            invalidation_events = []

            # Process changes and invalidate caches
            if changes.has_changes:
                self.logger.info(f"Detected {changes.total_changes} changes for project {project_name}")

                # Handle added files
                for file_change in changes.added_files:
                    event = await self.invalidate_file_caches(file_change.file_path, InvalidationReason.FILE_ADDED, cascade=True)
                    invalidation_events.append(event)

                # Handle modified files
                for file_change in changes.modified_files:
                    event = await self.invalidate_file_caches(file_change.file_path, InvalidationReason.FILE_MODIFIED, cascade=True)
                    invalidation_events.append(event)

                # Handle deleted files
                for file_change in changes.deleted_files:
                    event = await self.invalidate_file_caches(file_change.file_path, InvalidationReason.FILE_DELETED, cascade=True)
                    invalidation_events.append(event)

                # Handle moved files
                for file_change in changes.moved_files:
                    # Invalidate both old and new paths
                    if file_change.old_path:
                        old_event = await self.invalidate_file_caches(file_change.old_path, InvalidationReason.FILE_DELETED, cascade=True)
                        invalidation_events.append(old_event)

                    new_event = await self.invalidate_file_caches(file_change.file_path, InvalidationReason.FILE_ADDED, cascade=True)
                    invalidation_events.append(new_event)

                # Update project file tracking
                self._project_files[project_name] = set(current_files)

            return changes, invalidation_events

        except Exception as e:
            self.logger.error(f"Failed to detect and invalidate changes for project {project_name}: {e}")
            raise

    async def incremental_invalidation_check(
        self, project_name: str, file_paths: list[str], project_root: str | None = None
    ) -> list[InvalidationEvent]:
        """
        Perform incremental invalidation check for specific files.

        Args:
            project_name: Name of the project
            file_paths: List of file paths to check
            project_root: Optional project root

        Returns:
            List of invalidation events
        """
        if not self._file_metadata_service:
            raise RuntimeError("File metadata service not initialized")

        invalidation_events = []

        try:
            # Get stored metadata for these files
            stored_metadata = self._file_metadata_service.get_project_file_metadata(project_name)

            for file_path in file_paths:
                abs_path = str(Path(file_path).resolve())

                # Check if file exists
                if not Path(abs_path).exists():
                    # File was deleted
                    if abs_path in stored_metadata:
                        event = await self.invalidate_file_caches(abs_path, InvalidationReason.FILE_DELETED, cascade=True)
                        invalidation_events.append(event)
                    continue

                # Create current metadata
                try:
                    current_metadata = FileMetadata.from_file_path(abs_path, project_root)
                except Exception as e:
                    self.logger.warning(f"Failed to create metadata for {abs_path}: {e}")
                    continue

                # Check if file has changed
                if abs_path in stored_metadata:
                    stored_meta = stored_metadata[abs_path]
                    if self._has_file_changed(stored_meta, current_metadata):
                        event = await self.invalidate_file_caches(abs_path, InvalidationReason.FILE_MODIFIED, cascade=True)
                        invalidation_events.append(event)
                else:
                    # New file
                    event = await self.invalidate_file_caches(abs_path, InvalidationReason.FILE_ADDED, cascade=True)
                    invalidation_events.append(event)

            return invalidation_events

        except Exception as e:
            self.logger.error(f"Failed incremental invalidation check for project {project_name}: {e}")
            raise

    def register_project_files(self, project_name: str, file_paths: list[str]) -> None:
        """
        Register files for a project for change tracking.

        Args:
            project_name: Name of the project
            file_paths: List of file paths in the project
        """
        self._project_files[project_name] = set(file_paths)
        self.logger.debug(f"Registered {len(file_paths)} files for project {project_name}")

    def get_monitored_projects(self) -> list[str]:
        """Get list of projects being monitored for changes."""
        return list(self._project_files.keys())

    def get_project_files(self, project_name: str) -> set[str]:
        """
        Get files being monitored for a project.

        Args:
            project_name: Name of the project

        Returns:
            Set of file paths for the project
        """
        return self._project_files.get(project_name, set())

    def _has_file_changed(self, stored_metadata: FileMetadata, current_metadata: FileMetadata) -> bool:
        """
        Determine if a file has changed based on metadata comparison.

        Args:
            stored_metadata: Previously stored metadata
            current_metadata: Current file metadata

        Returns:
            True if file has changed
        """
        # Primary check: modification time and size
        if stored_metadata.has_changed(current_metadata.mtime, current_metadata.file_size):
            # Secondary verification: content hash
            return stored_metadata.content_hash != current_metadata.content_hash
        return False

    async def schedule_project_invalidation_check(self, project_name: str, delay_seconds: float = 0.0) -> None:
        """
        Schedule a project-wide invalidation check.

        Args:
            project_name: Name of the project
            delay_seconds: Delay before performing the check
        """
        task = {
            "type": "project_check",
            "project_name": project_name,
            "delay": delay_seconds,
            "reason": InvalidationReason.PROJECT_CHANGED,
        }

        if delay_seconds > 0:
            # Schedule for later
            asyncio.create_task(self._schedule_delayed_task(task, delay_seconds))
        else:
            # Add to immediate queue
            await self._invalidation_queue.put(task)

    async def _schedule_delayed_task(self, task: dict[str, Any], delay_seconds: float) -> None:
        """Schedule a task to be executed after a delay."""
        await asyncio.sleep(delay_seconds)
        await self._invalidation_queue.put(task)

    def get_recent_events(self, count: int = 10) -> list[InvalidationEvent]:
        """
        Get recent invalidation events.

        Args:
            count: Number of recent events to return

        Returns:
            List of recent invalidation events
        """
        return self._event_log[-count:] if count > 0 else []

    async def _invalidation_worker(self) -> None:
        """Background worker for processing invalidation queue."""
        while True:
            try:
                # Wait for invalidation task
                task = await self._invalidation_queue.get()

                # Process the task
                await self._process_invalidation_task(task)

                # Mark task as done
                self._invalidation_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in invalidation worker: {e}")

    async def _process_invalidation_task(self, task: dict[str, Any]) -> None:
        """Process a single invalidation task."""
        task_type = task.get("type")

        if task_type == "file":
            await self.invalidate_file_caches(task["file_path"], task.get("reason", InvalidationReason.FILE_MODIFIED))
        elif task_type == "project":
            await self.invalidate_project_caches(task["project_name"], task.get("reason", InvalidationReason.PROJECT_CHANGED))
        elif task_type == "keys":
            await self.invalidate_keys(task["keys"], task.get("reason", InvalidationReason.MANUAL_INVALIDATION))
        elif task_type == "pattern":
            await self.invalidate_pattern(task["pattern"], task.get("reason", InvalidationReason.MANUAL_INVALIDATION))
        elif task_type == "project_check":
            # Perform project-wide change detection and invalidation
            project_name = task["project_name"]
            if project_name in self._project_files:
                current_files = list(self._project_files[project_name])
                try:
                    changes, events = await self.detect_and_invalidate_changes(project_name, current_files)
                    if changes.has_changes:
                        self.logger.info(f"Processed {len(events)} invalidation events for project {project_name}")
                except Exception as e:
                    self.logger.error(f"Failed to process project check for {project_name}: {e}")
        else:
            self.logger.warning(f"Unknown invalidation task type: {task_type}")

    async def _process_pending_invalidations(self) -> None:
        """Process any pending invalidations in the queue."""
        while not self._invalidation_queue.empty():
            try:
                task = self._invalidation_queue.get_nowait()
                await self._process_invalidation_task(task)
                self._invalidation_queue.task_done()
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                self.logger.error(f"Error processing pending invalidation: {e}")

    async def _invalidate_keys_in_service(self, service_name: str, keys: list[str]) -> None:
        """Invalidate keys in a specific cache service."""
        if service_name not in self._cache_services:
            self.logger.warning(f"Unknown cache service: {service_name}")
            return

        cache_service = self._cache_services[service_name]

        try:
            # Use batch delete if available
            if hasattr(cache_service, "delete_batch"):
                await cache_service.delete_batch(keys)
            else:
                # Fall back to individual deletes
                for key in keys:
                    await cache_service.delete(key)

        except Exception as e:
            self.logger.error(f"Failed to invalidate keys in {service_name}: {e}")

    async def _generate_file_cache_keys(self, file_path: str) -> list[str]:
        """Generate cache keys related to a file."""
        keys = []

        # File processing cache keys
        keys.append(self.key_generator.generate_file_key(file_path))
        keys.append(self.key_generator.generate_parsing_key(file_path))
        keys.append(self.key_generator.generate_chunking_key(file_path))

        # Add metadata key
        keys.append(f"file_metadata:{file_path}")

        return keys

    async def _generate_embedding_cache_keys(self, file_path: str) -> list[str]:
        """Generate embedding cache keys related to a file."""
        keys = []

        # This would typically involve querying the file cache to get chunk information
        # For now, we'll use a pattern-based approach
        keys.append(f"embedding:*:{file_path}:*")

        return keys

    async def _generate_search_cache_keys(self, file_path: str) -> list[str]:
        """Generate search cache keys that might contain results from a file."""
        keys = []

        # Search results might contain chunks from this file
        # This is a conservative approach - we invalidate all search results
        keys.append("search:*")

        return keys

    async def _generate_project_cache_keys(self, project_name: str) -> list[str]:
        """Generate cache keys related to a project."""
        keys = []

        # Project-specific keys
        keys.append(f"project:{project_name}:*")
        keys.append(f"project_info:{project_name}")
        keys.append(f"project_stats:{project_name}")
        keys.append(f"file_filtering:{project_name}")

        return keys

    async def _get_dependent_keys(self, keys: list[str]) -> list[str]:
        """Get keys that depend on the given keys."""
        dependent_keys = []

        for key in keys:
            # Check if any files depend on this key
            for file_path, deps in self._dependency_map.items():
                if key in deps:
                    file_keys = await self._generate_file_cache_keys(file_path)
                    dependent_keys.extend(file_keys)

        return dependent_keys

    async def _invalidate_dependent_keys(self, dependent_keys: list[str]) -> None:
        """Invalidate dependent keys with cascade level tracking."""
        if not dependent_keys:
            return

        # Group keys by service and invalidate
        service_keys = {}
        for key in dependent_keys:
            service_name = self._extract_service_name(key)
            if service_name not in service_keys:
                service_keys[service_name] = []
            service_keys[service_name].append(key)

        for service_name, service_key_list in service_keys.items():
            await self._invalidate_keys_in_service(service_name, service_key_list)

    def _extract_service_name(self, key: str) -> str:
        """Extract service name from cache key."""
        if ":" in key:
            return key.split(":", 1)[0]
        return "main"

    async def _find_matching_keys(self, cache_service: BaseCacheService, pattern: str) -> list[str]:
        """Find keys matching a pattern in a cache service."""
        # This is a simplified implementation
        # In a real implementation, you'd need to scan the cache
        # For now, we'll return an empty list
        return []

    def _log_invalidation_event(self, event: InvalidationEvent) -> None:
        """Log an invalidation event."""
        self._event_log.append(event)

        # Keep log size manageable
        if len(self._event_log) > self._max_event_log_size:
            self._event_log = self._event_log[-self._max_event_log_size // 2 :]

        # Log to file if configured
        if self.config.debug_mode:
            self.logger.debug(f"Invalidation event: {event.to_dict()}")


# Global cache invalidation service instance
_cache_invalidation_service: CacheInvalidationService | None = None


async def get_cache_invalidation_service() -> CacheInvalidationService:
    """
    Get the global cache invalidation service instance.

    Returns:
        CacheInvalidationService: The global cache invalidation service instance
    """
    global _cache_invalidation_service
    if _cache_invalidation_service is None:
        _cache_invalidation_service = CacheInvalidationService()
        await _cache_invalidation_service.initialize()
    return _cache_invalidation_service


async def shutdown_cache_invalidation_service() -> None:
    """Shutdown the global cache invalidation service."""
    global _cache_invalidation_service
    if _cache_invalidation_service:
        await _cache_invalidation_service.shutdown()
        _cache_invalidation_service = None
