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


class ProjectInvalidationScope(Enum):
    """Scope of project-specific invalidation."""

    FILE_ONLY = "file_only"  # Only invalidate file-specific caches
    PROJECT_WIDE = "project_wide"  # Invalidate all project-related caches
    CASCADE = "cascade"  # Invalidate with full cascade to dependent caches
    CONSERVATIVE = "conservative"  # Minimal invalidation to preserve performance
    AGGRESSIVE = "aggressive"  # Broad invalidation to ensure consistency


class ProjectInvalidationTrigger(Enum):
    """Triggers for project-specific invalidation."""

    FILE_CHANGE = "file_change"  # Individual file changes
    BATCH_CHANGES = "batch_changes"  # Multiple file changes
    CONFIG_CHANGE = "config_change"  # Project configuration changes
    DEPENDENCY_CHANGE = "dependency_change"  # Dependency file changes
    MANUAL = "manual"  # Manual user-triggered invalidation
    SCHEDULED = "scheduled"  # Scheduled maintenance invalidation


@dataclass
class ProjectInvalidationPolicy:
    """Project-specific invalidation policy configuration."""

    project_name: str
    scope: ProjectInvalidationScope = ProjectInvalidationScope.CASCADE
    strategy: InvalidationStrategy = InvalidationStrategy.IMMEDIATE
    batch_threshold: int = 5  # Number of changes to trigger batch processing
    delay_seconds: float = 0.0  # Delay before processing invalidation
    file_patterns: list[str] = field(default_factory=list)  # File patterns to monitor
    exclude_patterns: list[str] = field(default_factory=list)  # Patterns to exclude

    # Performance settings
    max_concurrent_invalidations: int = 10
    cascade_depth_limit: int = 3

    # Cache type specific settings
    invalidate_embeddings: bool = True
    invalidate_search: bool = True
    invalidate_project: bool = True
    invalidate_file: bool = True

    # Trigger-specific configurations
    trigger_configs: dict[ProjectInvalidationTrigger, dict[str, Any]] = field(default_factory=dict)

    def should_invalidate_file(self, file_path: str) -> bool:
        """Check if a file should trigger invalidation based on patterns."""
        from fnmatch import fnmatch

        # Check exclude patterns first
        for pattern in self.exclude_patterns:
            if fnmatch(file_path, pattern):
                return False

        # If no include patterns, include all
        if not self.file_patterns:
            return True

        # Check include patterns
        for pattern in self.file_patterns:
            if fnmatch(file_path, pattern):
                return True

        return False

    def get_trigger_config(self, trigger: ProjectInvalidationTrigger) -> dict[str, Any]:
        """Get configuration for a specific trigger."""
        return self.trigger_configs.get(trigger, {})


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

        # Project-specific invalidation policies
        self._project_policies: dict[str, ProjectInvalidationPolicy] = {}
        self._default_policy = self._create_default_policy()

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

    def _create_default_policy(self) -> ProjectInvalidationPolicy:
        """Create default invalidation policy."""
        return ProjectInvalidationPolicy(
            project_name="__default__",
            scope=ProjectInvalidationScope.CASCADE,
            strategy=InvalidationStrategy.IMMEDIATE,
            batch_threshold=5,
            delay_seconds=0.0,
            file_patterns=["*.py", "*.js", "*.ts", "*.java", "*.cpp", "*.c", "*.h", "*.hpp"],
            exclude_patterns=["*.pyc", "*.log", "*.tmp", "__pycache__/*", ".git/*", "node_modules/*"],
        )

    def set_project_invalidation_policy(self, policy: ProjectInvalidationPolicy) -> None:
        """
        Set invalidation policy for a specific project.

        Args:
            policy: Project invalidation policy
        """
        self._project_policies[policy.project_name] = policy
        self.logger.info(f"Set invalidation policy for project: {policy.project_name}")

    def get_project_invalidation_policy(self, project_name: str) -> ProjectInvalidationPolicy:
        """
        Get invalidation policy for a project.

        Args:
            project_name: Name of the project

        Returns:
            Project invalidation policy
        """
        return self._project_policies.get(project_name, self._default_policy)

    def create_project_policy(
        self,
        project_name: str,
        scope: ProjectInvalidationScope = ProjectInvalidationScope.CASCADE,
        strategy: InvalidationStrategy = InvalidationStrategy.IMMEDIATE,
        **kwargs,
    ) -> ProjectInvalidationPolicy:
        """
        Create and register a new project invalidation policy.

        Args:
            project_name: Name of the project
            scope: Invalidation scope
            strategy: Invalidation strategy
            **kwargs: Additional policy parameters

        Returns:
            Created project invalidation policy
        """
        policy = ProjectInvalidationPolicy(
            project_name=project_name,
            scope=scope,
            strategy=strategy,
            **kwargs,
        )
        self.set_project_invalidation_policy(policy)
        return policy

    async def invalidate_file_with_policy(
        self,
        file_path: str,
        project_name: str,
        reason: InvalidationReason = InvalidationReason.FILE_MODIFIED,
        trigger: ProjectInvalidationTrigger = ProjectInvalidationTrigger.FILE_CHANGE,
    ) -> InvalidationEvent | None:
        """
        Invalidate file caches using project-specific policy.

        Args:
            file_path: Path to the file
            project_name: Name of the project
            reason: Reason for invalidation
            trigger: Trigger type for invalidation

        Returns:
            InvalidationEvent if invalidation was performed, None if skipped
        """
        policy = self.get_project_invalidation_policy(project_name)

        # Check if file should be invalidated based on policy
        if not policy.should_invalidate_file(file_path):
            self.logger.debug(f"Skipping invalidation for {file_path} based on policy patterns")
            return None

        start_time = time.time()
        affected_keys = []

        try:
            # Apply strategy-specific logic
            if policy.strategy == InvalidationStrategy.SCHEDULED:
                # Schedule for later processing
                await self._schedule_file_invalidation(file_path, project_name, reason, policy)
                return None

            elif policy.strategy == InvalidationStrategy.LAZY:
                # Mark for lazy invalidation (would be handled on next access)
                self._mark_for_lazy_invalidation(file_path, project_name, reason)
                return None

            # Handle immediate or batch strategies
            cascade = policy.scope in [ProjectInvalidationScope.CASCADE, ProjectInvalidationScope.AGGRESSIVE]

            # Generate cache keys based on policy scope
            if policy.scope == ProjectInvalidationScope.FILE_ONLY:
                affected_keys = await self._generate_file_cache_keys(file_path)
            elif policy.scope in [
                ProjectInvalidationScope.PROJECT_WIDE,
                ProjectInvalidationScope.CASCADE,
                ProjectInvalidationScope.AGGRESSIVE,
            ]:
                affected_keys = await self._generate_comprehensive_cache_keys(file_path, project_name, policy)
            elif policy.scope == ProjectInvalidationScope.CONSERVATIVE:
                affected_keys = await self._generate_minimal_cache_keys(file_path, policy)

            # Perform invalidation based on cache type settings
            await self._invalidate_by_policy(affected_keys, policy, cascade)

            # Create invalidation event
            event = InvalidationEvent(
                event_id=f"policy_{trigger.value}_{int(time.time() * 1000)}",
                reason=reason,
                timestamp=datetime.now(),
                affected_keys=affected_keys,
                affected_files=[file_path],
                project_name=project_name,
                metadata={
                    "policy_scope": policy.scope.value,
                    "policy_strategy": policy.strategy.value,
                    "trigger": trigger.value,
                    "cascade": cascade,
                },
            )

            # Update statistics
            duration = time.time() - start_time
            self._stats.update(len(affected_keys), duration, reason)

            # Log event
            self._log_invalidation_event(event)

            self.logger.info(
                f"Policy-based invalidation for {file_path}: {len(affected_keys)} keys, "
                f"scope={policy.scope.value}, strategy={policy.strategy.value}"
            )

            return event

        except Exception as e:
            self.logger.error(f"Failed policy-based invalidation for {file_path}: {e}")
            raise

    async def invalidate_project_with_policy(
        self,
        project_name: str,
        reason: InvalidationReason = InvalidationReason.PROJECT_CHANGED,
        trigger: ProjectInvalidationTrigger = ProjectInvalidationTrigger.MANUAL,
    ) -> InvalidationEvent:
        """
        Invalidate project caches using project-specific policy.

        Args:
            project_name: Name of the project
            reason: Reason for invalidation
            trigger: Trigger type for invalidation

        Returns:
            InvalidationEvent with invalidation details
        """
        policy = self.get_project_invalidation_policy(project_name)
        start_time = time.time()

        try:
            # Get trigger-specific configuration
            trigger_config = policy.get_trigger_config(trigger)

            # Apply strategy
            if policy.strategy == InvalidationStrategy.SCHEDULED:
                delay = trigger_config.get("delay_seconds", policy.delay_seconds)
                await self.schedule_project_invalidation_check(project_name, delay)

                # Create scheduled event
                event = InvalidationEvent(
                    event_id=f"scheduled_{project_name}_{int(time.time() * 1000)}",
                    reason=reason,
                    timestamp=datetime.now(),
                    affected_keys=["scheduled"],
                    project_name=project_name,
                    metadata={
                        "scheduled": True,
                        "delay_seconds": delay,
                        "trigger": trigger.value,
                    },
                )
                self._log_invalidation_event(event)
                return event

            # Generate keys based on policy scope
            if policy.scope == ProjectInvalidationScope.CONSERVATIVE:
                affected_keys = await self._generate_conservative_project_keys(project_name, policy)
            elif policy.scope == ProjectInvalidationScope.AGGRESSIVE:
                affected_keys = await self._generate_aggressive_project_keys(project_name, policy)
            else:
                affected_keys = await self._generate_project_cache_keys(project_name)

            # Perform invalidation
            await self._invalidate_by_policy(affected_keys, policy, cascade=True)

            # Create event
            event = InvalidationEvent(
                event_id=f"project_policy_{trigger.value}_{int(time.time() * 1000)}",
                reason=reason,
                timestamp=datetime.now(),
                affected_keys=affected_keys,
                project_name=project_name,
                metadata={
                    "policy_scope": policy.scope.value,
                    "policy_strategy": policy.strategy.value,
                    "trigger": trigger.value,
                },
            )

            # Update statistics
            duration = time.time() - start_time
            self._stats.update(len(affected_keys), duration, reason)

            # Log event
            self._log_invalidation_event(event)

            self.logger.info(
                f"Project policy invalidation for {project_name}: {len(affected_keys)} keys, "
                f"scope={policy.scope.value}, trigger={trigger.value}"
            )

            return event

        except Exception as e:
            self.logger.error(f"Failed project policy invalidation for {project_name}: {e}")
            raise

    async def batch_invalidate_with_policy(
        self,
        file_changes: list[tuple[str, InvalidationReason]],
        project_name: str,
        trigger: ProjectInvalidationTrigger = ProjectInvalidationTrigger.BATCH_CHANGES,
    ) -> list[InvalidationEvent]:
        """
        Perform batch invalidation using project policy.

        Args:
            file_changes: List of (file_path, reason) tuples
            project_name: Name of the project
            trigger: Trigger type for invalidation

        Returns:
            List of invalidation events
        """
        policy = self.get_project_invalidation_policy(project_name)
        events = []

        # Filter files based on policy patterns
        filtered_changes = [(file_path, reason) for file_path, reason in file_changes if policy.should_invalidate_file(file_path)]

        if not filtered_changes:
            self.logger.debug(f"No files to invalidate after policy filtering for project {project_name}")
            return events

        try:
            # Handle batch strategy
            if policy.strategy == InvalidationStrategy.BATCH and len(filtered_changes) >= policy.batch_threshold:
                # Process as single batch
                event = await self._process_batch_invalidation(filtered_changes, project_name, policy, trigger)
                events.append(event)
            else:
                # Process individually
                for file_path, reason in filtered_changes:
                    event = await self.invalidate_file_with_policy(file_path, project_name, reason, trigger)
                    if event:
                        events.append(event)

            return events

        except Exception as e:
            self.logger.error(f"Failed batch invalidation for project {project_name}: {e}")
            raise

    async def _process_batch_invalidation(
        self,
        file_changes: list[tuple[str, InvalidationReason]],
        project_name: str,
        policy: ProjectInvalidationPolicy,
        trigger: ProjectInvalidationTrigger,
    ) -> InvalidationEvent:
        """Process a batch of file changes as a single invalidation event."""
        start_time = time.time()
        all_affected_keys = []
        all_files = [file_path for file_path, _ in file_changes]

        # Collect all cache keys for batch processing
        for file_path, _ in file_changes:
            if policy.scope == ProjectInvalidationScope.FILE_ONLY:
                keys = await self._generate_file_cache_keys(file_path)
            else:
                keys = await self._generate_comprehensive_cache_keys(file_path, project_name, policy)
            all_affected_keys.extend(keys)

        # Remove duplicates while preserving order
        seen = set()
        unique_keys = []
        for key in all_affected_keys:
            if key not in seen:
                seen.add(key)
                unique_keys.append(key)

        # Perform batch invalidation
        cascade = policy.scope in [ProjectInvalidationScope.CASCADE, ProjectInvalidationScope.AGGRESSIVE]
        await self._invalidate_by_policy(unique_keys, policy, cascade)

        # Create batch event
        event = InvalidationEvent(
            event_id=f"batch_{trigger.value}_{int(time.time() * 1000)}",
            reason=InvalidationReason.PROJECT_CHANGED,  # Use project change for batch
            timestamp=datetime.now(),
            affected_keys=unique_keys,
            affected_files=all_files,
            project_name=project_name,
            metadata={
                "batch_size": len(file_changes),
                "policy_scope": policy.scope.value,
                "policy_strategy": policy.strategy.value,
                "trigger": trigger.value,
                "unique_keys": len(unique_keys),
            },
        )

        # Update statistics
        duration = time.time() - start_time
        self._stats.update(len(unique_keys), duration, InvalidationReason.PROJECT_CHANGED)

        # Log event
        self._log_invalidation_event(event)

        self.logger.info(
            f"Batch invalidation for {project_name}: {len(file_changes)} files, "
            f"{len(unique_keys)} unique keys, scope={policy.scope.value}"
        )

        return event

    async def _generate_comprehensive_cache_keys(self, file_path: str, project_name: str, policy: ProjectInvalidationPolicy) -> list[str]:
        """Generate comprehensive cache keys based on policy settings."""
        keys = []

        # File-specific keys
        if policy.invalidate_file:
            file_keys = await self._generate_file_cache_keys(file_path)
            keys.extend(file_keys)

        # Embedding keys
        if policy.invalidate_embeddings:
            embedding_keys = await self._generate_embedding_cache_keys(file_path)
            keys.extend(embedding_keys)

        # Search keys
        if policy.invalidate_search:
            search_keys = await self._generate_search_cache_keys(file_path)
            keys.extend(search_keys)

        # Project keys (if scope is project-wide)
        if policy.invalidate_project and policy.scope in [
            ProjectInvalidationScope.PROJECT_WIDE,
            ProjectInvalidationScope.CASCADE,
            ProjectInvalidationScope.AGGRESSIVE,
        ]:
            project_keys = await self._generate_project_cache_keys(project_name)
            keys.extend(project_keys)

        return keys

    async def _generate_minimal_cache_keys(self, file_path: str, policy: ProjectInvalidationPolicy) -> list[str]:
        """Generate minimal cache keys for conservative invalidation."""
        keys = []

        # Only file parsing and chunking keys
        if policy.invalidate_file:
            keys.append(self.key_generator.generate_file_key(file_path))
            keys.append(self.key_generator.generate_parsing_key(file_path))

        return keys

    async def _generate_conservative_project_keys(self, project_name: str, policy: ProjectInvalidationPolicy) -> list[str]:
        """Generate conservative set of project cache keys."""
        keys = []

        # Only core project info
        if policy.invalidate_project:
            keys.append(f"project_info:{project_name}")

        return keys

    async def _generate_aggressive_project_keys(self, project_name: str, policy: ProjectInvalidationPolicy) -> list[str]:
        """Generate aggressive set of project cache keys."""
        keys = await self._generate_project_cache_keys(project_name)

        # Add additional broad patterns for aggressive invalidation
        keys.extend(
            [
                f"*:{project_name}:*",
                "search:*",  # Invalidate all search results
                "embedding:*",  # Invalidate all embeddings (aggressive)
            ]
        )

        return keys

    async def _invalidate_by_policy(self, keys: list[str], policy: ProjectInvalidationPolicy, cascade: bool = False) -> None:
        """Invalidate keys according to policy settings."""
        if not keys:
            return

        # Group keys by service type
        service_keys = {}
        for key in keys:
            service_name = self._extract_service_name(key)

            # Check if this service type should be invalidated
            if service_name == "embedding" and not policy.invalidate_embeddings:
                continue
            if service_name == "search" and not policy.invalidate_search:
                continue
            if service_name == "project" and not policy.invalidate_project:
                continue
            if service_name == "file" and not policy.invalidate_file:
                continue

            if service_name not in service_keys:
                service_keys[service_name] = []
            service_keys[service_name].append(key)

        # Invalidate keys with concurrency limit
        semaphore = asyncio.Semaphore(policy.max_concurrent_invalidations)

        async def invalidate_service_keys(service_name: str, service_key_list: list[str]):
            async with semaphore:
                await self._invalidate_keys_in_service(service_name, service_key_list)

        # Execute invalidations concurrently
        tasks = [invalidate_service_keys(service_name, service_key_list) for service_name, service_key_list in service_keys.items()]

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Handle cascade invalidation with depth limit
        if cascade and policy.cascade_depth_limit > 0:
            await self._cascade_invalidate_with_limit(keys, policy, depth=1)

    async def _cascade_invalidate_with_limit(self, keys: list[str], policy: ProjectInvalidationPolicy, depth: int) -> None:
        """Perform cascade invalidation with depth limit."""
        if depth >= policy.cascade_depth_limit:
            return

        dependent_keys = await self._get_dependent_keys(keys)
        if dependent_keys:
            await self._invalidate_by_policy(dependent_keys, policy, cascade=False)

            # Continue cascade if within depth limit
            if depth + 1 < policy.cascade_depth_limit:
                await self._cascade_invalidate_with_limit(dependent_keys, policy, depth + 1)

    async def _schedule_file_invalidation(
        self,
        file_path: str,
        project_name: str,
        reason: InvalidationReason,
        policy: ProjectInvalidationPolicy,
    ) -> None:
        """Schedule file invalidation based on policy delay."""
        task = {
            "type": "file_policy",
            "file_path": file_path,
            "project_name": project_name,
            "reason": reason,
            "policy": policy,
        }

        if policy.delay_seconds > 0:
            asyncio.create_task(self._schedule_delayed_task(task, policy.delay_seconds))
        else:
            await self._invalidation_queue.put(task)

    def _mark_for_lazy_invalidation(self, file_path: str, project_name: str, reason: InvalidationReason) -> None:
        """Mark file for lazy invalidation (implementation would depend on cache service support)."""
        # This would typically set a flag in the cache service
        # For now, we'll just log it
        self.logger.debug(f"Marked {file_path} for lazy invalidation in project {project_name}")

    def get_project_policy_summary(self, project_name: str) -> dict[str, Any]:
        """
        Get summary of project invalidation policy.

        Args:
            project_name: Name of the project

        Returns:
            Dictionary with policy summary
        """
        policy = self.get_project_invalidation_policy(project_name)
        return {
            "project_name": policy.project_name,
            "scope": policy.scope.value,
            "strategy": policy.strategy.value,
            "batch_threshold": policy.batch_threshold,
            "delay_seconds": policy.delay_seconds,
            "file_patterns": policy.file_patterns,
            "exclude_patterns": policy.exclude_patterns,
            "cache_types": {
                "embeddings": policy.invalidate_embeddings,
                "search": policy.invalidate_search,
                "project": policy.invalidate_project,
                "file": policy.invalidate_file,
            },
            "performance_limits": {
                "max_concurrent_invalidations": policy.max_concurrent_invalidations,
                "cascade_depth_limit": policy.cascade_depth_limit,
            },
        }

    def list_project_policies(self) -> list[str]:
        """Get list of projects with custom invalidation policies."""
        return list(self._project_policies.keys())

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
        elif task_type == "file_policy":
            # Process file invalidation with policy
            try:
                await self.invalidate_file_with_policy(
                    task["file_path"],
                    task["project_name"],
                    task.get("reason", InvalidationReason.FILE_MODIFIED),
                    ProjectInvalidationTrigger.SCHEDULED,
                )
            except Exception as e:
                self.logger.error(f"Failed to process policy file invalidation: {e}")
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
