"""
File monitoring service for real-time cache invalidation.

This service provides real-time file monitoring capabilities that integrate
with the existing file modification tracking and cache invalidation systems.
It acts as a bridge between file system events and cache invalidation workflows.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ..models.file_metadata import FileMetadata
from ..services.cache_invalidation_service import (
    CacheInvalidationService,
    InvalidationReason,
    ProjectInvalidationTrigger,
    get_cache_invalidation_service,
)
from ..services.change_detector_service import ChangeDetectionResult, ChangeDetectorService, ChangeType
from ..services.file_metadata_service import FileMetadataService
from ..services.file_system_event_handler import (
    FileSystemEventInfo,
    FileSystemEventService,
    FileSystemEventType,
    get_file_system_event_service,
)


class MonitoringMode(Enum):
    """File monitoring modes."""

    POLLING = "polling"  # Regular polling-based monitoring
    EVENTS = "events"  # Real-time file system events only
    HYBRID = "hybrid"  # Combination of polling and system events
    DISABLED = "disabled"  # No monitoring


class MonitoringEventType(Enum):
    """Types of monitoring events."""

    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    FILE_MOVED = "file_moved"
    DIRECTORY_CREATED = "directory_created"
    DIRECTORY_DELETED = "directory_deleted"
    BATCH_CHANGES = "batch_changes"
    MONITORING_ERROR = "monitoring_error"


@dataclass
class MonitoringEvent:
    """Represents a file monitoring event."""

    event_id: str
    event_type: MonitoringEventType
    file_path: str
    timestamp: datetime
    project_name: str | None = None
    old_path: str | None = None  # For move events
    metadata: dict[str, Any] = field(default_factory=dict)
    file_metadata: FileMetadata | None = None


@dataclass
class ProjectMonitoringConfig:
    """Configuration for monitoring a specific project."""

    project_name: str
    root_directory: str
    file_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    polling_interval: float = 5.0  # seconds
    batch_threshold: int = 5  # Number of changes to trigger batch processing
    batch_timeout: float = 2.0  # seconds to wait for batch completion
    enable_real_time: bool = True
    enable_polling: bool = True
    invalidation_delay: float = 0.0  # seconds to delay invalidation
    max_file_size: int = 100 * 1024 * 1024  # 100MB limit

    def should_monitor_file(self, file_path: str) -> bool:
        """Check if a file should be monitored based on patterns."""
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


@dataclass
class MonitoringStats:
    """Statistics for file monitoring operations."""

    total_events: int = 0
    events_by_type: dict[MonitoringEventType, int] = field(default_factory=dict)
    invalidations_triggered: int = 0
    batch_operations: int = 0
    errors: int = 0
    last_event: datetime | None = None
    last_invalidation: datetime | None = None
    monitoring_start_time: datetime | None = None
    processed_files: set[str] = field(default_factory=set)

    def update_event(self, event_type: MonitoringEventType, file_path: str | None = None) -> None:
        """Update statistics with new event."""
        self.total_events += 1
        self.last_event = datetime.now()

        if event_type not in self.events_by_type:
            self.events_by_type[event_type] = 0
        self.events_by_type[event_type] += 1

        if file_path:
            self.processed_files.add(file_path)

        if event_type == MonitoringEventType.MONITORING_ERROR:
            self.errors += 1

    def update_invalidation(self) -> None:
        """Update statistics when invalidation is triggered."""
        self.invalidations_triggered += 1
        self.last_invalidation = datetime.now()

    def update_batch_operation(self) -> None:
        """Update statistics for batch operations."""
        self.batch_operations += 1


class FileMonitoringService:
    """
    Service for real-time file monitoring and cache invalidation integration.

    This service provides:
    - Real-time file change monitoring integration
    - Automatic cache invalidation triggers
    - Project-specific monitoring configurations
    - Batch processing for efficiency
    - Error handling and recovery
    """

    def __init__(
        self,
        invalidation_service: CacheInvalidationService | None = None,
        file_metadata_service: FileMetadataService | None = None,
        change_detector: ChangeDetectorService | None = None,
        event_service: FileSystemEventService | None = None,
    ):
        """Initialize the file monitoring service."""
        self.logger = logging.getLogger(__name__)

        # Service dependencies
        self._invalidation_service = invalidation_service
        self._file_metadata_service = file_metadata_service or FileMetadataService()
        self._change_detector = change_detector or ChangeDetectorService(self._file_metadata_service)
        self._event_service = event_service

        # Monitoring configuration
        self._project_configs: dict[str, ProjectMonitoringConfig] = {}
        self._monitoring_mode = MonitoringMode.HYBRID
        self._is_monitoring = False

        # Event processing
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._batch_queues: dict[str, list[MonitoringEvent]] = {}
        self._batch_timers: dict[str, asyncio.Task] = {}
        self._monitoring_task: asyncio.Task | None = None
        self._polling_tasks: dict[str, asyncio.Task] = {}

        # Statistics and state tracking
        self._stats = MonitoringStats()
        self._last_polling_times: dict[str, float] = {}
        self._file_timestamps: dict[str, dict[str, float]] = {}  # project -> file -> timestamp

        # Error handling
        self._error_count = 0
        self._max_errors = 100
        self._error_backoff = 1.0  # seconds

    async def initialize(self) -> None:
        """Initialize the file monitoring service."""
        try:
            # Get invalidation service if not provided
            if self._invalidation_service is None:
                self._invalidation_service = await get_cache_invalidation_service()

            # Get event service if not provided
            if self._event_service is None:
                self._event_service = await get_file_system_event_service()
                # Set callback for event handling
                self._event_service.set_invalidation_callback(self._handle_file_system_event)

            # Start monitoring worker
            self._monitoring_task = asyncio.create_task(self._monitoring_worker())
            self._stats.monitoring_start_time = datetime.now()

            self.logger.info("File monitoring service initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize file monitoring service: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the file monitoring service."""
        try:
            self._is_monitoring = False

            # Stop monitoring worker
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass

            # Stop polling tasks
            for task in self._polling_tasks.values():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Cancel batch timers
            for timer in self._batch_timers.values():
                timer.cancel()

            # Process any remaining events
            await self._process_pending_events()

            self.logger.info("File monitoring service shutdown successfully")

        except Exception as e:
            self.logger.error(f"Error during file monitoring service shutdown: {e}")

    def add_project_monitoring(self, config: ProjectMonitoringConfig) -> None:
        """
        Add monitoring for a project.

        Args:
            config: Project monitoring configuration
        """
        self._project_configs[config.project_name] = config
        self._last_polling_times[config.project_name] = time.time()
        self._file_timestamps[config.project_name] = {}

        # Start polling task if enabled
        if config.enable_polling and self._monitoring_mode in [MonitoringMode.POLLING, MonitoringMode.HYBRID]:
            self._start_project_polling(config)

        # Start event monitoring if enabled
        if config.enable_real_time and self._monitoring_mode in [MonitoringMode.EVENTS, MonitoringMode.HYBRID]:
            asyncio.create_task(self._start_project_event_monitoring(config))

        self.logger.info(f"Added monitoring for project: {config.project_name}")

    def remove_project_monitoring(self, project_name: str) -> None:
        """
        Remove monitoring for a project.

        Args:
            project_name: Name of the project to stop monitoring
        """
        if project_name in self._project_configs:
            del self._project_configs[project_name]

        if project_name in self._last_polling_times:
            del self._last_polling_times[project_name]

        if project_name in self._file_timestamps:
            del self._file_timestamps[project_name]

        # Stop polling task
        if project_name in self._polling_tasks:
            self._polling_tasks[project_name].cancel()
            del self._polling_tasks[project_name]

        # Stop event monitoring
        if self._event_service:
            asyncio.create_task(self._event_service.stop_monitoring(project_name))

        # Clear batch queue
        if project_name in self._batch_queues:
            del self._batch_queues[project_name]

        # Cancel batch timer
        if project_name in self._batch_timers:
            self._batch_timers[project_name].cancel()
            del self._batch_timers[project_name]

        self.logger.info(f"Removed monitoring for project: {project_name}")

    def create_project_config(
        self,
        project_name: str,
        root_directory: str,
        file_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        **kwargs,
    ) -> ProjectMonitoringConfig:
        """
        Create and register a project monitoring configuration.

        Args:
            project_name: Name of the project
            root_directory: Root directory to monitor
            file_patterns: File patterns to include
            exclude_patterns: File patterns to exclude
            **kwargs: Additional configuration parameters

        Returns:
            Created project monitoring configuration
        """
        default_file_patterns = [
            "*.py",
            "*.js",
            "*.ts",
            "*.tsx",
            "*.jsx",
            "*.java",
            "*.cpp",
            "*.c",
            "*.h",
            "*.hpp",
            "*.rs",
            "*.go",
            "*.json",
            "*.yaml",
            "*.yml",
            "*.md",
        ]

        default_exclude_patterns = [
            "*.pyc",
            "*.log",
            "*.tmp",
            "__pycache__/*",
            ".git/*",
            "node_modules/*",
            ".vscode/*",
            ".idea/*",
            "*.swp",
            "*.swo",
            ".DS_Store",
        ]

        config = ProjectMonitoringConfig(
            project_name=project_name,
            root_directory=root_directory,
            file_patterns=file_patterns or default_file_patterns,
            exclude_patterns=exclude_patterns or default_exclude_patterns,
            **kwargs,
        )

        self.add_project_monitoring(config)
        return config

    async def manual_project_scan(self, project_name: str) -> ChangeDetectionResult:
        """
        Perform manual project scan for changes.

        Args:
            project_name: Name of the project to scan

        Returns:
            Change detection result
        """
        if project_name not in self._project_configs:
            raise ValueError(f"Project {project_name} is not being monitored")

        config = self._project_configs[project_name]

        try:
            # Get current files in project
            current_files = await self._get_project_files(config)

            # Detect changes
            changes = self._change_detector.detect_changes(project_name, current_files, config.root_directory)

            # Trigger invalidations if changes found
            if changes.has_changes:
                await self._process_detected_changes(project_name, changes)

            return changes

        except Exception as e:
            self.logger.error(f"Failed manual project scan for {project_name}: {e}")
            raise

    async def trigger_file_invalidation(
        self,
        file_path: str,
        project_name: str | None = None,
        reason: InvalidationReason = InvalidationReason.FILE_MODIFIED,
        force: bool = False,
    ) -> None:
        """
        Manually trigger cache invalidation for a file.

        Args:
            file_path: Path to the file
            project_name: Project name (auto-detected if None)
            reason: Reason for invalidation
            force: Force invalidation even if file not monitored
        """
        try:
            # Auto-detect project if not provided
            if project_name is None:
                project_name = self._detect_project_for_file(file_path)

            # Check if file should be monitored
            if not force and project_name in self._project_configs:
                config = self._project_configs[project_name]
                if not config.should_monitor_file(file_path):
                    self.logger.debug(f"Skipping invalidation for unmonitored file: {file_path}")
                    return

            # Create monitoring event
            event = MonitoringEvent(
                event_id=f"manual_{int(time.time() * 1000)}",
                event_type=MonitoringEventType.FILE_MODIFIED,
                file_path=file_path,
                timestamp=datetime.now(),
                project_name=project_name,
                metadata={"reason": reason.value, "manual": True, "force": force},
            )

            # Queue for processing
            await self._event_queue.put(event)

        except Exception as e:
            self.logger.error(f"Failed to trigger file invalidation for {file_path}: {e}")
            raise

    def get_monitoring_stats(self) -> MonitoringStats:
        """Get file monitoring statistics."""
        return self._stats

    def get_project_configs(self) -> dict[str, ProjectMonitoringConfig]:
        """Get all project monitoring configurations."""
        return self._project_configs.copy()

    def set_monitoring_mode(self, mode: MonitoringMode) -> None:
        """
        Set monitoring mode.

        Args:
            mode: Monitoring mode to set
        """
        old_mode = self._monitoring_mode
        self._monitoring_mode = mode

        if mode == MonitoringMode.DISABLED:
            # Stop all polling tasks
            for task in self._polling_tasks.values():
                task.cancel()
            self._polling_tasks.clear()
            # Stop all event monitoring
            if self._event_service:
                for project_name in list(self._project_configs.keys()):
                    asyncio.create_task(self._event_service.stop_monitoring(project_name))
        elif mode == MonitoringMode.POLLING:
            # Stop event monitoring, start polling
            if self._event_service:
                for project_name in list(self._project_configs.keys()):
                    asyncio.create_task(self._event_service.stop_monitoring(project_name))
            for config in self._project_configs.values():
                if config.enable_polling:
                    self._start_project_polling(config)
        elif mode == MonitoringMode.EVENTS:
            # Stop polling, start event monitoring
            for task in self._polling_tasks.values():
                task.cancel()
            self._polling_tasks.clear()
            for config in self._project_configs.values():
                if config.enable_real_time:
                    asyncio.create_task(self._start_project_event_monitoring(config))
        elif mode == MonitoringMode.HYBRID:
            # Start both polling and event monitoring as configured
            for config in self._project_configs.values():
                if config.enable_polling:
                    self._start_project_polling(config)
                if config.enable_real_time:
                    asyncio.create_task(self._start_project_event_monitoring(config))

        self.logger.info(f"Monitoring mode changed from {old_mode.value} to {mode.value}")

    async def _monitoring_worker(self) -> None:
        """Background worker for processing monitoring events."""
        self._is_monitoring = True

        while self._is_monitoring:
            try:
                # Wait for monitoring event
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)

                # Process the event
                await self._process_monitoring_event(event)

                # Mark task as done
                self._event_queue.task_done()

            except asyncio.TimeoutError:
                # Normal timeout, continue monitoring
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring worker: {e}")
                self._stats.update_event(MonitoringEventType.MONITORING_ERROR)
                await asyncio.sleep(self._error_backoff)

    async def _process_monitoring_event(self, event: MonitoringEvent) -> None:
        """Process a single monitoring event."""
        try:
            self._stats.update_event(event.event_type, event.file_path)

            if event.project_name and event.project_name in self._project_configs:
                config = self._project_configs[event.project_name]

                # Check if we should batch this event
                if self._should_batch_event(event, config):
                    await self._add_to_batch(event, config)
                    return

            # Process event immediately
            await self._process_event_immediately(event)

        except Exception as e:
            self.logger.error(f"Failed to process monitoring event {event.event_id}: {e}")
            self._stats.update_event(MonitoringEventType.MONITORING_ERROR)

    def _should_batch_event(self, event: MonitoringEvent, config: ProjectMonitoringConfig) -> bool:
        """Determine if an event should be batched."""
        # Don't batch delete events or high-priority events
        if event.event_type in [MonitoringEventType.FILE_DELETED, MonitoringEventType.DIRECTORY_DELETED]:
            return False

        # Don't batch if batch threshold is 1
        if config.batch_threshold <= 1:
            return False

        # Don't batch manual events unless explicitly requested
        if event.metadata.get("manual", False) and not event.metadata.get("allow_batch", False):
            return False

        return True

    async def _add_to_batch(self, event: MonitoringEvent, config: ProjectMonitoringConfig) -> None:
        """Add event to batch queue."""
        project_name = event.project_name
        if not project_name:
            return

        # Initialize batch queue if needed
        if project_name not in self._batch_queues:
            self._batch_queues[project_name] = []

        # Add event to batch
        self._batch_queues[project_name].append(event)

        # Check if batch threshold reached
        if len(self._batch_queues[project_name]) >= config.batch_threshold:
            await self._process_batch(project_name, config)
        else:
            # Set/reset batch timer
            if project_name in self._batch_timers:
                self._batch_timers[project_name].cancel()

            self._batch_timers[project_name] = asyncio.create_task(self._batch_timer(project_name, config.batch_timeout))

    async def _batch_timer(self, project_name: str, timeout: float) -> None:
        """Timer for batch processing."""
        try:
            await asyncio.sleep(timeout)
            if project_name in self._project_configs:
                config = self._project_configs[project_name]
                await self._process_batch(project_name, config)
        except asyncio.CancelledError:
            pass

    async def _process_batch(self, project_name: str, config: ProjectMonitoringConfig) -> None:
        """Process a batch of events."""
        if project_name not in self._batch_queues or not self._batch_queues[project_name]:
            return

        events = self._batch_queues[project_name].copy()
        self._batch_queues[project_name].clear()

        # Cancel timer if it exists
        if project_name in self._batch_timers:
            self._batch_timers[project_name].cancel()
            del self._batch_timers[project_name]

        try:
            # Group events by file path and type
            file_changes = []
            for event in events:
                reason = self._get_invalidation_reason_from_event(event)
                file_changes.append((event.file_path, reason))

            # Trigger batch invalidation
            if file_changes and self._invalidation_service:
                invalidation_events = await self._invalidation_service.batch_invalidate_with_policy(
                    file_changes, project_name, ProjectInvalidationTrigger.BATCH_CHANGES
                )

                self._stats.update_batch_operation()
                self._stats.invalidations_triggered += len(invalidation_events)

                self.logger.info(f"Processed batch of {len(events)} events for project {project_name}")

        except Exception as e:
            self.logger.error(f"Failed to process batch for project {project_name}: {e}")

    async def _process_event_immediately(self, event: MonitoringEvent) -> None:
        """Process an event immediately without batching."""
        try:
            if not self._invalidation_service:
                self.logger.warning("No invalidation service available")
                return

            reason = self._get_invalidation_reason_from_event(event)

            if event.project_name:
                # Use project-specific invalidation
                invalidation_event = await self._invalidation_service.invalidate_file_with_policy(
                    event.file_path, event.project_name, reason, ProjectInvalidationTrigger.FILE_CHANGE
                )
            else:
                # Fall back to basic file invalidation
                invalidation_event = await self._invalidation_service.invalidate_file_caches(event.file_path, reason, cascade=True)

            if invalidation_event:
                self._stats.update_invalidation()

        except Exception as e:
            self.logger.error(f"Failed to process event immediately: {e}")

    def _get_invalidation_reason_from_event(self, event: MonitoringEvent) -> InvalidationReason:
        """Map monitoring event to invalidation reason."""
        event_mapping = {
            MonitoringEventType.FILE_CREATED: InvalidationReason.FILE_ADDED,
            MonitoringEventType.FILE_MODIFIED: InvalidationReason.FILE_MODIFIED,
            MonitoringEventType.FILE_DELETED: InvalidationReason.FILE_DELETED,
            MonitoringEventType.FILE_MOVED: InvalidationReason.FILE_MODIFIED,
        }
        return event_mapping.get(event.event_type, InvalidationReason.FILE_MODIFIED)

    def _start_project_polling(self, config: ProjectMonitoringConfig) -> None:
        """Start polling task for a project."""
        if config.project_name in self._polling_tasks:
            # Cancel existing task
            self._polling_tasks[config.project_name].cancel()

        self._polling_tasks[config.project_name] = asyncio.create_task(self._project_polling_worker(config))

    async def _project_polling_worker(self, config: ProjectMonitoringConfig) -> None:
        """Polling worker for a specific project."""
        project_name = config.project_name

        while self._is_monitoring and project_name in self._project_configs:
            try:
                # Wait for polling interval
                await asyncio.sleep(config.polling_interval)

                # Skip if monitoring is disabled
                if self._monitoring_mode == MonitoringMode.DISABLED:
                    continue

                # Perform project scan
                current_time = time.time()
                current_files = await self._get_project_files(config)

                # Check for changes since last poll
                changes_detected = await self._check_polling_changes(config, current_files)

                if changes_detected:
                    self.logger.debug(f"Polling detected changes in project {project_name}")

                self._last_polling_times[project_name] = current_time

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in polling worker for project {project_name}: {e}")
                await asyncio.sleep(config.polling_interval)

    async def _get_project_files(self, config: ProjectMonitoringConfig) -> list[str]:
        """Get list of files in a project directory."""
        files = []
        root_path = Path(config.root_directory)

        if not root_path.exists():
            self.logger.warning(f"Project root directory does not exist: {config.root_directory}")
            return files

        try:
            for file_path in root_path.rglob("*"):
                if file_path.is_file():
                    abs_path = str(file_path.resolve())

                    # Check if file should be monitored
                    if config.should_monitor_file(abs_path):
                        # Check file size limit
                        if file_path.stat().st_size <= config.max_file_size:
                            files.append(abs_path)

        except Exception as e:
            self.logger.error(f"Error scanning project directory {config.root_directory}: {e}")

        return files

    async def _check_polling_changes(self, config: ProjectMonitoringConfig, current_files: list[str]) -> bool:
        """Check for changes during polling."""
        project_name = config.project_name
        changes_detected = False

        try:
            # Get stored file timestamps for this project
            stored_timestamps = self._file_timestamps.get(project_name, {})
            current_timestamps = {}

            # Check each current file
            for file_path in current_files:
                try:
                    file_stat = Path(file_path).stat()
                    current_mtime = file_stat.st_mtime
                    current_timestamps[file_path] = current_mtime

                    # Check if file is new or modified
                    if file_path not in stored_timestamps:
                        # New file
                        await self._queue_monitoring_event(MonitoringEventType.FILE_CREATED, file_path, project_name)
                        changes_detected = True
                    elif current_mtime > stored_timestamps[file_path]:
                        # Modified file
                        await self._queue_monitoring_event(MonitoringEventType.FILE_MODIFIED, file_path, project_name)
                        changes_detected = True

                except Exception as e:
                    self.logger.warning(f"Error checking file {file_path}: {e}")

            # Check for deleted files
            for file_path in stored_timestamps:
                if file_path not in current_timestamps:
                    # Deleted file
                    await self._queue_monitoring_event(MonitoringEventType.FILE_DELETED, file_path, project_name)
                    changes_detected = True

            # Update stored timestamps
            self._file_timestamps[project_name] = current_timestamps

        except Exception as e:
            self.logger.error(f"Error checking polling changes for project {project_name}: {e}")

        return changes_detected

    async def _queue_monitoring_event(self, event_type: MonitoringEventType, file_path: str, project_name: str | None, **kwargs) -> None:
        """Queue a monitoring event for processing."""
        event = MonitoringEvent(
            event_id=f"poll_{event_type.value}_{int(time.time() * 1000)}",
            event_type=event_type,
            file_path=file_path,
            timestamp=datetime.now(),
            project_name=project_name,
            metadata=kwargs,
        )

        await self._event_queue.put(event)

    async def _process_detected_changes(self, project_name: str, changes: ChangeDetectionResult) -> None:
        """Process detected changes from change detection."""
        try:
            if not self._invalidation_service:
                self.logger.warning("No invalidation service available for processing changes")
                return

            # Use the invalidation service's change detection integration
            invalidation_events = await self._invalidation_service.detect_and_invalidate_changes(
                project_name, changes.get_files_to_reindex(), self._project_configs[project_name].root_directory
            )

            if invalidation_events[1]:  # invalidation_events is tuple of (changes, events)
                self._stats.invalidations_triggered += len(invalidation_events[1])
                self.logger.info(f"Triggered {len(invalidation_events[1])} invalidations for project {project_name}")

        except Exception as e:
            self.logger.error(f"Failed to process detected changes for project {project_name}: {e}")

    def _detect_project_for_file(self, file_path: str) -> str | None:
        """Auto-detect project for a file based on configured projects."""
        abs_path = str(Path(file_path).resolve())

        for project_name, config in self._project_configs.items():
            root_path = str(Path(config.root_directory).resolve())
            if abs_path.startswith(root_path):
                return project_name

        return None

    async def _start_project_event_monitoring(self, config: ProjectMonitoringConfig) -> None:
        """
        Start file system event monitoring for a project.

        Args:
            config: Project monitoring configuration
        """
        if not self._event_service:
            self.logger.warning(f"Cannot start event monitoring for {config.project_name} - event service not available")
            return

        try:
            success = await self._event_service.start_monitoring(
                project_name=config.project_name,
                root_directory=config.root_directory,
                file_patterns=config.file_patterns,
                exclude_patterns=config.exclude_patterns,
                recursive=True,
                debounce_interval=0.1,  # 100ms debounce
            )

            if success:
                self.logger.info(f"Started file system event monitoring for project: {config.project_name}")
            else:
                self.logger.warning(f"Failed to start event monitoring for project: {config.project_name}")

        except Exception as e:
            self.logger.error(f"Error starting event monitoring for project {config.project_name}: {e}")

    def _handle_file_system_event(self, event_info: FileSystemEventInfo) -> None:
        """
        Handle file system events from the event service.

        Args:
            event_info: File system event information
        """
        try:
            # Convert file system event to monitoring event
            monitoring_event_type = self._map_fs_event_to_monitoring_event(event_info.event_type)

            # Create monitoring event
            monitoring_event = MonitoringEvent(
                event_id=f"fs_{event_info.event_type.value}_{int(time.time() * 1000)}",
                event_type=monitoring_event_type,
                file_path=event_info.src_path,
                timestamp=event_info.timestamp,
                project_name=event_info.project_name,
                old_path=event_info.dest_path,
                metadata={
                    "source": "file_system_events",
                    "is_directory": event_info.is_directory,
                    **event_info.metadata,
                },
            )

            # Queue for processing
            asyncio.create_task(self._queue_event_for_processing(monitoring_event))

        except Exception as e:
            self.logger.error(f"Error handling file system event: {e}")

    def _map_fs_event_to_monitoring_event(self, fs_event_type: FileSystemEventType) -> MonitoringEventType:
        """
        Map file system event type to monitoring event type.

        Args:
            fs_event_type: File system event type

        Returns:
            Corresponding monitoring event type
        """
        mapping = {
            FileSystemEventType.CREATED: MonitoringEventType.FILE_CREATED,
            FileSystemEventType.MODIFIED: MonitoringEventType.FILE_MODIFIED,
            FileSystemEventType.DELETED: MonitoringEventType.FILE_DELETED,
            FileSystemEventType.MOVED: MonitoringEventType.FILE_MOVED,
            FileSystemEventType.DIRECTORY_CREATED: MonitoringEventType.DIRECTORY_CREATED,
            FileSystemEventType.DIRECTORY_DELETED: MonitoringEventType.DIRECTORY_DELETED,
            FileSystemEventType.DIRECTORY_MODIFIED: MonitoringEventType.DIRECTORY_CREATED,  # Treat as created for simplicity
        }
        return mapping.get(fs_event_type, MonitoringEventType.FILE_MODIFIED)

    async def _queue_event_for_processing(self, event: MonitoringEvent) -> None:
        """
        Queue a monitoring event for processing.

        Args:
            event: Monitoring event to queue
        """
        await self._event_queue.put(event)

    async def _process_pending_events(self) -> None:
        """Process any pending events in the queue."""
        while not self._event_queue.empty():
            try:
                event = self._event_queue.get_nowait()
                await self._process_monitoring_event(event)
                self._event_queue.task_done()
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                self.logger.error(f"Error processing pending event: {e}")


# Global file monitoring service instance
_file_monitoring_service: FileMonitoringService | None = None


async def get_file_monitoring_service() -> FileMonitoringService:
    """
    Get the global file monitoring service instance.

    Returns:
        FileMonitoringService: The global file monitoring service instance
    """
    global _file_monitoring_service
    if _file_monitoring_service is None:
        _file_monitoring_service = FileMonitoringService()
        await _file_monitoring_service.initialize()
    return _file_monitoring_service


async def shutdown_file_monitoring_service() -> None:
    """Shutdown the global file monitoring service."""
    global _file_monitoring_service
    if _file_monitoring_service:
        await _file_monitoring_service.shutdown()
        _file_monitoring_service = None
