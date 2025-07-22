"""
File system event handler for real-time cache invalidation.

This service provides native file system event monitoring capabilities
using platform-specific APIs for immediate response to file changes,
complementing the polling-based monitoring with real-time responsiveness.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    FileSystemEvent = None
    FileSystemEventHandler = None
    Observer = None


class EventHandlerStatus(Enum):
    """Status of file system event handler."""

    INITIALIZED = "initialized"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    DISABLED = "disabled"


class FileSystemEventType(Enum):
    """Types of file system events."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"
    DIRECTORY_CREATED = "directory_created"
    DIRECTORY_DELETED = "directory_deleted"
    DIRECTORY_MODIFIED = "directory_modified"


@dataclass
class FileSystemEventInfo:
    """Information about a file system event."""

    event_type: FileSystemEventType
    src_path: str
    dest_path: str | None = None  # For move events
    is_directory: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    project_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventHandlerStats:
    """Statistics for file system event handling."""

    total_events: int = 0
    events_by_type: dict[FileSystemEventType, int] = field(default_factory=dict)
    processed_events: int = 0
    filtered_events: int = 0
    error_events: int = 0
    last_event_time: datetime | None = None
    handler_start_time: datetime | None = None
    watched_paths: set[str] = field(default_factory=set)

    def update_event(self, event_type: FileSystemEventType, processed: bool = True) -> None:
        """Update statistics with new event."""
        self.total_events += 1
        self.last_event_time = datetime.now()

        if event_type not in self.events_by_type:
            self.events_by_type[event_type] = 0
        self.events_by_type[event_type] += 1

        if processed:
            self.processed_events += 1
        else:
            self.filtered_events += 1

    def update_error(self) -> None:
        """Update error count."""
        self.error_events += 1


class CacheInvalidationEventHandler(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """
    File system event handler that triggers cache invalidation.

    This handler processes file system events and triggers appropriate
    cache invalidation operations through the monitoring service.
    """

    def __init__(
        self,
        project_name: str,
        invalidation_callback: Callable[[FileSystemEventInfo], None],
        file_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        debounce_interval: float = 0.1,
    ):
        """
        Initialize the cache invalidation event handler.

        Args:
            project_name: Name of the project being monitored
            invalidation_callback: Callback function for handling invalidation events
            file_patterns: File patterns to monitor
            exclude_patterns: File patterns to exclude
            debounce_interval: Minimum interval between events for the same file
        """
        if WATCHDOG_AVAILABLE:
            super().__init__()

        self.project_name = project_name
        self.invalidation_callback = invalidation_callback
        self.file_patterns = file_patterns or []
        self.exclude_patterns = exclude_patterns or []
        self.debounce_interval = debounce_interval
        self.logger = logging.getLogger(f"{__name__}.{project_name}")

        # Event tracking and debouncing
        self._last_event_times: dict[str, float] = {}
        self._pending_events: dict[str, FileSystemEventInfo] = {}
        self._debounce_tasks: dict[str, asyncio.Task] = {}

        # Statistics
        self.stats = EventHandlerStats()

    def should_process_file(self, file_path: str) -> bool:
        """
        Check if a file should be processed based on patterns.

        Args:
            file_path: Path to the file

        Returns:
            True if file should be processed
        """
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

    def should_debounce_event(self, file_path: str) -> bool:
        """
        Check if an event should be debounced.

        Args:
            file_path: Path to the file

        Returns:
            True if event should be debounced
        """
        current_time = time.time()
        last_time = self._last_event_times.get(file_path, 0)

        if current_time - last_time < self.debounce_interval:
            return True

        self._last_event_times[file_path] = current_time
        return False

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file/directory creation events."""
        if not WATCHDOG_AVAILABLE:
            return

        event_type = FileSystemEventType.DIRECTORY_CREATED if event.is_directory else FileSystemEventType.CREATED
        self._process_event(event_type, event.src_path, event.is_directory)

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file/directory modification events."""
        if not WATCHDOG_AVAILABLE:
            return

        event_type = FileSystemEventType.DIRECTORY_MODIFIED if event.is_directory else FileSystemEventType.MODIFIED
        self._process_event(event_type, event.src_path, event.is_directory)

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file/directory deletion events."""
        if not WATCHDOG_AVAILABLE:
            return

        event_type = FileSystemEventType.DIRECTORY_DELETED if event.is_directory else FileSystemEventType.DELETED
        self._process_event(event_type, event.src_path, event.is_directory)

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file/directory move events."""
        if not WATCHDOG_AVAILABLE:
            return

        # Handle move as delete + create for simplicity
        self._process_event(FileSystemEventType.DELETED, event.src_path, event.is_directory)
        self._process_event(FileSystemEventType.CREATED, event.dest_path, event.is_directory)

    def _process_event(self, event_type: FileSystemEventType, src_path: str, is_directory: bool, dest_path: str | None = None) -> None:
        """
        Process a file system event.

        Args:
            event_type: Type of the event
            src_path: Source path of the event
            is_directory: Whether the event is for a directory
            dest_path: Destination path for move events
        """
        try:
            # Skip directory events for now (we focus on file changes)
            if is_directory:
                self.stats.update_event(event_type, processed=False)
                return

            # Check if file should be processed
            if not self.should_process_file(src_path):
                self.stats.update_event(event_type, processed=False)
                return

            # Check debouncing
            if self.should_debounce_event(src_path):
                self._handle_debounced_event(event_type, src_path, dest_path)
                return

            # Create event info
            event_info = FileSystemEventInfo(
                event_type=event_type,
                src_path=src_path,
                dest_path=dest_path,
                is_directory=is_directory,
                project_name=self.project_name,
                metadata={"handler": "watchdog", "debounced": False},
            )

            # Trigger invalidation callback
            self.invalidation_callback(event_info)
            self.stats.update_event(event_type, processed=True)

            self.logger.debug(f"Processed {event_type.value} event for: {src_path}")

        except Exception as e:
            self.logger.error(f"Error processing file system event: {e}")
            self.stats.update_error()

    def _handle_debounced_event(self, event_type: FileSystemEventType, src_path: str, dest_path: str | None = None) -> None:
        """
        Handle a debounced event by updating pending events.

        Args:
            event_type: Type of the event
            src_path: Source path of the event
            dest_path: Destination path for move events
        """
        # Update or create pending event
        self._pending_events[src_path] = FileSystemEventInfo(
            event_type=event_type,
            src_path=src_path,
            dest_path=dest_path,
            is_directory=False,
            project_name=self.project_name,
            metadata={"handler": "watchdog", "debounced": True},
        )

        # Cancel existing debounce task if any
        if src_path in self._debounce_tasks:
            self._debounce_tasks[src_path].cancel()

        # Create new debounce task
        self._debounce_tasks[src_path] = asyncio.create_task(self._debounce_delay(src_path))

    async def _debounce_delay(self, src_path: str) -> None:
        """
        Wait for debounce interval and then process pending event.

        Args:
            src_path: Path of the file to process after delay
        """
        try:
            await asyncio.sleep(self.debounce_interval)

            # Process pending event if it still exists
            if src_path in self._pending_events:
                event_info = self._pending_events.pop(src_path)
                self.invalidation_callback(event_info)
                self.stats.update_event(event_info.event_type, processed=True)

            # Clean up debounce task
            if src_path in self._debounce_tasks:
                del self._debounce_tasks[src_path]

        except asyncio.CancelledError:
            # Task was cancelled, clean up
            if src_path in self._pending_events:
                del self._pending_events[src_path]
            if src_path in self._debounce_tasks:
                del self._debounce_tasks[src_path]

    def get_stats(self) -> EventHandlerStats:
        """Get statistics for this event handler."""
        return self.stats


class FileSystemEventService:
    """
    Service for managing file system event monitoring.

    This service provides real-time file system event monitoring using
    platform-specific APIs to complement polling-based monitoring.
    """

    def __init__(self, invalidation_callback: Callable[[FileSystemEventInfo], None] | None = None):
        """
        Initialize the file system event service.

        Args:
            invalidation_callback: Callback for handling invalidation events
        """
        self.logger = logging.getLogger(__name__)
        self.invalidation_callback = invalidation_callback

        # Event monitoring
        self._observers: dict[str, Observer] = {}
        self._event_handlers: dict[str, CacheInvalidationEventHandler] = {}
        self._status = EventHandlerStatus.INITIALIZED

        # Statistics
        self._global_stats = EventHandlerStats()

        # Check availability
        if not WATCHDOG_AVAILABLE:
            self.logger.warning("Watchdog library not available - file system events disabled")
            self._status = EventHandlerStatus.DISABLED

    async def initialize(self) -> None:
        """Initialize the file system event service."""
        try:
            if not WATCHDOG_AVAILABLE:
                self.logger.info("File system event service disabled (watchdog not available)")
                return

            self._status = EventHandlerStatus.RUNNING
            self._global_stats.handler_start_time = datetime.now()
            self.logger.info("File system event service initialized successfully")

        except Exception as e:
            self._status = EventHandlerStatus.ERROR
            self.logger.error(f"Failed to initialize file system event service: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the file system event service."""
        try:
            # Stop all observers
            for project_name in list(self._observers.keys()):
                await self.stop_monitoring(project_name)

            self._status = EventHandlerStatus.STOPPED
            self.logger.info("File system event service shutdown successfully")

        except Exception as e:
            self.logger.error(f"Error during file system event service shutdown: {e}")

    async def start_monitoring(
        self,
        project_name: str,
        root_directory: str,
        file_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        recursive: bool = True,
        debounce_interval: float = 0.1,
    ) -> bool:
        """
        Start monitoring a directory for file system events.

        Args:
            project_name: Name of the project
            root_directory: Root directory to monitor
            file_patterns: File patterns to monitor
            exclude_patterns: File patterns to exclude
            recursive: Whether to monitor subdirectories
            debounce_interval: Minimum interval between events for the same file

        Returns:
            True if monitoring started successfully
        """
        if not WATCHDOG_AVAILABLE:
            self.logger.warning(f"Cannot start monitoring for {project_name} - watchdog not available")
            return False

        try:
            # Stop existing monitoring if any
            if project_name in self._observers:
                await self.stop_monitoring(project_name)

            # Validate directory
            if not Path(root_directory).exists():
                self.logger.error(f"Directory does not exist: {root_directory}")
                return False

            # Create event handler
            event_handler = CacheInvalidationEventHandler(
                project_name=project_name,
                invalidation_callback=self._handle_invalidation_event,
                file_patterns=file_patterns,
                exclude_patterns=exclude_patterns,
                debounce_interval=debounce_interval,
            )

            # Create and start observer
            observer = Observer()
            observer.schedule(event_handler, root_directory, recursive=recursive)
            observer.start()

            # Store references
            self._observers[project_name] = observer
            self._event_handlers[project_name] = event_handler
            self._global_stats.watched_paths.add(root_directory)

            self.logger.info(f"Started file system monitoring for project: {project_name} at {root_directory}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start monitoring for project {project_name}: {e}")
            return False

    async def stop_monitoring(self, project_name: str) -> bool:
        """
        Stop monitoring for a project.

        Args:
            project_name: Name of the project to stop monitoring

        Returns:
            True if monitoring stopped successfully
        """
        try:
            if project_name in self._observers:
                observer = self._observers[project_name]
                observer.stop()
                observer.join(timeout=5.0)  # Wait up to 5 seconds for clean shutdown

                del self._observers[project_name]

            if project_name in self._event_handlers:
                # Cancel any pending debounce tasks
                handler = self._event_handlers[project_name]
                for task in handler._debounce_tasks.values():
                    task.cancel()

                del self._event_handlers[project_name]

            self.logger.info(f"Stopped file system monitoring for project: {project_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to stop monitoring for project {project_name}: {e}")
            return False

    def is_monitoring(self, project_name: str) -> bool:
        """
        Check if a project is being monitored.

        Args:
            project_name: Name of the project

        Returns:
            True if project is being monitored
        """
        return project_name in self._observers and self._observers[project_name].is_alive()

    def get_monitored_projects(self) -> list[str]:
        """Get list of projects being monitored."""
        return list(self._observers.keys())

    def get_service_status(self) -> EventHandlerStatus:
        """Get the status of the event service."""
        return self._status

    def get_global_stats(self) -> EventHandlerStats:
        """Get global statistics for all event handlers."""
        # Aggregate stats from all handlers
        aggregated_stats = EventHandlerStats()
        aggregated_stats.handler_start_time = self._global_stats.handler_start_time
        aggregated_stats.watched_paths = self._global_stats.watched_paths.copy()

        for handler in self._event_handlers.values():
            handler_stats = handler.get_stats()
            aggregated_stats.total_events += handler_stats.total_events
            aggregated_stats.processed_events += handler_stats.processed_events
            aggregated_stats.filtered_events += handler_stats.filtered_events
            aggregated_stats.error_events += handler_stats.error_events

            # Merge events by type
            for event_type, count in handler_stats.events_by_type.items():
                if event_type not in aggregated_stats.events_by_type:
                    aggregated_stats.events_by_type[event_type] = 0
                aggregated_stats.events_by_type[event_type] += count

            # Use most recent event time
            if handler_stats.last_event_time:
                if not aggregated_stats.last_event_time or handler_stats.last_event_time > aggregated_stats.last_event_time:
                    aggregated_stats.last_event_time = handler_stats.last_event_time

        return aggregated_stats

    def get_project_stats(self, project_name: str) -> EventHandlerStats | None:
        """
        Get statistics for a specific project.

        Args:
            project_name: Name of the project

        Returns:
            Event handler statistics or None if project not monitored
        """
        if project_name in self._event_handlers:
            return self._event_handlers[project_name].get_stats()
        return None

    def set_invalidation_callback(self, callback: Callable[[FileSystemEventInfo], None]) -> None:
        """
        Set the invalidation callback function.

        Args:
            callback: Callback function to handle invalidation events
        """
        self.invalidation_callback = callback

    def _handle_invalidation_event(self, event_info: FileSystemEventInfo) -> None:
        """
        Handle invalidation event from file system event handler.

        Args:
            event_info: File system event information
        """
        try:
            if self.invalidation_callback:
                self.invalidation_callback(event_info)
            else:
                self.logger.warning("No invalidation callback set for file system events")

        except Exception as e:
            self.logger.error(f"Error handling invalidation event: {e}")

    def is_available(self) -> bool:
        """Check if file system event monitoring is available."""
        return WATCHDOG_AVAILABLE

    def get_availability_info(self) -> dict[str, Any]:
        """Get information about event monitoring availability."""
        return {
            "watchdog_available": WATCHDOG_AVAILABLE,
            "status": self._status.value,
            "supported_platforms": ["Linux", "Windows", "macOS"] if WATCHDOG_AVAILABLE else [],
            "active_monitors": len(self._observers),
            "total_watched_paths": len(self._global_stats.watched_paths),
        }


# Global file system event service instance
_file_system_event_service: FileSystemEventService | None = None


async def get_file_system_event_service() -> FileSystemEventService:
    """
    Get the global file system event service instance.

    Returns:
        FileSystemEventService: The global event service instance
    """
    global _file_system_event_service
    if _file_system_event_service is None:
        _file_system_event_service = FileSystemEventService()
        await _file_system_event_service.initialize()
    return _file_system_event_service


async def shutdown_file_system_event_service() -> None:
    """Shutdown the global file system event service."""
    global _file_system_event_service
    if _file_system_event_service:
        await _file_system_event_service.shutdown()
        _file_system_event_service = None
