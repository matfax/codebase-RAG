"""
File Change Watcher Service for Real-time Incremental Call Detection.

This service provides real-time file system monitoring for automatic incremental
call detection processing when files are modified. It integrates with the
incremental call detection service to provide immediate updates.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

try:
    from watchdog.events import FileCreatedEvent, FileDeletedEvent, FileModifiedEvent, FileMovedEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


@dataclass
class FileChangeEvent:
    """Represents a file change event."""

    file_path: str
    event_type: str  # 'created', 'modified', 'deleted', 'moved'
    timestamp: float
    old_path: str | None = None  # For move events

    @property
    def age_seconds(self) -> float:
        """Get age of the event in seconds."""
        return time.time() - self.timestamp


@dataclass
class WatcherConfig:
    """Configuration for file change watching."""

    enable_watching: bool = True
    watch_patterns: list[str] = field(default_factory=lambda: ["*.py", "*.js", "*.ts", "*.java", "*.cpp"])
    ignore_patterns: list[str] = field(default_factory=lambda: ["*/__pycache__/*", "*/.git/*", "*/node_modules/*"])
    debounce_delay_seconds: float = 2.0  # Wait for file changes to settle
    batch_processing_delay_seconds: float = 5.0  # Batch multiple changes
    max_events_per_batch: int = 100
    enable_recursive_watching: bool = True

    @classmethod
    def from_env(cls) -> "WatcherConfig":
        """Create configuration from environment variables."""
        import os

        patterns = os.getenv("FILE_WATCHER_PATTERNS", "*.py,*.js,*.ts,*.java,*.cpp").split(",")
        ignore = os.getenv("FILE_WATCHER_IGNORE", "*/__pycache__/*,*/.git/*,*/node_modules/*").split(",")

        return cls(
            enable_watching=os.getenv("FILE_WATCHER_ENABLED", "true").lower() == "true",
            watch_patterns=[p.strip() for p in patterns],
            ignore_patterns=[p.strip() for p in ignore],
            debounce_delay_seconds=float(os.getenv("FILE_WATCHER_DEBOUNCE", "2.0")),
            batch_processing_delay_seconds=float(os.getenv("FILE_WATCHER_BATCH_DELAY", "5.0")),
            max_events_per_batch=int(os.getenv("FILE_WATCHER_MAX_BATCH", "100")),
            enable_recursive_watching=os.getenv("FILE_WATCHER_RECURSIVE", "true").lower() == "true",
        )


class FileChangeHandler(FileSystemEventHandler):
    """Handles file system events for the watcher service."""

    def __init__(self, watcher_service: "FileChangeWatcherService"):
        """Initialize the event handler."""
        super().__init__()
        self.watcher_service = watcher_service
        self.logger = logging.getLogger(__name__)

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self.watcher_service._handle_file_event(event.src_path, "modified", time.time())

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self.watcher_service._handle_file_event(event.src_path, "created", time.time())

    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory:
            self.watcher_service._handle_file_event(event.src_path, "deleted", time.time())

    def on_moved(self, event):
        """Handle file move events."""
        if not event.is_directory:
            self.watcher_service._handle_file_event(event.dest_path, "moved", time.time(), event.src_path)


class FileChangeWatcherService:
    """
    Service for real-time file change monitoring and incremental processing.

    This service monitors file system changes and triggers incremental call
    detection processing automatically when relevant files are modified.
    """

    def __init__(self, config: WatcherConfig | None = None, processing_callback: Callable[[list[FileChangeEvent]], None] | None = None):
        """
        Initialize the file change watcher service.

        Args:
            config: Watcher configuration
            processing_callback: Callback function for processing file changes
        """
        self.config = config or WatcherConfig.from_env()
        self.processing_callback = processing_callback
        self.logger = logging.getLogger(__name__)

        # Check watchdog availability
        if not WATCHDOG_AVAILABLE:
            self.logger.warning("Watchdog not available, file watching disabled")
            self.config.enable_watching = False

        # Event tracking
        self._pending_events: dict[str, FileChangeEvent] = {}  # file_path -> latest event
        self._watched_directories: set[str] = set()
        self._observer: Observer | None = None
        self._event_handler: FileChangeHandler | None = None

        # Processing control
        self._processing_task: asyncio.Task | None = None
        self._is_running = False
        self._last_batch_processing = time.time()

        # Statistics
        self._stats = {
            "total_events_received": 0,
            "total_batches_processed": 0,
            "total_files_processed": 0,
            "events_by_type": {"created": 0, "modified": 0, "deleted": 0, "moved": 0},
            "processing_errors": 0,
        }

        self.logger.info(f"FileChangeWatcherService initialized with watching: {self.config.enable_watching}")

    async def start_watching(self, directories: list[str]):
        """
        Start watching specified directories for file changes.

        Args:
            directories: List of directory paths to watch
        """
        if not self.config.enable_watching or not WATCHDOG_AVAILABLE:
            self.logger.info("File watching disabled or unavailable")
            return

        try:
            self._observer = Observer()
            self._event_handler = FileChangeHandler(self)

            # Set up directory watching
            for directory in directories:
                directory_path = Path(directory)
                if directory_path.exists() and directory_path.is_dir():
                    self._observer.schedule(self._event_handler, str(directory_path), recursive=self.config.enable_recursive_watching)
                    self._watched_directories.add(str(directory_path))
                    self.logger.info(f"Watching directory: {directory_path}")
                else:
                    self.logger.warning(f"Directory does not exist or is not a directory: {directory}")

            # Start the observer
            self._observer.start()
            self._is_running = True

            # Start batch processing task
            self._processing_task = asyncio.create_task(self._batch_processing_loop())

            self.logger.info(f"Started watching {len(self._watched_directories)} directories")

        except Exception as e:
            self.logger.error(f"Error starting file watcher: {e}")
            await self.stop_watching()

    async def stop_watching(self):
        """Stop watching for file changes."""
        self._is_running = False

        # Stop processing task
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        # Stop observer
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5.0)  # Wait up to 5 seconds

        self._watched_directories.clear()
        self._pending_events.clear()

        self.logger.info("File watching stopped")

    def _handle_file_event(self, file_path: str, event_type: str, timestamp: float, old_path: str | None = None):
        """Handle a file system event."""
        try:
            # Check if file matches watch patterns
            if not self._should_watch_file(file_path):
                return

            # Create event
            event = FileChangeEvent(file_path=file_path, event_type=event_type, timestamp=timestamp, old_path=old_path)

            # Update pending events (debounce by keeping latest event per file)
            self._pending_events[file_path] = event

            # Update statistics
            self._stats["total_events_received"] += 1
            self._stats["events_by_type"][event_type] += 1

            self.logger.debug(f"File event: {event_type} - {file_path}")

        except Exception as e:
            self.logger.error(f"Error handling file event: {e}")
            self._stats["processing_errors"] += 1

    def _should_watch_file(self, file_path: str) -> bool:
        """Determine if a file should be watched based on patterns."""
        path = Path(file_path)

        # Check ignore patterns first
        for ignore_pattern in self.config.ignore_patterns:
            if path.match(ignore_pattern):
                return False

        # Check watch patterns
        for watch_pattern in self.config.watch_patterns:
            if path.match(watch_pattern):
                return True

        return False

    async def _batch_processing_loop(self):
        """Main loop for batch processing file change events."""
        self.logger.info("Started batch processing loop")

        while self._is_running:
            try:
                await asyncio.sleep(1.0)  # Check every second

                current_time = time.time()

                # Check if it's time to process a batch
                time_since_last_batch = current_time - self._last_batch_processing

                if time_since_last_batch >= self.config.batch_processing_delay_seconds and len(self._pending_events) > 0:
                    await self._process_pending_events()
                    self._last_batch_processing = current_time

                # Also process if we have too many pending events
                elif len(self._pending_events) >= self.config.max_events_per_batch:
                    await self._process_pending_events()
                    self._last_batch_processing = current_time

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in batch processing loop: {e}")
                self._stats["processing_errors"] += 1
                await asyncio.sleep(5.0)  # Wait before retrying

        self.logger.info("Batch processing loop stopped")

    async def _process_pending_events(self):
        """Process all pending file change events."""
        if not self._pending_events:
            return

        try:
            # Get events to process (debounced)
            current_time = time.time()
            events_to_process = []

            for file_path, event in list(self._pending_events.items()):
                # Only process events that are old enough (debounced)
                if current_time - event.timestamp >= self.config.debounce_delay_seconds:
                    events_to_process.append(event)
                    del self._pending_events[file_path]

            if not events_to_process:
                return

            self.logger.info(f"Processing batch of {len(events_to_process)} file change events")

            # Call processing callback if provided
            if self.processing_callback:
                try:
                    # If callback is async
                    if asyncio.iscoroutinefunction(self.processing_callback):
                        await self.processing_callback(events_to_process)
                    else:
                        self.processing_callback(events_to_process)
                except Exception as e:
                    self.logger.error(f"Error in processing callback: {e}")
                    self._stats["processing_errors"] += 1

            # Update statistics
            self._stats["total_batches_processed"] += 1
            self._stats["total_files_processed"] += len(events_to_process)

        except Exception as e:
            self.logger.error(f"Error processing pending events: {e}")
            self._stats["processing_errors"] += 1

    def set_processing_callback(self, callback: Callable[[list[FileChangeEvent]], None]):
        """Set the callback function for processing file changes."""
        self.processing_callback = callback
        self.logger.info("Processing callback updated")

    def get_statistics(self) -> dict[str, Any]:
        """Get watcher statistics."""
        return {
            "watcher_stats": self._stats.copy(),
            "current_state": {
                "is_running": self._is_running,
                "watched_directories": list(self._watched_directories),
                "pending_events": len(self._pending_events),
                "oldest_pending_event_age": min([event.age_seconds for event in self._pending_events.values()], default=0),
            },
            "configuration": self.config.__dict__,
        }

    def get_pending_events(self) -> list[FileChangeEvent]:
        """Get list of pending events."""
        return list(self._pending_events.values())

    def clear_pending_events(self):
        """Clear all pending events."""
        cleared_count = len(self._pending_events)
        self._pending_events.clear()
        self.logger.info(f"Cleared {cleared_count} pending events")

    async def force_process_pending(self):
        """Force processing of all pending events immediately."""
        if self._pending_events:
            self.logger.info(f"Force processing {len(self._pending_events)} pending events")
            await self._process_pending_events()

    def is_watching_directory(self, directory: str) -> bool:
        """Check if a directory is being watched."""
        return str(Path(directory)) in self._watched_directories

    async def add_watch_directory(self, directory: str):
        """Add a new directory to watch."""
        if not self.config.enable_watching or not WATCHDOG_AVAILABLE or not self._observer:
            return

        directory_path = Path(directory)
        if directory_path.exists() and directory_path.is_dir():
            if str(directory_path) not in self._watched_directories:
                self._observer.schedule(self._event_handler, str(directory_path), recursive=self.config.enable_recursive_watching)
                self._watched_directories.add(str(directory_path))
                self.logger.info(f"Added watch directory: {directory_path}")
        else:
            self.logger.warning(f"Cannot add watch directory (not found): {directory}")

    async def remove_watch_directory(self, directory: str):
        """Remove a directory from watching."""
        if not self._observer:
            return

        directory_path = str(Path(directory))
        if directory_path in self._watched_directories:
            # Note: Watchdog doesn't provide easy way to remove specific watch
            # Would need to restart observer with new set of directories
            self._watched_directories.discard(directory_path)
            self.logger.info(f"Removed watch directory: {directory_path}")
            self.logger.warning("Note: Restart watcher to fully remove directory watch")
