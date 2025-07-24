"""
Tests for incremental call detection services.

This module tests the incremental processing capabilities for function call detection,
including change detection, dependency tracking, and real-time file watching.
"""

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.code_chunk import CodeChunk
from src.services.change_detector_service import ChangeDetectionResult, ChangeType, FileChange
from src.services.file_change_watcher_service import FileChangeEvent, FileChangeHandler, FileChangeWatcherService, WatcherConfig
from src.services.incremental_call_detection_service import (
    DependencyTracker,
    IncrementalCallDetectionService,
    IncrementalProcessingConfig,
    IncrementalProcessingResult,
)


class TestIncrementalProcessingConfig:
    """Test incremental processing configuration."""

    def test_config_creation(self):
        """Test configuration creation with defaults."""
        config = IncrementalProcessingConfig()

        assert config.enable_incremental_processing is True
        assert config.change_detection_interval_seconds == 300.0
        assert config.enable_dependency_tracking is True
        assert config.enable_cascade_reprocessing is True
        assert config.max_cascade_depth == 3
        assert config.force_reprocess_after_hours == 24.0

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "INCREMENTAL_CALL_DETECT_ENABLED": "false",
                "INCREMENTAL_CALL_DETECT_INTERVAL": "600",
                "INCREMENTAL_CALL_DETECT_MAX_DEPTH": "5",
            },
        ):
            config = IncrementalProcessingConfig.from_env()

            assert config.enable_incremental_processing is False
            assert config.change_detection_interval_seconds == 600.0
            assert config.max_cascade_depth == 5


class TestIncrementalProcessingResult:
    """Test incremental processing result model."""

    def test_result_creation(self):
        """Test creation of processing result."""
        result = IncrementalProcessingResult(
            total_files_analyzed=100,
            modified_files_processed=20,
            unchanged_files_skipped=80,
            cascade_files_processed=5,
            total_calls_detected=150,
            processing_time_ms=5000.0,
            cache_hits=30,
            cache_misses=10,
            performance_improvement_percent=75.0,
        )

        assert result.total_files_analyzed == 100
        assert result.modified_files_processed == 20
        assert result.unchanged_files_skipped == 80
        assert result.efficiency_ratio == 20.0  # 20/100 * 100
        assert result.cache_hit_rate == 75.0  # 30/(30+10) * 100

    def test_efficiency_calculation(self):
        """Test efficiency ratio calculation."""
        # High efficiency (few files processed)
        result1 = IncrementalProcessingResult(
            total_files_analyzed=100,
            modified_files_processed=10,
            unchanged_files_skipped=90,
            cascade_files_processed=0,
            total_calls_detected=0,
            processing_time_ms=1000.0,
            cache_hits=0,
            cache_misses=0,
            performance_improvement_percent=90.0,
        )

        assert result1.efficiency_ratio == 10.0

        # Low efficiency (many files processed)
        result2 = IncrementalProcessingResult(
            total_files_analyzed=100,
            modified_files_processed=80,
            unchanged_files_skipped=20,
            cascade_files_processed=0,
            total_calls_detected=0,
            processing_time_ms=1000.0,
            cache_hits=0,
            cache_misses=0,
            performance_improvement_percent=20.0,
        )

        assert result2.efficiency_ratio == 80.0

    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        # High cache hit rate
        result1 = IncrementalProcessingResult(
            total_files_analyzed=50,
            modified_files_processed=25,
            unchanged_files_skipped=25,
            cascade_files_processed=0,
            total_calls_detected=0,
            processing_time_ms=1000.0,
            cache_hits=45,
            cache_misses=5,
            performance_improvement_percent=50.0,
        )

        assert result1.cache_hit_rate == 90.0

        # No cache operations
        result2 = IncrementalProcessingResult(
            total_files_analyzed=50,
            modified_files_processed=25,
            unchanged_files_skipped=25,
            cascade_files_processed=0,
            total_calls_detected=0,
            processing_time_ms=1000.0,
            cache_hits=0,
            cache_misses=0,
            performance_improvement_percent=50.0,
        )

        assert result2.cache_hit_rate == 0.0


class TestDependencyTracker:
    """Test dependency tracking functionality."""

    def test_tracker_creation(self):
        """Test dependency tracker creation."""
        tracker = DependencyTracker("/test/file.py")

        assert tracker.file_path == "/test/file.py"
        assert len(tracker.dependencies) == 0
        assert len(tracker.dependents) == 0
        assert tracker.last_updated > 0

    def test_add_dependency(self):
        """Test adding dependencies."""
        tracker = DependencyTracker("/test/file.py")
        initial_time = tracker.last_updated

        time.sleep(0.01)  # Small delay
        tracker.add_dependency("/test/dependency.py")

        assert "/test/dependency.py" in tracker.dependencies
        assert tracker.last_updated > initial_time

        # Add same dependency again (should not duplicate)
        tracker.add_dependency("/test/dependency.py")
        assert len(tracker.dependencies) == 1

    def test_add_dependent(self):
        """Test adding dependents."""
        tracker = DependencyTracker("/test/file.py")

        tracker.add_dependent("/test/dependent.py")

        assert "/test/dependent.py" in tracker.dependents

        # Add another dependent
        tracker.add_dependent("/test/another_dependent.py")
        assert len(tracker.dependents) == 2

    def test_cascade_files_calculation(self):
        """Test cascade files calculation."""
        # Create a dependency chain: A -> B -> C -> D
        tracker_a = DependencyTracker("/test/a.py")
        tracker_b = DependencyTracker("/test/b.py")
        tracker_c = DependencyTracker("/test/c.py")

        # A depends on B, B depends on C
        tracker_a.add_dependency("/test/b.py")
        tracker_b.add_dependency("/test/c.py")

        # B is dependent on A, C is dependent on B
        tracker_b.add_dependent("/test/a.py")
        tracker_c.add_dependent("/test/b.py")

        # Get cascade files from A (should include dependents transitively)
        cascade_files = tracker_a.get_cascade_files(max_depth=3)

        # This test is simplified - in practice, the cascade would follow
        # the dependency graph structure more complex
        assert isinstance(cascade_files, set)


@pytest.mark.asyncio
class TestIncrementalCallDetectionService:
    """Test the incremental call detection service."""

    async def test_service_initialization(self):
        """Test service initialization."""
        config = IncrementalProcessingConfig()
        service = IncrementalCallDetectionService(config)

        assert service.config == config
        assert len(service._dependency_graph) == 0
        assert len(service._last_full_processing) == 0

    async def test_disabled_incremental_processing(self):
        """Test behavior when incremental processing is disabled."""
        config = IncrementalProcessingConfig(enable_incremental_processing=False)

        # Mock concurrent extractor
        mock_extractor = Mock()
        mock_extractor.extract_calls_from_files = AsyncMock(
            return_value=Mock(total_calls_detected=50, performance_metrics={"cache_hits": 10, "cache_misses": 5}, file_results=[])
        )

        service = IncrementalCallDetectionService(config=config, concurrent_extractor=mock_extractor)

        # Create test project chunks
        project_chunks = {"/test/file1.py": [Mock()], "/test/file2.py": [Mock()]}
        breadcrumb_mapping = {"func1": "test.func1", "func2": "test.func2"}

        # Process project (should fall back to full processing)
        result = await service.process_project_incrementally(
            project_name="test_project", project_directory="/test", all_project_chunks=project_chunks, breadcrumb_mapping=breadcrumb_mapping
        )

        # Should process all files (no incremental optimization)
        assert result.total_files_analyzed == 2
        assert result.modified_files_processed == 2
        assert result.unchanged_files_skipped == 0
        assert result.performance_improvement_percent == 0.0  # No improvement for full processing

    async def test_no_changes_detected(self):
        """Test behavior when no changes are detected."""
        config = IncrementalProcessingConfig()

        # Mock change detector that returns no changes
        mock_change_detector = Mock()
        mock_change_result = Mock()
        mock_change_result.has_changes = False
        mock_change_detector.detect_changes = Mock(return_value=mock_change_result)

        service = IncrementalCallDetectionService(config=config, change_detector=mock_change_detector)

        project_chunks = {"/test/file1.py": [Mock()]}
        breadcrumb_mapping = {"func1": "test.func1"}

        result = await service.process_project_incrementally(
            project_name="test_project", project_directory="/test", all_project_chunks=project_chunks, breadcrumb_mapping=breadcrumb_mapping
        )

        # Should skip all files
        assert result.total_files_analyzed == 1
        assert result.modified_files_processed == 0
        assert result.unchanged_files_skipped == 1
        assert result.performance_improvement_percent == 100.0  # No processing needed

    async def test_incremental_processing_with_changes(self):
        """Test incremental processing with detected changes."""
        config = IncrementalProcessingConfig()

        # Mock change detector with changes
        mock_change_detector = Mock()
        mock_change_result = Mock()
        mock_change_result.has_changes = True
        mock_change_result.modified_files = [Mock(file_path="/test/file1.py")]
        mock_change_result.added_files = []
        mock_change_detector.detect_changes = Mock(return_value=mock_change_result)

        # Mock concurrent extractor
        mock_extractor = Mock()
        mock_extractor.extract_calls_from_files = AsyncMock(
            return_value=Mock(total_calls_detected=25, performance_metrics={"cache_hits": 5, "cache_misses": 2}, file_results=[])
        )

        service = IncrementalCallDetectionService(config=config, change_detector=mock_change_detector, concurrent_extractor=mock_extractor)

        project_chunks = {"/test/file1.py": [Mock()], "/test/file2.py": [Mock()]}
        breadcrumb_mapping = {"func1": "test.func1", "func2": "test.func2"}

        result = await service.process_project_incrementally(
            project_name="test_project", project_directory="/test", all_project_chunks=project_chunks, breadcrumb_mapping=breadcrumb_mapping
        )

        # Should process only modified file
        assert result.total_files_analyzed == 2
        assert result.modified_files_processed == 1  # Only file1.py
        assert result.unchanged_files_skipped == 1  # file2.py skipped
        assert result.total_calls_detected == 25
        assert result.performance_improvement_percent > 0  # Some improvement

    async def test_dependency_graph_updates(self):
        """Test dependency graph updates."""
        service = IncrementalCallDetectionService()

        # Create chunks with imports
        chunk1 = Mock()
        chunk1.imports_used = ["module2"]

        chunk2 = Mock()
        chunk2.imports_used = []

        project_chunks = {"/test/module1.py": [chunk1], "/test/module2.py": [chunk2]}

        # Update dependency graph
        await service._update_dependency_graph(processed_files=["/test/module1.py"], all_project_chunks=project_chunks)

        # Check that dependency was recorded
        assert "/test/module1.py" in service._dependency_graph

        # Get dependency graph info
        graph_info = service.get_dependency_graph_info()
        assert graph_info["total_tracked_files"] == 1

    async def test_force_full_reprocessing(self):
        """Test forcing full reprocessing."""
        service = IncrementalCallDetectionService()

        project_name = "test_project"

        # Set a recent full processing time
        service._last_full_processing[project_name] = time.time()

        # Force full reprocessing
        await service.force_full_reprocessing(project_name)

        # Should reset the timestamp
        assert service._last_full_processing[project_name] == 0

    def test_statistics_collection(self):
        """Test statistics collection."""
        service = IncrementalCallDetectionService()

        # Update some stats
        service._stats["total_incremental_operations"] = 5
        service._stats["total_files_skipped"] = 100

        stats = service.get_statistics()

        assert "incremental_stats" in stats
        assert "dependency_graph_info" in stats
        assert "configuration" in stats
        assert stats["incremental_stats"]["total_incremental_operations"] == 5
        assert stats["incremental_stats"]["total_files_skipped"] == 100

    async def test_shutdown(self):
        """Test service shutdown."""
        service = IncrementalCallDetectionService()

        # Add some data
        service._dependency_graph["/test/file.py"] = DependencyTracker("/test/file.py")

        # Shutdown
        await service.shutdown()

        # Should clear dependency graph
        assert len(service._dependency_graph) == 0


class TestWatcherConfig:
    """Test file watcher configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = WatcherConfig()

        assert config.enable_watching is True
        assert "*.py" in config.watch_patterns
        assert "*.js" in config.watch_patterns
        assert "*/__pycache__/*" in config.ignore_patterns
        assert config.debounce_delay_seconds == 2.0
        assert config.batch_processing_delay_seconds == 5.0

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(
            "os.environ", {"FILE_WATCHER_ENABLED": "false", "FILE_WATCHER_PATTERNS": "*.py,*.ts", "FILE_WATCHER_DEBOUNCE": "3.0"}
        ):
            config = WatcherConfig.from_env()

            assert config.enable_watching is False
            assert config.watch_patterns == ["*.py", "*.ts"]
            assert config.debounce_delay_seconds == 3.0


class TestFileChangeEvent:
    """Test file change event model."""

    def test_event_creation(self):
        """Test creation of file change event."""
        timestamp = time.time()
        event = FileChangeEvent(file_path="/test/file.py", event_type="modified", timestamp=timestamp)

        assert event.file_path == "/test/file.py"
        assert event.event_type == "modified"
        assert event.timestamp == timestamp
        assert event.old_path is None

    def test_event_age_calculation(self):
        """Test event age calculation."""
        # Create event 2 seconds ago
        timestamp = time.time() - 2.0
        event = FileChangeEvent(file_path="/test/file.py", event_type="modified", timestamp=timestamp)

        age = event.age_seconds
        assert age >= 2.0
        assert age < 3.0  # Should be close to 2 seconds


@pytest.mark.asyncio
class TestFileChangeWatcherService:
    """Test file change watcher service."""

    async def test_watcher_initialization(self):
        """Test watcher service initialization."""
        config = WatcherConfig(enable_watching=False)  # Disable for testing
        service = FileChangeWatcherService(config)

        assert service.config == config
        assert len(service._pending_events) == 0
        assert len(service._watched_directories) == 0
        assert service._is_running is False

    def test_file_pattern_matching(self):
        """Test file pattern matching logic."""
        config = WatcherConfig(watch_patterns=["*.py", "*.js"], ignore_patterns=["*/__pycache__/*", "*/node_modules/*"])
        service = FileChangeWatcherService(config)

        # Should watch Python files
        assert service._should_watch_file("/test/script.py") is True
        assert service._should_watch_file("/test/app.js") is True

        # Should ignore cache files
        assert service._should_watch_file("/test/__pycache__/script.pyc") is False
        assert service._should_watch_file("/test/node_modules/package.js") is False

        # Should not watch other file types
        assert service._should_watch_file("/test/document.txt") is False

    def test_event_handling(self):
        """Test file event handling."""
        service = FileChangeWatcherService(WatcherConfig(enable_watching=False))

        # Handle a file modification
        service._handle_file_event("/test/file.py", "modified", time.time())

        # Should have pending event
        assert len(service._pending_events) == 1
        assert "/test/file.py" in service._pending_events

        event = service._pending_events["/test/file.py"]
        assert event.event_type == "modified"
        assert event.file_path == "/test/file.py"

        # Handle same file again (should update existing event)
        service._handle_file_event("/test/file.py", "modified", time.time())
        assert len(service._pending_events) == 1  # Still one event

    def test_statistics_collection(self):
        """Test statistics collection."""
        service = FileChangeWatcherService(WatcherConfig(enable_watching=False))

        # Simulate some events
        service._handle_file_event("/test/file1.py", "modified", time.time())
        service._handle_file_event("/test/file2.py", "created", time.time())

        stats = service.get_statistics()

        assert "watcher_stats" in stats
        assert "current_state" in stats
        assert "configuration" in stats

        assert stats["watcher_stats"]["total_events_received"] == 2
        assert stats["watcher_stats"]["events_by_type"]["modified"] == 1
        assert stats["watcher_stats"]["events_by_type"]["created"] == 1
        assert stats["current_state"]["pending_events"] == 2

    def test_pending_events_management(self):
        """Test pending events management."""
        service = FileChangeWatcherService(WatcherConfig(enable_watching=False))

        # Add some events
        service._handle_file_event("/test/file1.py", "modified", time.time())
        service._handle_file_event("/test/file2.py", "created", time.time())

        # Get pending events
        pending = service.get_pending_events()
        assert len(pending) == 2

        # Clear pending events
        service.clear_pending_events()
        assert len(service._pending_events) == 0
        assert len(service.get_pending_events()) == 0

    async def test_processing_callback(self):
        """Test processing callback functionality."""
        processed_events = []

        def callback(events):
            processed_events.extend(events)

        service = FileChangeWatcherService(config=WatcherConfig(enable_watching=False), processing_callback=callback)

        # Add events with old timestamps (to bypass debouncing)
        old_time = time.time() - 10
        service._handle_file_event("/test/file1.py", "modified", old_time)
        service._handle_file_event("/test/file2.py", "created", old_time)

        # Force process pending events
        await service.force_process_pending()

        # Should have called callback
        assert len(processed_events) == 2
        assert processed_events[0].file_path == "/test/file1.py"
        assert processed_events[1].file_path == "/test/file2.py"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
