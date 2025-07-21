"""
Tests for concurrent function call extraction services.

This module tests the concurrent processing capabilities for function call extraction,
including performance optimization, resource management, and batch processing.
"""

import asyncio
import tempfile
import time
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.code_chunk import CodeChunk
from src.models.function_call import CallDetectionResult, CallType, FunctionCall
from src.services.batch_call_processing_service import (
    BatchCallProcessingService,
    BatchProcessingSummary,
    BatchSchedulingConfig,
    FileBatchInfo,
    ProcessingBatch,
)
from src.services.concurrent_call_extractor_service import (
    BatchProcessingResult,
    ConcurrentCallExtractor,
    ConcurrentProcessingConfig,
    FileProcessingResult,
)


class TestConcurrentProcessingConfig:
    """Test concurrent processing configuration."""

    def test_config_creation(self):
        """Test configuration creation with defaults."""
        config = ConcurrentProcessingConfig()

        assert config.max_concurrent_files == 10
        assert config.max_concurrent_chunks_per_file == 5
        assert config.chunk_batch_size == 50
        assert config.timeout_seconds == 300.0
        assert config.enable_progress_tracking is True
        assert config.enable_memory_monitoring is True

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "CONCURRENT_CALL_EXTRACT_MAX_FILES": "20",
                "CONCURRENT_CALL_EXTRACT_MAX_CHUNKS": "10",
                "CONCURRENT_CALL_EXTRACT_BATCH_SIZE": "100",
            },
        ):
            config = ConcurrentProcessingConfig.from_env()

            assert config.max_concurrent_files == 20
            assert config.max_concurrent_chunks_per_file == 10
            assert config.chunk_batch_size == 100


class TestFileProcessingResult:
    """Test file processing result model."""

    def test_successful_result_creation(self):
        """Test creation of successful processing result."""
        result = FileProcessingResult(
            file_path="/test/file.py", success=True, total_calls_detected=5, processing_time_ms=150.0, chunks_processed=3, chunks_failed=0
        )

        assert result.file_path == "/test/file.py"
        assert result.success is True
        assert result.total_calls_detected == 5
        assert result.processing_time_ms == 150.0
        assert result.chunks_processed == 3
        assert result.chunks_failed == 0
        assert result.error_message is None

    def test_failed_result_creation(self):
        """Test creation of failed processing result."""
        result = FileProcessingResult(file_path="/test/file.py", success=False, error_message="Parse error", processing_time_ms=50.0)

        assert result.success is False
        assert result.error_message == "Parse error"
        assert result.total_calls_detected == 0


class TestBatchProcessingResult:
    """Test batch processing result model."""

    def test_batch_result_properties(self):
        """Test batch result calculated properties."""
        # Create some file results
        file_results = [
            FileProcessingResult("/file1.py", True, total_calls_detected=3),
            FileProcessingResult("/file2.py", True, total_calls_detected=7),
            FileProcessingResult("/file3.py", False, error_message="Error"),
        ]

        result = BatchProcessingResult(
            total_files=3,
            successful_files=2,
            failed_files=1,
            total_calls_detected=10,
            total_processing_time_ms=1000.0,
            file_results=file_results,
        )

        assert result.success_rate == pytest.approx(66.67, rel=1e-2)
        assert result.average_calls_per_file == 5.0


@pytest.mark.asyncio
class TestConcurrentCallExtractor:
    """Test the concurrent call extractor service."""

    async def test_extractor_initialization(self):
        """Test extractor initialization."""
        config = ConcurrentProcessingConfig(max_concurrent_files=5, cleanup_interval_seconds=0)
        extractor = ConcurrentCallExtractor(config)

        assert extractor.config.max_concurrent_files == 5
        assert len(extractor._extractor_pool) == 0  # Not initialized until pool init
        assert extractor._adaptive_concurrency == 5

    async def test_extractor_pool_initialization(self):
        """Test extractor pool initialization."""
        config = ConcurrentProcessingConfig(max_concurrent_files=3, cleanup_interval_seconds=0)
        extractor = ConcurrentCallExtractor(config)

        await extractor.initialize_pool()

        assert len(extractor._extractor_pool) == 3
        assert all(pool_extractor is not None for pool_extractor in extractor._extractor_pool)

    @patch("src.services.concurrent_call_extractor_service.FunctionCallExtractor")
    async def test_single_file_processing(self, mock_extractor_class):
        """Test processing a single file."""
        # Mock the extractor
        mock_extractor = Mock()
        mock_extractor.extract_calls_from_chunk = AsyncMock(
            return_value=CallDetectionResult(
                source_file_path="/test/file.py",
                source_breadcrumb="test.function",
                function_calls=[Mock()],  # One mock function call
                processing_time_ms=100.0,
                success=True,
            )
        )
        mock_extractor_class.return_value = mock_extractor

        config = ConcurrentProcessingConfig(max_concurrent_files=2, cleanup_interval_seconds=0)
        extractor = ConcurrentCallExtractor(config)

        # Create test data
        chunks = [
            CodeChunk(
                content="def test_function(): pass",
                file_path="/test/file.py",
                chunk_type="function",
                name="test_function",
                language="python",
            )
        ]

        file_chunks = {"/test/file.py": chunks}
        breadcrumb_mapping = {"test_function": "test.test_function"}

        # Process files
        result = await extractor.extract_calls_from_files(file_chunks, breadcrumb_mapping)

        assert result.total_files == 1
        assert result.successful_files == 1
        assert result.failed_files == 0
        assert result.total_calls_detected == 1

    @patch("src.services.concurrent_call_extractor_service.FunctionCallExtractor")
    async def test_multiple_files_processing(self, mock_extractor_class):
        """Test processing multiple files concurrently."""
        # Mock the extractor
        mock_extractor = Mock()
        mock_extractor.extract_calls_from_chunk = AsyncMock(
            return_value=CallDetectionResult(
                source_file_path="/test/file.py",
                source_breadcrumb="test.function",
                function_calls=[Mock(), Mock()],  # Two mock function calls
                processing_time_ms=100.0,
                success=True,
            )
        )
        mock_extractor_class.return_value = mock_extractor

        config = ConcurrentProcessingConfig(max_concurrent_files=5, cleanup_interval_seconds=0)
        extractor = ConcurrentCallExtractor(config)

        # Create test data for multiple files
        file_chunks = {}
        breadcrumb_mapping = {}

        for i in range(3):
            file_path = f"/test/file{i}.py"
            chunks = [
                CodeChunk(
                    content=f"def function_{i}(): pass", file_path=file_path, chunk_type="function", name=f"function_{i}", language="python"
                )
            ]
            file_chunks[file_path] = chunks
            breadcrumb_mapping[f"function_{i}"] = f"test.function_{i}"

        # Process files
        result = await extractor.extract_calls_from_files(file_chunks, breadcrumb_mapping)

        assert result.total_files == 3
        assert result.successful_files == 3
        assert result.failed_files == 0
        assert result.total_calls_detected == 6  # 2 calls per file * 3 files

    @patch("src.services.concurrent_call_extractor_service.FunctionCallExtractor")
    async def test_error_handling(self, mock_extractor_class):
        """Test error handling in concurrent processing."""
        # Mock the extractor to raise an exception
        mock_extractor = Mock()
        mock_extractor.extract_calls_from_chunk = AsyncMock(side_effect=Exception("Processing error"))
        mock_extractor_class.return_value = mock_extractor

        config = ConcurrentProcessingConfig(max_concurrent_files=2, cleanup_interval_seconds=0)
        extractor = ConcurrentCallExtractor(config)

        # Create test data
        chunks = [
            CodeChunk(
                content="def test_function(): pass",
                file_path="/test/file.py",
                chunk_type="function",
                name="test_function",
                language="python",
            )
        ]

        file_chunks = {"/test/file.py": chunks}
        breadcrumb_mapping = {"test_function": "test.test_function"}

        # Process files (should handle errors gracefully)
        result = await extractor.extract_calls_from_files(file_chunks, breadcrumb_mapping)

        assert result.total_files == 1
        assert result.successful_files == 0  # Should fail due to exception
        assert result.failed_files == 1
        assert result.total_calls_detected == 0

    @patch("src.services.concurrent_call_extractor_service.FunctionCallExtractor")
    async def test_timeout_handling(self, mock_extractor_class):
        """Test timeout handling in concurrent processing."""

        # Mock the extractor to hang
        async def slow_extract(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout
            return CallDetectionResult(
                source_file_path="/test/file.py",
                source_breadcrumb="test.function",
                function_calls=[],
                processing_time_ms=10000.0,
                success=True,
            )

        mock_extractor = Mock()
        mock_extractor.extract_calls_from_chunk = slow_extract
        mock_extractor_class.return_value = mock_extractor

        config = ConcurrentProcessingConfig(max_concurrent_files=2, timeout_seconds=0.1, cleanup_interval_seconds=0)  # Very short timeout
        extractor = ConcurrentCallExtractor(config)

        # Create test data
        chunks = [
            CodeChunk(
                content="def test_function(): pass",
                file_path="/test/file.py",
                chunk_type="function",
                name="test_function",
                language="python",
            )
        ]

        file_chunks = {"/test/file.py": chunks}
        breadcrumb_mapping = {"test_function": "test.test_function"}

        # Process files (should timeout)
        result = await extractor.extract_calls_from_files(file_chunks, breadcrumb_mapping)

        assert result.total_files == 1
        assert result.successful_files == 0
        assert result.failed_files == 1
        assert "timeout" in result.performance_metrics

    async def test_statistics_collection(self):
        """Test statistics collection."""
        config = ConcurrentProcessingConfig(cleanup_interval_seconds=0)
        extractor = ConcurrentCallExtractor(config)

        stats = extractor.get_statistics()

        assert "config" in stats
        assert "runtime_stats" in stats
        assert "current_state" in stats
        assert stats["config"]["max_concurrent_files"] == config.max_concurrent_files
        assert stats["runtime_stats"]["total_files_processed"] == 0

    async def test_shutdown(self):
        """Test extractor shutdown."""
        config = ConcurrentProcessingConfig(cleanup_interval_seconds=0)
        extractor = ConcurrentCallExtractor(config)

        await extractor.initialize_pool()
        assert len(extractor._extractor_pool) > 0

        await extractor.shutdown()
        assert len(extractor._extractor_pool) == 0


class TestBatchSchedulingConfig:
    """Test batch scheduling configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = BatchSchedulingConfig()

        assert config.max_files_per_batch == 100
        assert config.max_chunks_per_batch == 1000
        assert config.adaptive_batch_sizing is True
        assert config.enable_smart_grouping is True
        assert "python" in config.language_priorities

    def test_config_from_env(self):
        """Test configuration from environment."""
        with patch.dict(
            "os.environ",
            {"BATCH_CALL_PROCESS_MAX_FILES": "50", "BATCH_CALL_PROCESS_MAX_CHUNKS": "500", "BATCH_CALL_PROCESS_ADAPTIVE": "false"},
        ):
            config = BatchSchedulingConfig.from_env()

            assert config.max_files_per_batch == 50
            assert config.max_chunks_per_batch == 500
            assert config.adaptive_batch_sizing is False


class TestFileBatchInfo:
    """Test file batch information model."""

    def test_file_batch_info_creation(self):
        """Test creation of file batch info."""
        chunks = [Mock() for _ in range(15)]  # 15 chunks

        info = FileBatchInfo(
            file_path="/test/file.py", chunks=chunks, total_chunks=15, language="python", estimated_size=1000, priority_score=75.0
        )

        assert info.file_path == "/test/file.py"
        assert info.total_chunks == 15
        assert info.language == "python"
        assert info.size_category == "medium"  # 15 chunks = medium

    def test_size_categorization(self):
        """Test file size categorization."""
        # Test different size categories
        test_cases = [(5, "small"), (25, "medium"), (100, "large"), (300, "xlarge")]

        for chunk_count, expected_category in test_cases:
            info = FileBatchInfo(
                file_path="/test/file.py",
                chunks=[Mock() for _ in range(chunk_count)],
                total_chunks=chunk_count,
                language="python",
                estimated_size=1000,
                priority_score=50.0,
            )
            assert info.size_category == expected_category


class TestProcessingBatch:
    """Test processing batch model."""

    def test_batch_creation(self):
        """Test batch creation and properties."""
        file_infos = [
            FileBatchInfo("/file1.py", [Mock()], 1, "python", 100, 50.0),
            FileBatchInfo("/file2.py", [Mock(), Mock()], 2, "python", 200, 60.0),
        ]

        batch = ProcessingBatch(batch_id="test_batch", files=file_infos, total_files=2, total_chunks=3, estimated_processing_time_ms=1000.0)

        assert batch.batch_id == "test_batch"
        assert batch.total_files == 2
        assert batch.total_chunks == 3
        assert batch.estimated_processing_time_ms == 1000.0

    def test_file_chunks_dict_conversion(self):
        """Test conversion to file chunks dictionary."""
        chunks1 = [Mock(), Mock()]
        chunks2 = [Mock()]

        file_infos = [
            FileBatchInfo("/file1.py", chunks1, 2, "python", 100, 50.0),
            FileBatchInfo("/file2.py", chunks2, 1, "python", 50, 40.0),
        ]

        batch = ProcessingBatch(batch_id="test_batch", files=file_infos, total_files=2, total_chunks=3, estimated_processing_time_ms=1000.0)

        file_chunks_dict = batch.to_file_chunks_dict()

        assert "/file1.py" in file_chunks_dict
        assert "/file2.py" in file_chunks_dict
        assert file_chunks_dict["/file1.py"] == chunks1
        assert file_chunks_dict["/file2.py"] == chunks2

    def test_breadcrumb_mapping_creation(self):
        """Test breadcrumb mapping creation."""
        # Mock chunks with breadcrumbs
        chunk1 = Mock()
        chunk1.name = "function1"
        chunk1.breadcrumb = "module.function1"

        chunk2 = Mock()
        chunk2.name = "function2"
        chunk2.breadcrumb = None  # No breadcrumb

        file_infos = [
            FileBatchInfo("/file1.py", [chunk1], 1, "python", 100, 50.0),
            FileBatchInfo("/file2.py", [chunk2], 1, "python", 50, 40.0),
        ]

        batch = ProcessingBatch(batch_id="test_batch", files=file_infos, total_files=2, total_chunks=2, estimated_processing_time_ms=1000.0)

        breadcrumb_mapping = batch.create_breadcrumb_mapping()

        assert "function1" in breadcrumb_mapping
        assert "function2" in breadcrumb_mapping
        assert breadcrumb_mapping["function1"] == "module.function1"
        assert breadcrumb_mapping["function2"] == "file2.function2"  # Generated fallback


@pytest.mark.asyncio
class TestBatchCallProcessingService:
    """Test the batch call processing service."""

    async def test_service_initialization(self):
        """Test service initialization."""
        scheduling_config = BatchSchedulingConfig(max_files_per_batch=50)
        processing_config = ConcurrentProcessingConfig(max_concurrent_files=5)

        service = BatchCallProcessingService(scheduling_config, processing_config)

        assert service.scheduling_config.max_files_per_batch == 50
        assert service.processing_config.max_concurrent_files == 5
        assert service.concurrent_extractor is not None

    @patch("src.services.batch_call_processing_service.ConcurrentCallExtractor")
    async def test_priority_score_calculation(self, mock_extractor_class):
        """Test priority score calculation for files."""
        service = BatchCallProcessingService()

        # Create test chunks
        chunks = [Mock(chunk_type="function"), Mock(chunk_type="class"), Mock(chunk_type="variable")]

        # Test priority calculation
        score = await service._calculate_priority_score("/test/file.py", chunks, "python")

        assert score > 0
        # Python should get high priority, and function/class density should boost score

    @patch("src.services.batch_call_processing_service.ConcurrentCallExtractor")
    async def test_processing_time_estimation(self, mock_extractor_class):
        """Test processing time estimation."""
        service = BatchCallProcessingService()

        # Create test chunks with different content sizes
        chunks = [Mock(content="def small_function(): pass"), Mock(content="def larger_function():\n    # Much longer content\n    pass")]

        # Estimate time for Python
        time_estimate = await service._estimate_processing_time(chunks, "python")

        assert time_estimate > 0
        # Should be reasonable estimate (not too high or too low)
        assert 50 <= time_estimate <= 1000  # Between 50ms and 1s for 2 small chunks

    @patch("src.services.batch_call_processing_service.ConcurrentCallExtractor")
    async def test_file_grouping_logic(self, mock_extractor_class):
        """Test smart file grouping logic."""
        config = BatchSchedulingConfig(enable_smart_grouping=True)
        service = BatchCallProcessingService(scheduling_config=config)

        # Test files with same language and size category
        file1 = FileBatchInfo("/file1.py", [Mock()], 5, "python", 100, 50.0)  # small
        file2 = FileBatchInfo("/file2.py", [Mock()], 8, "python", 150, 60.0)  # small

        should_group = service._should_group_together(file1, file2)
        assert should_group is True

        # Test files with different languages
        file3 = FileBatchInfo("/file3.js", [Mock()], 5, "javascript", 100, 50.0)
        should_group = service._should_group_together(file1, file3)
        assert should_group is False

    async def test_statistics_collection(self):
        """Test statistics collection."""
        service = BatchCallProcessingService()

        stats = service.get_statistics()

        assert "batch_stats" in stats
        assert "concurrent_extractor_stats" in stats
        assert "configuration" in stats
        assert stats["batch_stats"]["total_batches_processed"] == 0

    async def test_shutdown(self):
        """Test service shutdown."""
        service = BatchCallProcessingService()

        # Should not raise any exceptions
        await service.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
