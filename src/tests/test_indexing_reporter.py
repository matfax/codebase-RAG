"""
Tests for IndexingReporter.

Tests comprehensive error reporting, analytics, and recommendation generation.
"""

import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.indexing_reporter import IndexingError, IndexingReport, IndexingReporter


class TestIndexingError(unittest.TestCase):
    """Test IndexingError dataclass."""

    def test_indexing_error_creation(self):
        """Test IndexingError creation with all fields."""
        error = IndexingError(
            error_type="syntax",
            file_path="/test/file.py",
            line_number=10,
            error_message="Syntax error",
            severity="error",
            context="print('hello'",
            suggestion="Add closing parenthesis",
        )

        self.assertEqual(error.error_type, "syntax")
        self.assertEqual(error.file_path, "/test/file.py")
        self.assertEqual(error.line_number, 10)
        self.assertEqual(error.error_message, "Syntax error")
        self.assertEqual(error.severity, "error")
        self.assertEqual(error.context, "print('hello'")
        self.assertEqual(error.suggestion, "Add closing parenthesis")
        self.assertIsNotNone(error.timestamp)

    def test_indexing_error_minimal(self):
        """Test IndexingError creation with minimal fields."""
        error = IndexingError(error_type="processing", file_path="/test/file.py")

        self.assertEqual(error.error_type, "processing")
        self.assertEqual(error.file_path, "/test/file.py")
        self.assertIsNone(error.line_number)
        self.assertEqual(error.error_message, "")
        self.assertEqual(error.severity, "error")
        self.assertEqual(error.context, "")
        self.assertEqual(error.suggestion, "")
        self.assertIsNotNone(error.timestamp)

    def test_indexing_error_timestamp_auto_set(self):
        """Test that timestamp is automatically set."""
        error = IndexingError(error_type="test", file_path="/test")

        # Timestamp should be in ISO format
        datetime.fromisoformat(error.timestamp)  # Will raise if invalid


class TestIndexingReport(unittest.TestCase):
    """Test IndexingReport dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        self.report = IndexingReport(
            operation_type="full_indexing", directory="/test", project_name="test_project", start_time=datetime.now().isoformat()
        )

    def test_indexing_report_creation(self):
        """Test IndexingReport creation."""
        self.assertEqual(self.report.operation_type, "full_indexing")
        self.assertEqual(self.report.directory, "/test")
        self.assertEqual(self.report.project_name, "test_project")
        self.assertIsNotNone(self.report.start_time)
        self.assertEqual(self.report.total_files, 0)
        self.assertEqual(len(self.report.errors), 0)
        self.assertEqual(len(self.report.warnings), 0)
        self.assertEqual(len(self.report.syntax_errors), 0)

    def test_add_error_categorization(self):
        """Test error categorization when adding errors."""
        # Add warning
        warning = IndexingError(error_type="processing", file_path="/test", severity="warning")
        self.report.add_error(warning)

        # Add syntax error
        syntax_error = IndexingError(error_type="syntax", file_path="/test")
        self.report.add_error(syntax_error)

        # Add regular error
        regular_error = IndexingError(error_type="embedding", file_path="/test")
        self.report.add_error(regular_error)

        self.assertEqual(len(self.report.warnings), 1)
        self.assertEqual(len(self.report.syntax_errors), 1)
        self.assertEqual(len(self.report.errors), 1)

    def test_get_error_summary(self):
        """Test error summary generation."""
        # Add multiple errors of different types
        errors = [
            IndexingError(error_type="syntax", file_path="/test1"),
            IndexingError(error_type="syntax", file_path="/test2"),
            IndexingError(error_type="embedding", file_path="/test3"),
            IndexingError(error_type="processing", file_path="/test4", severity="warning"),
        ]

        for error in errors:
            self.report.add_error(error)

        summary = self.report.get_error_summary()

        self.assertEqual(summary["syntax"], 2)
        self.assertEqual(summary["embedding"], 1)
        self.assertEqual(summary["processing"], 1)

    def test_has_critical_errors_true(self):
        """Test critical error detection when critical errors exist."""
        critical_error = IndexingError(error_type="embedding", file_path="/test")
        self.report.add_error(critical_error)

        self.assertTrue(self.report.has_critical_errors())

    def test_has_critical_errors_false(self):
        """Test critical error detection when no critical errors exist."""
        non_critical_error = IndexingError(error_type="syntax", file_path="/test")
        self.report.add_error(non_critical_error)

        self.assertFalse(self.report.has_critical_errors())

    def test_get_success_rate(self):
        """Test success rate calculation."""
        # Test with no files
        self.assertEqual(self.report.get_success_rate(), 100.0)

        # Test with some successful files
        self.report.total_files = 10
        self.report.successful_files = 8
        self.assertEqual(self.report.get_success_rate(), 80.0)

        # Test with all successful files
        self.report.successful_files = 10
        self.assertEqual(self.report.get_success_rate(), 100.0)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        self.report.total_files = 5
        self.report.successful_files = 4
        self.report.total_chunks = 20
        self.report.total_points = 20
        self.report.collections_used = ["test_collection"]

        error = IndexingError(error_type="syntax", file_path="/test")
        self.report.add_error(error)

        result_dict = self.report.to_dict()

        self.assertEqual(result_dict["operation_type"], "full_indexing")
        self.assertEqual(result_dict["total_files"], 5)
        self.assertEqual(result_dict["successful_files"], 4)
        self.assertEqual(result_dict["success_rate"], 80.0)
        self.assertIn("error_summary", result_dict)
        self.assertIn("has_critical_errors", result_dict)
        self.assertEqual(len(result_dict["syntax_errors"]), 1)


class TestIndexingReporter(unittest.TestCase):
    """Test IndexingReporter functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.reporter = IndexingReporter()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        """Test reporter initialization."""
        self.assertIsNotNone(self.reporter.logger)
        self.assertIsNone(self.reporter.current_report)
        self.assertEqual(self.reporter.processed_files_count, 0)
        self.assertEqual(self.reporter.successful_files_count, 0)

    def test_start_report(self):
        """Test starting a new report."""
        report = self.reporter.start_report("full_indexing", "/test/directory", "test_project")

        self.assertIsNotNone(report)
        self.assertEqual(report.operation_type, "full_indexing")
        self.assertEqual(report.directory, "/test/directory")
        self.assertEqual(report.project_name, "test_project")
        self.assertIsNotNone(report.start_time)
        self.assertEqual(self.reporter.current_report, report)
        self.assertEqual(self.reporter.processed_files_count, 0)
        self.assertEqual(self.reporter.successful_files_count, 0)

    def test_add_error_with_report(self):
        """Test adding error when report exists."""
        self.reporter.start_report("test", "/test", "test_project")

        with patch.object(self.reporter.logger, "error") as mock_log:
            self.reporter.add_error(
                error_type="syntax",
                file_path="/test/file.py",
                error_message="Syntax error",
                line_number=10,
                severity="error",
                context="print('hello'",
                suggestion="Add closing parenthesis",
            )

            self.assertEqual(len(self.reporter.current_report.syntax_errors), 1)
            error = self.reporter.current_report.syntax_errors[0]
            self.assertEqual(error.error_type, "syntax")
            self.assertEqual(error.file_path, "/test/file.py")
            self.assertEqual(error.error_message, "Syntax error")
            mock_log.assert_called_once()

    def test_add_error_without_report(self):
        """Test adding error when no report exists."""
        with patch.object(self.reporter.logger, "warning") as mock_warning, patch.object(self.reporter.logger, "error") as mock_error:
            self.reporter.add_error(error_type="test", file_path="/test", error_message="Test error")

            # Should create default report
            self.assertIsNotNone(self.reporter.current_report)
            self.assertEqual(self.reporter.current_report.operation_type, "unknown")
            mock_warning.assert_called_once()
            mock_error.assert_called_once()

    def test_update_file_counts(self):
        """Test updating file processing counts."""
        self.reporter.update_file_counts(10, 8)

        self.assertEqual(self.reporter.processed_files_count, 10)
        self.assertEqual(self.reporter.successful_files_count, 8)

    def test_finalize_report_success(self):
        """Test finalizing a report successfully."""
        report = self.reporter.start_report("test", "/test", "test_project")
        self.reporter.update_file_counts(5, 4)

        with patch.object(self.reporter, "_generate_recommendations") as mock_gen:
            finalized = self.reporter.finalize_report(
                total_chunks=20, total_points=20, collections_used=["test_collection"], performance_metrics={"memory_mb": 256}
            )

            self.assertEqual(finalized, report)
            self.assertIsNotNone(report.end_time)
            self.assertEqual(report.total_files, 5)
            self.assertEqual(report.successful_files, 4)
            self.assertEqual(report.failed_files, 1)
            self.assertEqual(report.total_chunks, 20)
            self.assertEqual(report.total_points, 20)
            self.assertEqual(report.collections_used, ["test_collection"])
            mock_gen.assert_called_once()

    def test_finalize_report_no_active_report(self):
        """Test finalizing when no active report exists."""
        with self.assertRaises(ValueError):
            self.reporter.finalize_report()

    def test_save_report_success(self):
        """Test saving report to file."""
        self.reporter.start_report("test", "/test", "test_project")
        self.reporter.finalize_report()

        report_path = self.reporter.save_report(str(self.temp_path))

        self.assertTrue(Path(report_path).exists())

        # Verify report content
        with open(report_path) as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data["operation_type"], "test")
        self.assertEqual(saved_data["project_name"], "test_project")

    def test_save_report_no_report(self):
        """Test saving when no report exists."""
        with self.assertRaises(ValueError):
            self.reporter.save_report()

    def test_save_report_default_directory(self):
        """Test saving report to default directory."""
        self.reporter.start_report("test", "/test", "test_project")
        self.reporter.finalize_report()

        with patch("services.indexing_reporter.Path.cwd") as mock_cwd:
            mock_cwd.return_value = self.temp_path

            report_path = self.reporter.save_report()

            self.assertTrue(Path(report_path).exists())

    @patch("builtins.print")
    def test_print_summary_no_errors(self, mock_print):
        """Test printing summary with no errors."""
        report = self.reporter.start_report("test", "/test", "test_project")
        report.total_files = 5
        report.successful_files = 5
        report.total_chunks = 25
        report.total_points = 25
        report.collections_used = ["test_collection"]

        self.reporter.print_summary()

        # Verify print was called (exact calls depend on formatting)
        self.assertTrue(mock_print.called)

    @patch("builtins.print")
    def test_print_summary_with_errors(self, mock_print):
        """Test printing summary with errors."""
        report = self.reporter.start_report("test", "/test", "test_project")
        report.total_files = 5
        report.successful_files = 3

        # Add some errors
        error = IndexingError(error_type="syntax", file_path="/test")
        report.add_error(error)

        critical_error = IndexingError(error_type="embedding", file_path="/test")
        report.add_error(critical_error)

        # Add recommendations
        report.recommendations = ["Fix syntax errors", "Check Ollama service"]

        self.reporter.print_summary()

        self.assertTrue(mock_print.called)

    def test_print_summary_no_report(self):
        """Test printing summary when no report exists."""
        with patch.object(self.reporter.logger, "warning") as mock_warning:
            self.reporter.print_summary()

            mock_warning.assert_called_once()

    def test_extract_syntax_errors_from_indexing_service(self):
        """Test extracting syntax errors from indexing service."""
        # Create mock indexing service with parse results
        mock_service = MagicMock()
        mock_syntax_error = MagicMock()
        mock_syntax_error.start_line = 10
        mock_syntax_error.error_type = "SyntaxError"
        mock_syntax_error.context = "print('hello'"
        mock_syntax_error.severity = "error"

        mock_parse_result = MagicMock()
        mock_parse_result.file_path = "/test/file.py"
        mock_parse_result.syntax_errors = [mock_syntax_error]
        mock_parse_result.fallback_used = True

        mock_service._parse_results = [mock_parse_result]

        self.reporter.start_report("test", "/test", "test_project")
        self.reporter.extract_syntax_errors_from_indexing_service(mock_service)

        # Should have added syntax error and fallback warning
        self.assertEqual(len(self.reporter.current_report.syntax_errors), 1)
        # Check for fallback warning in appropriate category
        all_errors = (
            self.reporter.current_report.errors + self.reporter.current_report.warnings + self.reporter.current_report.syntax_errors
        )
        fallback_errors = [e for e in all_errors if e.error_type == "parsing"]
        self.assertEqual(len(fallback_errors), 1)

    def test_extract_syntax_errors_no_parse_results(self):
        """Test extracting syntax errors when service has no parse results."""
        mock_service = MagicMock()
        del mock_service._parse_results  # Simulate no attribute

        self.reporter.start_report("test", "/test", "test_project")

        # Should not raise exception
        self.reporter.extract_syntax_errors_from_indexing_service(mock_service)

    def test_generate_recommendations_syntax_errors(self):
        """Test recommendation generation for syntax errors."""
        report = self.reporter.start_report("test", "/test", "test_project")

        syntax_error = IndexingError(error_type="syntax", file_path="/test")
        report.add_error(syntax_error)

        self.reporter._generate_recommendations()

        self.assertTrue(len(report.recommendations) > 0)
        syntax_rec = [r for r in report.recommendations if "syntax" in r.lower()]
        self.assertTrue(len(syntax_rec) > 0)

    def test_generate_recommendations_embedding_errors(self):
        """Test recommendation generation for embedding errors."""
        report = self.reporter.start_report("test", "/test", "test_project")

        embedding_error = IndexingError(error_type="embedding", file_path="/test")
        report.add_error(embedding_error)

        self.reporter._generate_recommendations()

        ollama_rec = [r for r in report.recommendations if "ollama" in r.lower()]
        self.assertTrue(len(ollama_rec) > 0)

    def test_generate_recommendations_storage_errors(self):
        """Test recommendation generation for storage errors."""
        report = self.reporter.start_report("test", "/test", "test_project")

        storage_error = IndexingError(error_type="storage", file_path="/test")
        report.add_error(storage_error)

        self.reporter._generate_recommendations()

        qdrant_rec = [r for r in report.recommendations if "qdrant" in r.lower()]
        self.assertTrue(len(qdrant_rec) > 0)

    def test_generate_recommendations_low_success_rate(self):
        """Test recommendation generation for low success rate."""
        report = self.reporter.start_report("test", "/test", "test_project")
        report.total_files = 10
        report.successful_files = 5  # 50% success rate

        self.reporter._generate_recommendations()

        performance_rec = [r for r in report.recommendations if "success rate" in r.lower()]
        self.assertTrue(len(performance_rec) > 0)

    def test_generate_recommendations_no_report(self):
        """Test recommendation generation when no report exists."""
        # Should not raise exception
        self.reporter._generate_recommendations()


if __name__ == "__main__":
    unittest.main()
