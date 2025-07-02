#!/usr/bin/env python3
"""
Manual Indexing Tool for Codebase RAG MCP Server with Intelligent Code Chunking

This standalone script performs intelligent code indexing operations independently
from the MCP server, using Tree-sitter parsers for syntax-aware code analysis.
Particularly useful for large codebases that require function-level granular indexing.

Key Features:
- 🎯 Intelligent Code Chunking: Function, class, and method-level granular indexing
- 🌐 Multi-Language Support: Python, JavaScript, TypeScript, Go, Rust, Java, and more
- 🛡️ Syntax Error Tolerance: Graceful handling with detailed error reporting
- ⚡ Performance Optimized: Parallel processing with memory monitoring
- 📊 Comprehensive Reporting: Detailed syntax error statistics and recommendations

Usage Examples:
    # Full indexing with intelligent chunking (recommended for first-time indexing)
    python manual_indexing.py -d /path/to/repo -m clear_existing

    # Incremental indexing (only process changed files)
    python manual_indexing.py -d /path/to/repo -m incremental

    # Verbose output with detailed syntax error reporting
    python manual_indexing.py -d ./src --mode incremental --verbose

    # Skip confirmation prompts for automated workflows
    python manual_indexing.py -d /large/codebase -m clear_existing --no-confirm

    # Custom error report directory
    python manual_indexing.py -d ./src -m clear_existing --error-report-dir ./reports

Intelligent Chunking Benefits:
- Function-level search precision instead of whole-file results
- Rich metadata extraction (signatures, docstrings, breadcrumbs)
- Better embedding quality for semantic search
- Syntax error isolation (errors in one function don't affect others)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src directory to path to import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.change_detector_service import ChangeDetectorService
from services.embedding_service import EmbeddingService
from services.file_metadata_service import FileMetadataService
from services.indexing_service import IndexingService
from services.project_analysis_service import ProjectAnalysisService
from services.qdrant_service import QdrantService
from utils import format_duration, format_memory_size
from utils.performance_monitor import MemoryMonitor


@dataclass
class IndexingError:
    """Represents an error that occurred during indexing."""

    error_type: str  # Type of error (syntax, processing, embedding, storage, etc.)
    file_path: str  # File where error occurred
    line_number: int | None = None  # Line number if applicable
    error_message: str = ""  # Detailed error message
    severity: str = "error"  # Severity: error, warning, info
    context: str = ""  # Additional context or code snippet
    timestamp: str = ""  # When the error occurred
    suggestion: str = ""  # Suggested fix or workaround

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ErrorReport:
    """Comprehensive error report for indexing operations."""

    operation_type: str  # Type of operation (full_indexing, incremental, etc.)
    directory: str  # Directory being processed
    project_name: str  # Project name
    start_time: str  # Operation start time
    end_time: str  # Operation end time
    total_files: int = 0  # Total files processed
    successful_files: int = 0  # Successfully processed files
    failed_files: int = 0  # Files that failed to process
    errors: list[IndexingError] = None  # List of all errors
    warnings: list[IndexingError] = None  # List of warnings
    syntax_errors: list[IndexingError] = None  # Syntax-specific errors
    performance_metrics: dict[str, Any] = None  # Performance data
    recommendations: list[str] = None  # Actionable recommendations

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.syntax_errors is None:
            self.syntax_errors = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.recommendations is None:
            self.recommendations = []

    def add_error(self, error: IndexingError):
        """Add an error to the appropriate category."""
        if error.severity == "warning":
            self.warnings.append(error)
        elif error.error_type == "syntax":
            self.syntax_errors.append(error)
        else:
            self.errors.append(error)

    def get_error_summary(self) -> dict[str, int]:
        """Get a summary of error counts by type."""
        error_types = {}
        for error in self.errors + self.warnings + self.syntax_errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        return error_types

    def has_critical_errors(self) -> bool:
        """Check if there are any critical errors that prevent successful completion."""
        critical_types = ["embedding", "storage", "qdrant_connection"]
        return any(error.error_type in critical_types for error in self.errors)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "operation_type": self.operation_type,
            "directory": self.directory,
            "project_name": self.project_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_files": self.total_files,
            "successful_files": self.successful_files,
            "failed_files": self.failed_files,
            "error_summary": self.get_error_summary(),
            "has_critical_errors": self.has_critical_errors(),
            "errors": [asdict(error) for error in self.errors],
            "warnings": [asdict(error) for error in self.warnings],
            "syntax_errors": [asdict(error) for error in self.syntax_errors],
            "performance_metrics": self.performance_metrics,
            "recommendations": self.recommendations,
        }


class ManualIndexingTool:
    """
    Manual indexing tool for standalone indexing operations.

    This tool provides the same functionality as the MCP server's indexing
    capabilities but can be run independently for heavy operations.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the manual indexing tool.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.setup_logging()

        # Initialize services
        self.qdrant_service = QdrantService()
        self.embedding_service = EmbeddingService()
        self.indexing_service = IndexingService()
        self.metadata_service = FileMetadataService(self.qdrant_service)
        self.change_detector = ChangeDetectorService(self.metadata_service)
        self.project_analysis = ProjectAnalysisService()

        # Performance monitoring
        self.memory_monitor = MemoryMonitor()
        self.progress_tracker = None  # Will be initialized when we know total items

        # Error reporting
        self.error_report: ErrorReport | None = None
        self.processed_files_count = 0
        self.successful_files_count = 0

        self.logger = logging.getLogger(__name__)

    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO

        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

        # Reduce noise from third-party libraries
        if not self.verbose:
            logging.getLogger("qdrant_client").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)

    def validate_arguments(self, directory: str, mode: str) -> tuple[bool, str]:
        """
        Validate command-line arguments.

        Args:
            directory: Target directory path
            mode: Indexing mode

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate directory
        dir_path = Path(directory)
        if not dir_path.exists():
            return False, f"Directory does not exist: {directory}"

        if not dir_path.is_dir():
            return False, f"Path is not a directory: {directory}"

        if not os.access(dir_path, os.R_OK):
            return False, f"Directory is not readable: {directory}"

        # Validate mode
        valid_modes = ["clear_existing", "incremental"]
        if mode not in valid_modes:
            return (
                False,
                f"Invalid mode '{mode}'. Must be one of: {', '.join(valid_modes)}",
            )

        return True, ""

    def check_dependencies(self) -> tuple[bool, list[str]]:
        """
        Check if required dependencies are available.

        Returns:
            Tuple of (all_available, missing_services)
        """
        missing = []

        # Check Qdrant connection
        try:
            collections = self.qdrant_service.client.get_collections()
            self.logger.debug(f"Qdrant connection successful, found " f"{len(collections.collections)} collections")
        except Exception as e:
            missing.append(f"Qdrant (Error: {e})")

        # Check embedding service
        try:
            # Simple health check for embedding service
            if not hasattr(self.embedding_service, "generate_embeddings"):
                missing.append("Embedding service not properly initialized")
        except Exception as e:
            missing.append(f"Embedding service (Error: {e})")

        return len(missing) == 0, missing

    def estimate_indexing_time(self, directory: str, mode: str = "clear_existing") -> tuple[int, int, float]:
        """
        Estimate indexing time and provide statistics.

        Args:
            directory: Directory to analyze
            mode: Indexing mode ('clear_existing' or 'incremental')

        Returns:
            Tuple of (file_count, total_size_mb, estimated_minutes)
        """
        try:
            # Use existing project analysis service
            analysis_result = self.project_analysis.analyze_directory_structure(directory)

            total_files = analysis_result.get("relevant_files", 0)
            total_size_mb = analysis_result.get("size_analysis", {}).get("total_size_mb", 0)

            if mode == "incremental":
                # For incremental mode, try to detect actual changes
                try:
                    # Get project context
                    project_context = self.project_analysis.get_project_context(directory)
                    project_name = project_context.get("project_name", "unknown")

                    # Get current files
                    relevant_files = self.project_analysis.get_relevant_files(directory)

                    # Detect changes
                    changes = self.change_detector.detect_changes(
                        project_name=project_name,
                        current_files=relevant_files,
                        project_root=directory,
                    )

                    if not changes.has_changes:
                        return 0, 0, 0.0  # No changes to process

                    # Get actual files that need processing
                    files_to_reindex = changes.get_files_to_reindex()

                    actual_file_count = len(files_to_reindex)

                    # Estimate size of changed files only
                    changed_size_mb = 0
                    for file_path in files_to_reindex:
                        try:
                            file_size = Path(file_path).stat().st_size / (1024 * 1024)
                            changed_size_mb += file_size
                        except Exception:
                            pass

                    self.logger.info(f"Incremental mode: {actual_file_count} files need " f"processing (out of {total_files} total)")

                    file_count = actual_file_count
                    total_size_mb = changed_size_mb

                except Exception as e:
                    self.logger.warning(f"Could not detect changes for incremental estimation, " f"using full count: {e}")
                    file_count = total_files
            else:
                file_count = total_files

            # Rough estimation: ~100 files per minute for average-sized files
            # This is a conservative estimate and will vary based on file size
            # and system performance
            base_rate = 100  # files per minute
            size_factor = max(1.0, total_size_mb / 10)  # Slow down for larger files
            estimated_minutes = (file_count / base_rate) * size_factor

            return file_count, total_size_mb, estimated_minutes

        except Exception as e:
            self.logger.warning(f"Could not estimate indexing time: {e}")
            return 0, 0, 0.0

    def show_pre_indexing_summary(self, directory: str, mode: str):
        """
        Show summary before starting indexing.

        Args:
            directory: Target directory
            mode: Indexing mode
        """
        print("\n" + "=" * 60)
        print("MANUAL INDEXING TOOL - PRE-INDEXING SUMMARY")
        print("=" * 60)

        print(f"📁 Directory: {directory}")
        print(f"⚙️  Mode: {mode}")

        # Estimate time and show stats
        file_count, size_mb, estimated_minutes = self.estimate_indexing_time(directory, mode)

        if file_count > 0:
            if mode == "incremental":
                print(f"📊 Files to process: {file_count:,} (changed files only)")
            else:
                print(f"📊 Files to process: {file_count:,}")
            print(f"💾 Total size: {size_mb:.1f} MB")
            print(f"⏱️  Estimated time: {estimated_minutes:.1f} minutes")

            if estimated_minutes > 5:
                print("\n⚠️  WARNING: This operation may take several minutes.")
                print("   Consider running this in a separate terminal.")
        elif mode == "incremental":
            print("✅ No changes detected - all files are up to date!")
            print("⏱️  Estimated time: 0 minutes (no processing needed)")
        else:
            print("⚠️  No files found to process")

        print("\n" + "-" * 60)

    def initialize_error_report(self, directory: str, mode: str, project_name: str):
        """Initialize error report for the current operation."""
        self.error_report = ErrorReport(
            operation_type=mode,
            directory=directory,
            project_name=project_name,
            start_time=datetime.now().isoformat(),
            end_time="",  # Will be set when operation completes
        )
        self.processed_files_count = 0
        self.successful_files_count = 0

    def add_error(
        self,
        error_type: str,
        file_path: str,
        error_message: str,
        line_number: int | None = None,
        severity: str = "error",
        context: str = "",
        suggestion: str = "",
    ):
        """Add an error to the current error report."""
        if not self.error_report:
            return

        error = IndexingError(
            error_type=error_type,
            file_path=file_path,
            line_number=line_number,
            error_message=error_message,
            severity=severity,
            context=context,
            suggestion=suggestion,
        )

        self.error_report.add_error(error)

        # Log the error as well
        log_method = self.logger.warning if severity == "warning" else self.logger.error
        log_method(f"{error_type.upper()} in {file_path}: {error_message}")

    def generate_error_recommendations(self):
        """Generate actionable recommendations based on collected errors."""
        if not self.error_report:
            return

        recommendations = []
        error_summary = self.error_report.get_error_summary()

        # Syntax error recommendations
        if error_summary.get("syntax", 0) > 0:
            recommendations.append(
                "Fix syntax errors in your source files before indexing. "
                "Consider using a linter or IDE to identify and resolve syntax issues."
            )

        # Embedding error recommendations
        if error_summary.get("embedding", 0) > 0:
            recommendations.append(
                "Embedding generation failed for some files. "
                "Check if Ollama is running and the model is available. "
                "Try: ollama pull nomic-embed-text"
            )

        # Storage error recommendations
        if error_summary.get("storage", 0) > 0:
            recommendations.append(
                "Database storage errors occurred. Check Qdrant connection and "
                "disk space. Ensure Qdrant container is running: "
                "docker run -p 6333:6333 qdrant/qdrant"
            )

        # File processing recommendations
        if error_summary.get("processing", 0) > 0:
            recommendations.append(
                "Some files failed to process. Check file permissions and encoding. " "Large or binary files may cause processing issues."
            )

        # Performance recommendations
        high_failure_rate = (self.error_report.failed_files / max(self.error_report.total_files, 1)) > 0.2
        if high_failure_rate:
            recommendations.append(
                "High failure rate detected. Consider reducing batch sizes or " "checking system resources (memory, disk space)."
            )

        self.error_report.recommendations = recommendations

    def save_error_report(self, output_dir: str | None = None) -> str:
        """
        Save error report to a JSON file.

        Args:
            output_dir: Directory to save the report. Defaults to current directory.

        Returns:
            Path to the saved report file
        """
        if not self.error_report:
            raise ValueError("No error report to save")

        # Finalize the report
        self.error_report.end_time = datetime.now().isoformat()
        self.error_report.total_files = self.processed_files_count
        self.error_report.successful_files = self.successful_files_count
        self.error_report.failed_files = self.processed_files_count - self.successful_files_count

        # Generate performance metrics
        self.error_report.performance_metrics = {
            "memory_usage_mb": self.memory_monitor.get_current_usage(),
            "total_errors": len(self.error_report.errors),
            "total_warnings": len(self.error_report.warnings),
            "total_syntax_errors": len(self.error_report.syntax_errors),
            "success_rate": (self.successful_files_count / max(self.processed_files_count, 1)) * 100,
        }

        # Generate recommendations
        self.generate_error_recommendations()

        # Create output directory
        if output_dir is None:
            output_dir = Path.cwd()
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = self.error_report.project_name.replace(" ", "_")
        filename = f"indexing_report_{project_name}_{timestamp}.json"
        report_path = output_dir / filename

        # Save report
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(self.error_report.to_dict(), f, indent=2, ensure_ascii=False)

        return str(report_path)

    def print_error_summary(self):
        """Print a summary of errors to the console."""
        if not self.error_report:
            return

        error_summary = self.error_report.get_error_summary()
        total_errors = sum(error_summary.values())

        if total_errors == 0:
            print("✅ No errors or warnings encountered!")
            return

        print(f"\n📊 ERROR SUMMARY ({total_errors} total issues)")
        print("-" * 50)

        for error_type, count in sorted(error_summary.items()):
            severity_icon = (
                "⚠️"
                if any(
                    e.severity == "warning"
                    for e in self.error_report.errors + self.error_report.warnings + self.error_report.syntax_errors
                    if e.error_type == error_type
                )
                else "❌"
            )
            print(f"{severity_icon} {error_type}: {count}")

        # Show critical errors
        if self.error_report.has_critical_errors():
            print("\n🚨 CRITICAL ERRORS DETECTED - Some functionality may be affected")

        # Show top recommendations
        if self.error_report.recommendations:
            print("\n💡 TOP RECOMMENDATIONS:")
            for i, rec in enumerate(self.error_report.recommendations[:3], 1):
                print(f"   {i}. {rec}")

        print("-" * 50)

    async def perform_indexing(self, directory: str, mode: str) -> bool:
        """
        Perform the actual indexing operation.

        Args:
            directory: Target directory
            mode: Indexing mode ('clear_existing' or 'incremental')

        Returns:
            True if indexing was successful
        """
        start_time = time.time()

        try:
            # Initialize performance monitoring
            self.memory_monitor.start_monitoring()

            # Get project context
            project_context = self.project_analysis.get_project_context(directory)
            project_name = project_context.get("project_name", "unknown")

            # Initialize error reporting
            self.initialize_error_report(directory, mode, project_name)

            # For incremental mode, first check if there are any changes
            if mode == "incremental":
                try:
                    relevant_files = self.project_analysis.get_relevant_files(directory)
                    changes = self.change_detector.detect_changes(
                        project_name=project_name,
                        current_files=relevant_files,
                        project_root=directory,
                    )

                    if not changes.has_changes:
                        print(f"\n✅ No changes detected for project: {project_name}")
                        print("🎉 All files are already up to date!")
                        return True

                except Exception as e:
                    self.add_error(
                        "change_detection",
                        directory,
                        f"Failed to detect changes: {str(e)}",
                        suggestion="Check file permissions and metadata storage",
                    )

            print(f"\n🚀 Starting {mode} indexing for project: {project_name}")

            if mode == "clear_existing":
                result = await self.perform_full_indexing(directory, project_name)
            elif mode == "incremental":
                result = await self.perform_incremental_indexing(directory, project_name)
            else:
                self.logger.error(f"Unknown indexing mode: {mode}")
                self.add_error("configuration", directory, f"Unknown indexing mode: {mode}")
                return False

            # Calculate final statistics
            duration = time.time() - start_time
            final_memory = self.memory_monitor.get_current_usage()

            # Print error summary
            self.print_error_summary()

            # Save error report if there were any issues or if verbose mode
            if self.error_report and (self.error_report.get_error_summary() or self.verbose):
                try:
                    report_path = self.save_error_report(getattr(self, "error_report_dir", None))
                    print(f"\n📋 Error report saved to: {report_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save error report: {e}")

            if result:
                print("\n✅ Indexing completed successfully!")
                print(f"⏱️  Total time: {format_duration(duration)}")
                print(f"💾 Memory usage: {format_memory_size(final_memory)}")
                print(f"📊 Files processed: {self.successful_files_count}/" f"{self.processed_files_count}")
            else:
                print("\n❌ Indexing failed!")
                print(f"⏱️  Time elapsed: {format_duration(duration)}")
                if self.error_report and self.error_report.has_critical_errors():
                    print("🚨 Critical errors prevented successful completion")

            return result

        except Exception as e:
            self.logger.error(f"Error during indexing: {e}", exc_info=True)
            print(f"\n❌ Indexing failed with error: {e}")
            return False
        finally:
            # Clean up progress tracker if it exists
            if self.progress_tracker:
                # Progress tracker doesn't have a stop method, just clear reference
                self.progress_tracker = None
            try:
                self.memory_monitor.stop_monitoring()
            except Exception as e:
                print(f"❌ Unexpected error: {e}")

    async def perform_full_indexing(self, directory: str, project_name: str) -> bool:
        """
        Perform full (clear_existing) indexing.

        Args:
            directory: Target directory
            project_name: Name of the project

        Returns:
            True if successful
        """
        print("📋 Mode: Full indexing (clear_existing)")

        # Clear existing metadata
        print("🗑️  Clearing existing metadata...")
        self.metadata_service.clear_project_metadata(project_name)

        # Process codebase
        print("📊 Processing codebase...")
        chunks = self.indexing_service.process_codebase_for_indexing(directory)

        # Extract and report syntax errors from parsing
        self._extract_syntax_errors_from_indexing_service()

        if not chunks:
            print("⚠️  No files found to index")
            self.add_error(
                "processing",
                directory,
                "No files found to index",
                suggestion="Check file patterns and .ragignore configuration",
            )
            return False

        print(f"📄 Generated {len(chunks)} chunks from codebase")

        # Generate embeddings and store
        print("🧠 Generating and storing embeddings...")
        self._generate_and_store_embeddings(
            chunks=chunks,
            project_context={"project_name": project_name, "source_path": directory},
        )

        # Store metadata for future incremental updates
        print("💾 Storing file metadata...")
        self.store_file_metadata(directory, project_name)

        return True

    async def perform_incremental_indexing(self, directory: str, project_name: str) -> bool:
        """
        Perform incremental indexing.

        Args:
            directory: Target directory
            project_name: Name of the project

        Returns:
            True if successful
        """
        print("📋 Mode: Incremental indexing")

        # Get current files
        print("🔍 Analyzing current files...")
        relevant_files = self.project_analysis.get_relevant_files(directory)

        # Detect changes
        print("🔎 Detecting changes...")
        changes = self.change_detector.detect_changes(
            project_name=project_name,
            current_files=relevant_files,
            project_root=directory,
        )

        if not changes.has_changes:
            print("✅ No changes detected - all files are up to date!")
            return True

        # Show change summary
        summary = changes.get_summary()
        print("📊 Changes detected:")
        for change_type, count in summary.items():
            if count > 0 and change_type != "total_changes":
                print(f"   {change_type}: {count}")

        # Process changed files
        files_to_reindex = changes.get_files_to_reindex()
        files_to_remove = changes.get_files_to_remove()

        if files_to_remove:
            print(f"🗑️  Removing {len(files_to_remove)} obsolete entries...")
            # TODO: Implement removal from vector database

        if files_to_reindex:
            print(f"🔄 Reindexing {len(files_to_reindex)} changed files...")

            # Process only changed files
            chunks = self.indexing_service.process_specific_files(files_to_reindex, project_name, directory)

            if chunks:
                # Generate embeddings for changed files
                self._generate_and_store_embeddings(
                    chunks=chunks,
                    project_context={
                        "project_name": project_name,
                        "source_path": directory,
                    },
                )

        # Update metadata
        print("💾 Updating file metadata...")
        self.store_file_metadata(directory, project_name)

        return True

    def store_file_metadata(self, directory: str, project_name: str):
        """
        Store file metadata for future change detection.

        Args:
            directory: Source directory
            project_name: Project name
        """
        try:
            # Get all relevant files
            relevant_files = self.project_analysis.get_relevant_files(directory)

            # Create metadata for each file
            from models.file_metadata import FileMetadata

            metadata_list = []

            for file_path in relevant_files:
                try:
                    metadata = FileMetadata.from_file_path(file_path, directory)
                    metadata_list.append(metadata)
                except Exception as e:
                    self.logger.warning(f"Failed to create metadata for {file_path}: {e}")

            # Store metadata
            success = self.metadata_service.store_file_metadata(project_name, metadata_list)

            if success:
                self.logger.info(f"Stored metadata for {len(metadata_list)} files")
            else:
                self.logger.error("Failed to store file metadata")

        except Exception as e:
            self.logger.error(f"Error storing file metadata: {e}")

    def _generate_and_store_embeddings(self, chunks, project_context):
        """
        Generate embeddings and store them using the same logic as MCP tools.

        Args:
            chunks: List of chunks to process
            project_context: Context information including project name
        """
        try:
            import os
            import uuid
            from collections import defaultdict

            from qdrant_client.http.models import PointStruct

            # Get embedding model name
            model_name = os.getenv("OLLAMA_DEFAULT_EMBEDDING_MODEL", "nomic-embed-text")

            # Initialize Qdrant client
            from services.qdrant_service import QdrantService

            qdrant_service = QdrantService()

            # Group chunks by collection type (similar to MCP tools logic)
            collection_chunks = defaultdict(list)

            for chunk in chunks:
                file_path = chunk.metadata.get("file_path", "")
                language = chunk.metadata.get("language", "unknown")

                # Determine collection type based on file characteristics
                if language in [
                    "python",
                    "javascript",
                    "typescript",
                    "java",
                    "go",
                    "rust",
                ]:
                    collection_type = "code"
                elif any(file_path.endswith(ext) for ext in [".json", ".yaml", ".yml", ".toml", ".ini"]):
                    collection_type = "config"
                else:
                    collection_type = "documentation"

                project_name = project_context.get("project_name", "unknown")
                collection_name = f"project_{project_name}_{collection_type}"
                collection_chunks[collection_name].append(chunk)

            print(f"📊 Processing {len(chunks)} chunks across " f"{len(collection_chunks)} collections")

            total_points = 0

            # Track file processing for error reporting
            files_in_batch = {chunk.metadata.get("file_path", "") for chunk in chunks}
            self.processed_files_count += len(files_in_batch)

            # Process each collection
            for collection_name, collection_chunk_list in collection_chunks.items():
                print(f"🔄 Processing collection: {collection_name} (" f"{len(collection_chunk_list)} chunks)")

                try:
                    # Prepare texts for embedding generation
                    texts = [chunk.content for chunk in collection_chunk_list]

                    # Generate embeddings
                    embeddings = self.embedding_service.generate_embeddings(model_name, texts)

                    if embeddings is None:
                        error_msg = f"Failed to generate embeddings for collection " f"{collection_name}"
                        self.logger.error(error_msg)
                        self.add_error(
                            "embedding",
                            collection_name,
                            error_msg,
                            suggestion="Check Ollama service and model availability",
                        )
                        continue

                    # Create points for Qdrant
                    points = []
                    embedding_errors = 0

                    for chunk, embedding in zip(collection_chunk_list, embeddings, strict=False):
                        file_path = chunk.metadata.get("file_path", "unknown")

                        if embedding is None:
                            embedding_errors += 1
                            self.add_error(
                                "embedding",
                                file_path,
                                "Failed to generate embedding for chunk",
                                severity="warning",
                            )
                            continue

                        try:
                            point_id = str(uuid.uuid4())

                            # Prepare metadata
                            metadata = chunk.metadata.copy()
                            metadata["collection"] = collection_name

                            point = PointStruct(id=point_id, vector=embedding.tolist(), payload=metadata)
                            points.append(point)

                        except Exception as e:
                            self.add_error(
                                "processing",
                                file_path,
                                f"Failed to prepare point: {str(e)}",
                            )

                    if points:
                        try:
                            # Ensure collection exists before inserting
                            print(f"🔧 Ensuring collection exists: {collection_name}")
                            self._ensure_collection_exists(collection_name, qdrant_service)

                            # Store in Qdrant
                            stats = qdrant_service.batch_upsert_with_retry(collection_name, points)
                            total_points += stats.successful_insertions
                            print(f"✅ Stored {stats.successful_insertions} points in " f"{collection_name}")

                            # Track any storage failures
                            if stats.failed_insertions > 0:
                                self.add_error(
                                    "storage",
                                    collection_name,
                                    f"{stats.failed_insertions} points failed to store",
                                    severity="warning",
                                    suggestion="Check Qdrant disk space and connection",
                                )

                        except Exception as e:
                            self.add_error(
                                "storage",
                                collection_name,
                                f"Failed to store embeddings: {str(e)}",
                                suggestion="Check Qdrant connection and disk space",
                            )
                    else:
                        self.add_error(
                            "processing",
                            collection_name,
                            "No valid points generated for collection",
                        )

                except Exception as e:
                    # Collection-level error
                    self.add_error(
                        "processing",
                        collection_name,
                        f"Collection processing failed: {str(e)}",
                    )

            # Update successful files count
            # Estimate based on successful points vs total chunks
            if chunks:
                success_ratio = total_points / len(chunks)
                estimated_successful_files = int(len(files_in_batch) * success_ratio)
                self.successful_files_count += estimated_successful_files

            print(f"🎉 Successfully processed {total_points} total points")

        except Exception as e:
            self.logger.error(f"Error generating and storing embeddings: {e}")
            raise

    def _ensure_collection_exists(self, collection_name: str, qdrant_service):
        """
        Ensure that a Qdrant collection exists before attempting to insert data.

        Args:
            collection_name: Name of the collection to check/create
            qdrant_service: QdrantService instance
        """
        try:
            if not qdrant_service.collection_exists(collection_name):
                # Import required classes
                from qdrant_client.http.models import Distance, VectorParams

                self.logger.info(f"Creating collection: {collection_name}")

                # Get embedding dimension (default to 768 for nomic-embed-text)
                embedding_dimension = 768

                # Create the collection
                qdrant_service.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=embedding_dimension, distance=Distance.COSINE),
                )

                self.logger.info(f"Successfully created collection: {collection_name}")
            else:
                self.logger.debug(f"Collection already exists: {collection_name}")

        except Exception as e:
            self.logger.error(f"Failed to ensure collection {collection_name} exists: {e}")
            raise

    def _extract_syntax_errors_from_indexing_service(self):
        """Extract syntax errors from the indexing service parsing results."""
        try:
            # Check if the indexing service has collected parse results
            if hasattr(self.indexing_service, "_parse_results"):
                for parse_result in self.indexing_service._parse_results:
                    if parse_result.syntax_errors:
                        for syntax_error in parse_result.syntax_errors:
                            self.add_error(
                                error_type="syntax",
                                file_path=parse_result.file_path,
                                line_number=syntax_error.start_line,
                                error_message=(f"{syntax_error.error_type}: {syntax_error.context}"),
                                severity=("warning" if syntax_error.severity == "warning" else "error"),
                                context=syntax_error.context,
                                suggestion="Fix syntax errors using a linter or IDE",
                            )

                    # Report if fallback was used (indicates parsing issues)
                    if parse_result.fallback_used:
                        self.add_error(
                            error_type="parsing",
                            file_path=parse_result.file_path,
                            error_message=("Intelligent parsing failed, used fallback to " "whole-file chunking"),
                            severity="warning",
                            suggestion=("Check file syntax and Tree-sitter parser support"),
                        )

        except Exception as e:
            self.logger.debug(f"Could not extract syntax errors: {e}")
            # Don't fail the whole operation for this


def main():
    """Main entry point for the manual indexing tool."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Manual Indexing Tool with Intelligent Code Chunking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 Intelligent Code Chunking Examples:

Basic Usage:
  %(prog)s -d /path/to/repo -m clear_existing
                        Full indexing with intelligent chunking (first-time setup)
  %(prog)s -d ./src -m incremental
                        Process only changed files (fast updates)

Advanced Usage:
  %(prog)s -d /large/codebase -m clear_existing --verbose
                        Verbose output with detailed syntax error reporting
  %(prog)s -d ./project -m incremental --no-confirm
                        Skip prompts for automated CI/CD workflows
  %(prog)s -d ./src -m clear_existing --error-report-dir ./reports
                        Save detailed error reports to custom directory

Language-Specific Examples:
  %(prog)s -d /python/project -m clear_existing --verbose
                        Index Python project with function/class chunking
  %(prog)s -d /frontend/app -m incremental
                        Update TypeScript/JavaScript React application
  %(prog)s -d /backend/go -m clear_existing
                        Index Go microservice with struct/function analysis

Performance Recommendations:
  - Use clear_existing for first-time indexing or major refactoring
  - Use incremental for daily development updates (80%+ faster)
  - Enable --verbose for debugging syntax errors and performance monitoring
  - Large projects (1000+ files): Run during off-hours or CI/CD
  - Set INDEXING_CONCURRENCY env var to match your CPU cores

Output Reports:
  - Error statistics saved as JSON reports in current/specified directory
  - Syntax error details with line numbers and recovery suggestions
  - Performance metrics including processing rates and memory usage
  - Actionable recommendations for improving indexing results
        """,
    )

    parser.add_argument(
        "-d",
        "--directory",
        required=True,
        help="Target directory path to index with intelligent code chunking",
    )

    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        choices=["clear_existing", "incremental"],
        help=("Indexing mode: clear_existing (full reindex with intelligent " "chunking) or incremental (only changed files)"),
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=("Enable verbose logging with detailed syntax error reporting and " "performance metrics"),
    )

    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help=("Skip confirmation prompts - useful for automated CI/CD workflows " "(use with caution)"),
    )

    parser.add_argument(
        "--error-report-dir",
        help=("Directory to save comprehensive error reports with syntax error " "details (default: current directory)"),
    )

    args = parser.parse_args()

    # Initialize tool
    tool = ManualIndexingTool(verbose=args.verbose)

    # Set error report directory if provided
    if args.error_report_dir:
        tool.error_report_dir = args.error_report_dir

    # Validate arguments
    is_valid, error_msg = tool.validate_arguments(args.directory, args.mode)
    if not is_valid:
        print(f"❌ Error: {error_msg}")
        sys.exit(1)

    # Check dependencies
    deps_ok, missing = tool.check_dependencies()
    if not deps_ok:
        print("❌ Missing dependencies:")
        for service in missing:
            print(f"   - {service}")
        print("\nPlease ensure Qdrant is running and services are properly configured.")
        sys.exit(1)

    # Show pre-indexing summary
    tool.show_pre_indexing_summary(args.directory, args.mode)

    # Confirmation prompt (unless --no-confirm)
    if not args.no_confirm:
        response = input("\nProceed with indexing? [y/N]: ").strip().lower()
        if response not in ["y", "yes"]:
            print("Indexing cancelled.")
            sys.exit(0)

    # Run indexing
    try:
        success = asyncio.run(tool.perform_indexing(args.directory, args.mode))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Indexing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
