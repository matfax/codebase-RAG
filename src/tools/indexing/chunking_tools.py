"""Chunking and parsing tools for intelligent code analysis.

This module provides tools for monitoring and managing the intelligent code chunking system
that uses Tree-sitter AST parsing for syntax-aware code analysis.
"""

import logging
import traceback
from datetime import datetime
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.tools.core.error_utils import handle_tool_error, log_tool_usage
from src.tools.core.errors import ChunkingError

# Configure logging
logger = logging.getLogger(__name__)


def get_chunking_metrics(language: str | None = None, export_path: str | None = None) -> dict[str, Any]:
    """
    Get comprehensive chunking performance metrics.

    This tool provides detailed metrics about intelligent code chunking performance,
    including success rates per language, processing speeds, error rates, and quality metrics.

    Args:
        language: Optional specific language to get metrics for (e.g., "python", "javascript")
        export_path: Optional path to export detailed metrics to a JSON file

    Returns:
        Dictionary with chunking performance metrics and statistics
    """
    with log_tool_usage("get_chunking_metrics", {"language": language, "export_path": bool(export_path)}):
        try:
            from src.services.code_parser_service import CodeParserService

            parser_service = CodeParserService()

            # Get performance summary from parser service
            performance_report = parser_service.get_performance_summary()

            # Get comprehensive metrics from tracker
            from src.utils.chunking_metrics_tracker import chunking_metrics_tracker

            if language:
                # Get language-specific metrics
                lang_metrics = chunking_metrics_tracker.get_language_metrics(language)
                if not lang_metrics:
                    return {
                        "error": f"No metrics found for language: {language}",
                        "available_languages": list(parser_service.get_supported_languages()),
                        "performance_summary": performance_report,
                        "timestamp": datetime.now().isoformat(),
                    }

                logger.info(
                    f"Retrieved metrics for {language}: "
                    f"{lang_metrics.success_rate:.1f}% success rate, "
                    f"{lang_metrics.total_files} files processed"
                )

                result = {
                    "language": language,
                    "language_metrics": {
                        "total_files": lang_metrics.total_files,
                        "successful_files": lang_metrics.successful_files,
                        "failed_files": lang_metrics.failed_files,
                        "success_rate": lang_metrics.success_rate,
                        "total_chunks": lang_metrics.total_chunks,
                        "average_processing_time_ms": lang_metrics.average_processing_time_ms,
                        "chunk_types": lang_metrics.chunk_types,
                    },
                    "performance_summary": performance_report,
                    "timestamp": datetime.now().isoformat(),
                }

            else:
                # Get comprehensive metrics for all languages
                all_metrics = chunking_metrics_tracker.get_all_metrics()

                logger.info(
                    f"Retrieved comprehensive chunking metrics: "
                    f"{all_metrics['global']['total_operations']} total operations, "
                    f"{all_metrics['global']['overall_success_rate']:.1f}% overall success rate"
                )

                result = {
                    "comprehensive_metrics": all_metrics,
                    "performance_summary": performance_report,
                    "timestamp": datetime.now().isoformat(),
                }

            # Export metrics if requested
            if export_path:
                try:
                    import json

                    with open(export_path, "w") as f:
                        json.dump(result, f, indent=2, default=str)
                    result["export_status"] = f"Metrics exported to {export_path}"
                    logger.info(f"Metrics exported to {export_path}")
                except Exception as export_error:
                    result["export_error"] = f"Failed to export: {str(export_error)}"
                    logger.error(f"Failed to export metrics: {export_error}")

            return result

        except Exception as e:
            error_msg = f"Failed to get chunking metrics: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ChunkingError(error_msg) from e


def reset_chunking_metrics() -> dict[str, Any]:
    """
    Reset session-specific chunking metrics.

    This tool resets the current session metrics while preserving historical data.
    Useful for starting fresh performance measurements for a new indexing session.

    Returns:
        Confirmation of metrics reset
    """
    with log_tool_usage("reset_chunking_metrics", {}):
        try:
            from src.services.code_parser_service import CodeParserService

            parser_service = CodeParserService()
            parser_service.reset_session_metrics()

            logger.info("Chunking session metrics reset successfully")

            return {
                "status": "success",
                "message": "Session metrics have been reset",
                "timestamp": datetime.now().isoformat(),
                "note": "Historical metrics are preserved, only session counters were reset",
            }

        except Exception as e:
            error_msg = f"Failed to reset chunking metrics: {str(e)}"
            logger.error(error_msg)
            raise ChunkingError(error_msg) from e


def get_chunking_metrics_sync(language: str | None = None, export_path: str | None = None) -> dict[str, Any]:
    """Synchronous wrapper for get_chunking_metrics."""
    return handle_tool_error(get_chunking_metrics, language=language, export_path=export_path)


def reset_chunking_metrics_sync() -> dict[str, Any]:
    """Synchronous wrapper for reset_chunking_metrics."""
    return handle_tool_error(reset_chunking_metrics)


def register_chunking_tools(mcp_app: FastMCP):
    """Register chunking-related MCP tools."""

    @mcp_app.tool()
    def get_chunking_metrics_tool(language: str | None = None, export_path: str | None = None) -> dict[str, Any]:
        """
        Get comprehensive chunking performance metrics.

        This tool provides detailed metrics about intelligent code chunking performance,
        including success rates per language, processing speeds, error rates, and quality metrics.

        Args:
            language: Optional specific language to get metrics for (e.g., "python", "javascript")
            export_path: Optional path to export detailed metrics to a JSON file

        Returns:
            Dictionary with chunking performance metrics and statistics
        """
        return get_chunking_metrics_sync(language, export_path)

    @mcp_app.tool()
    def reset_chunking_metrics_tool() -> dict[str, Any]:
        """
        Reset session-specific chunking metrics.

        This tool resets the current session metrics while preserving historical data.
        Useful for starting fresh performance measurements for a new indexing session.

        Returns:
            Confirmation of metrics reset
        """
        return reset_chunking_metrics_sync()
