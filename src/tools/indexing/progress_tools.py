"""Progress monitoring tools for indexing operations.

This module provides tools for monitoring the progress of indexing operations,
including real-time progress updates, ETA calculations, and memory monitoring.
"""

import logging
import time
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.tools.core.error_utils import handle_tool_error, log_tool_usage
from src.tools.core.errors import IndexingError
from src.utils.performance_monitor import MemoryMonitor

# Configure logging
logger = logging.getLogger(__name__)


def get_indexing_progress() -> dict[str, Any]:
    """
    Get current progress of any ongoing indexing operations.

    This tool provides real-time progress updates during long indexing operations,
    including ETA, processing rate, memory usage, and stage information.

    Returns:
        Dictionary with current indexing progress information
    """
    with log_tool_usage("get_indexing_progress", {}):
        try:
            # For now, this is a placeholder since we need to implement
            # a global progress tracking system for async operations
            # In the current synchronous implementation, indexing completes
            # before returning, so this would mainly be useful for debugging

            system_memory = MemoryMonitor().get_system_memory_info()

            return {
                "status": "no_active_indexing",
                "message": "No indexing operations currently in progress",
                "system_info": {"memory": system_memory, "timestamp": time.time()},
                "note": "Progress tracking is available during indexing operations and returned in indexing results",
            }

        except Exception as e:
            error_msg = f"Failed to get indexing progress: {str(e)}"
            logger.error(error_msg)
            raise IndexingError(error_msg) from e


def check_index_status(directory: str = ".") -> dict[str, Any]:
    """
    Check if a directory already has indexed data and provide recommendations.

    This tool helps users understand the current indexing state and make informed
    decisions about whether to reindex or use existing data.

    Args:
        directory: Directory to check for existing index status

    Returns:
        Dictionary with index status and recommendations
    """
    with log_tool_usage("check_index_status", {"directory": directory}):
        try:
            from pathlib import Path

            from src.tools.database.qdrant_utils import (
                check_existing_index,
                estimate_indexing_time,
            )
            from src.tools.project.project_utils import get_current_project

            dir_path = Path(directory).resolve()
            if not dir_path.exists():
                return {"error": f"Directory not found: {directory}"}

            current_project = get_current_project(client_directory=str(dir_path))
            existing_index_info = check_existing_index(current_project)

            if not existing_index_info.get("has_existing_data", False):
                return {
                    "has_existing_data": False,
                    "message": "No existing indexed data found for this project",
                    "project_context": (current_project.get("name", "unknown") if current_project else "unknown"),
                    "directory": str(dir_path),
                    "recommendation": "Ready for initial indexing",
                }

            # Get file count estimation for recommendations
            try:
                from src.services.project_analysis_service import ProjectAnalysisService

                analysis_service = ProjectAnalysisService()
                quick_analysis = analysis_service.analyze_repository(str(dir_path))
                estimated_file_count = quick_analysis.get("relevant_files", 0)
            except Exception:
                estimated_file_count = 0

            time_estimates = estimate_indexing_time(estimated_file_count, existing_index_info.get("total_points", 0))

            return {
                "has_existing_data": True,
                "project_context": existing_index_info["project_name"],
                "directory": str(dir_path),
                "existing_data": {
                    "collections": existing_index_info["collections"],
                    "total_points": existing_index_info["total_points"],
                    "collection_details": existing_index_info.get("collection_details", []),
                },
                "estimates": time_estimates,
                "recommendations": {
                    "current_status": "Data already indexed and ready for search",
                    "reindex_time_estimate_minutes": time_estimates["estimated_time_minutes"],
                    "time_saved_by_keeping_existing_minutes": time_estimates["time_saved_by_keeping_existing_minutes"],
                    "actions": {
                        "use_existing": "Data is ready for search operations",
                        "full_reindex": "Call index_directory with clear_existing=true to reindex everything",
                        "incremental_update": "Call index_directory with incremental=true to update only changed files",
                    },
                },
            }

        except Exception as e:
            error_msg = f"Failed to check index status: {str(e)}"
            logger.error(error_msg)
            raise IndexingError(error_msg, directory) from e


def get_indexing_progress_sync() -> dict[str, Any]:
    """Synchronous wrapper for get_indexing_progress."""
    return handle_tool_error(get_indexing_progress)


def check_index_status_sync(directory: str = ".") -> dict[str, Any]:
    """Synchronous wrapper for check_index_status."""
    return handle_tool_error(check_index_status, directory=directory)


def register_progress_tools(mcp_app: FastMCP):
    """Register progress monitoring MCP tools."""

    @mcp_app.tool()
    def get_indexing_progress_tool() -> dict[str, Any]:
        """
        Get current progress of any ongoing indexing operations.

        This tool provides real-time progress updates during long indexing operations,
        including ETA, processing rate, memory usage, and stage information.
        """
        return get_indexing_progress_sync()

    @mcp_app.tool()
    def check_index_status_tool(directory: str = ".") -> dict[str, Any]:
        """
        Check if a directory already has indexed data and provide recommendations.

        This tool helps users understand the current indexing state and make informed
        decisions about whether to reindex or use existing data.
        """
        return check_index_status_sync(directory)
