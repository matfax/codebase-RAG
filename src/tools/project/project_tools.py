"""Project management tools for MCP.

This module provides tools for managing project information, collections,
and project-level operations.
"""

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP
from tools.core.error_utils import handle_tool_error, log_tool_usage
from tools.core.errors import ProjectError
from tools.project.project_utils import clear_project_collections, get_current_project

# Configure logging
logger = logging.getLogger(__name__)


def get_project_info(directory: str = ".") -> dict[str, Any]:
    """
    Get information about the current project.

    Args:
        directory: Directory to analyze for project information

    Returns:
        Dictionary with project information
    """
    with log_tool_usage("get_project_info", {"directory": directory}):
        try:
            from pathlib import Path

            dir_path = Path(directory).resolve()
            if not dir_path.exists():
                return {"error": f"Directory not found: {directory}"}

            project_info = get_current_project(client_directory=str(dir_path))

            if not project_info:
                return {
                    "error": "No project detected",
                    "directory": str(dir_path),
                    "message": "Could not detect project boundaries",
                }

            # Get additional project statistics
            from tools.database.qdrant_utils import check_existing_index

            index_info = check_existing_index(project_info)

            return {
                "project_name": project_info["name"],
                "project_root": project_info["root"],
                "collection_prefix": project_info["collection_prefix"],
                "directory": str(dir_path),
                "has_indexed_data": index_info.get("has_existing_data", False),
                "collections": index_info.get("collections", []),
                "total_points": index_info.get("total_points", 0),
                "collection_details": index_info.get("collection_details", []),
            }

        except Exception as e:
            error_msg = f"Failed to get project info: {str(e)}"
            logger.error(error_msg)
            raise ProjectError(error_msg) from e


def list_indexed_projects() -> dict[str, Any]:
    """
    List all projects that have indexed data.

    Returns:
        Dictionary with information about all indexed projects
    """
    with log_tool_usage("list_indexed_projects", {}):
        try:
            from tools.database.qdrant_utils import get_qdrant_client

            client = get_qdrant_client()
            collections = client.get_collections().collections

            # Group collections by project
            projects = {}

            for collection in collections:
                name = collection.name

                # Skip metadata collection
                if name.endswith("_metadata"):
                    continue

                # Parse project name from collection name
                if name.startswith("project_"):
                    parts = name.split("_")
                    if len(parts) >= 3:
                        project_name = "_".join(parts[1:-1])  # Everything between 'project_' and last '_type'
                        collection_type = parts[-1]

                        if project_name not in projects:
                            projects[project_name] = {
                                "name": project_name,
                                "collections": {},
                                "total_points": 0,
                            }

                        # Get collection info
                        try:
                            collection_info = client.get_collection(name)
                            point_count = collection_info.points_count

                            projects[project_name]["collections"][collection_type] = {
                                "name": name,
                                "points": point_count,
                            }
                            projects[project_name]["total_points"] += point_count

                        except Exception as e:
                            logger.warning(f"Could not get info for collection {name}: {e}")
                            projects[project_name]["collections"][collection_type] = {
                                "name": name,
                                "points": 0,
                                "error": str(e),
                            }

                elif name.startswith("dir_") or name.startswith("global_"):
                    # Handle directory-based and global collections
                    if name.startswith("dir_"):
                        project_name = name[4:].rsplit("_", 1)[0]  # Remove 'dir_' prefix and last '_type'
                        collection_type = name.rsplit("_", 1)[-1]
                    else:
                        project_name = "global"
                        collection_type = name[7:]  # Remove 'global_' prefix

                    if project_name not in projects:
                        projects[project_name] = {
                            "name": project_name,
                            "collections": {},
                            "total_points": 0,
                        }

                    try:
                        collection_info = client.get_collection(name)
                        point_count = collection_info.points_count

                        projects[project_name]["collections"][collection_type] = {
                            "name": name,
                            "points": point_count,
                        }
                        projects[project_name]["total_points"] += point_count

                    except Exception as e:
                        logger.warning(f"Could not get info for collection {name}: {e}")
                        projects[project_name]["collections"][collection_type] = {
                            "name": name,
                            "points": 0,
                            "error": str(e),
                        }

            project_list = list(projects.values())
            project_list.sort(key=lambda x: x["total_points"], reverse=True)

            return {
                "projects": project_list,
                "total_projects": len(project_list),
                "total_collections": len(collections),
                "summary": {
                    "total_points": sum(p["total_points"] for p in project_list),
                    "largest_project": (project_list[0]["name"] if project_list else None),
                    "largest_project_points": (project_list[0]["total_points"] if project_list else 0),
                },
            }

        except Exception as e:
            error_msg = f"Failed to list indexed projects: {str(e)}"
            logger.error(error_msg)
            raise ProjectError(error_msg) from e


def clear_project_data(project_name: str | None = None, directory: str = ".") -> dict[str, Any]:
    """
    Clear all indexed data for a project.

    Args:
        project_name: Optional specific project name to clear
        directory: Directory to determine project context if project_name not provided

    Returns:
        Dictionary with clearing results
    """
    with log_tool_usage("clear_project_data", {"project_name": project_name, "directory": directory}):
        try:
            if project_name:
                # Clear specific project by name
                from tools.database.qdrant_utils import get_qdrant_client

                client = get_qdrant_client()
                collections = [c.name for c in client.get_collections().collections]

                cleared = []
                errors = []

                # Find collections for the specified project
                for collection_name in collections:
                    if collection_name.startswith(f"project_{project_name}_"):
                        try:
                            client.delete_collection(collection_name)
                            cleared.append(collection_name)
                            logger.info(f"Cleared collection: {collection_name}")
                        except Exception as e:
                            error_msg = f"Failed to clear {collection_name}: {str(e)}"
                            errors.append(error_msg)
                            logger.error(error_msg)

                return {
                    "project": project_name,
                    "cleared_collections": cleared,
                    "errors": errors if errors else None,
                    "success": len(cleared) > 0,
                }
            else:
                # Clear current project
                return clear_project_collections()

        except Exception as e:
            error_msg = f"Failed to clear project data: {str(e)}"
            logger.error(error_msg)
            raise ProjectError(error_msg) from e


def get_project_info_sync(directory: str = ".") -> dict[str, Any]:
    """Synchronous wrapper for get_project_info."""
    return handle_tool_error(get_project_info, directory=directory)


def list_indexed_projects_sync() -> dict[str, Any]:
    """Synchronous wrapper for list_indexed_projects."""
    return handle_tool_error(list_indexed_projects)


def clear_project_data_sync(project_name: str | None = None, directory: str = ".") -> dict[str, Any]:
    """Synchronous wrapper for clear_project_data."""
    return handle_tool_error(clear_project_data, project_name=project_name, directory=directory)


def register_project_tools(mcp_app: FastMCP):
    """Register project management MCP tools."""

    @mcp_app.tool()
    def get_project_info_tool(directory: str = ".") -> dict[str, Any]:
        """
        Get information about the current project.

        Args:
            directory: Directory to analyze for project information

        Returns:
            Dictionary with project information
        """
        return get_project_info_sync(directory)

    @mcp_app.tool()
    def list_indexed_projects_tool() -> dict[str, Any]:
        """
        List all projects that have indexed data.

        Returns:
            Dictionary with information about all indexed projects
        """
        return list_indexed_projects_sync()

    @mcp_app.tool()
    def clear_project_data_tool(project_name: str | None = None, directory: str = ".") -> dict[str, Any]:
        """
        Clear all indexed data for a project.

        Args:
            project_name: Optional specific project name to clear
            directory: Directory to determine project context if project_name not provided

        Returns:
            Dictionary with clearing results
        """
        return clear_project_data_sync(project_name, directory)
