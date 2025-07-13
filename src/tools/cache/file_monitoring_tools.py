"""
MCP tools for file monitoring and real-time cache invalidation.

This module provides tools to manage file monitoring, configure real-time
cache invalidation, and monitor file system integration status.
"""

import asyncio
import logging
from typing import Any

from src.services.file_monitoring_integration import get_file_monitoring_integration
from src.services.file_monitoring_service import MonitoringMode

logger = logging.getLogger(__name__)


async def setup_project_monitoring_tool(
    project_name: str,
    root_directory: str,
    auto_detect: bool = True,
    file_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    polling_interval: float | None = None,
    batch_threshold: int | None = None,
    enable_real_time: bool = True,
    enable_polling: bool = True,
) -> dict[str, Any]:
    """
    Set up file monitoring for a project with real-time cache invalidation.

    Args:
        project_name: Name of the project to monitor
        root_directory: Root directory of the project
        auto_detect: Whether to auto-detect project characteristics
        file_patterns: File patterns to monitor (e.g., ["*.py", "*.js"])
        exclude_patterns: File patterns to exclude (e.g., ["*.pyc", "node_modules/*"])
        polling_interval: Polling interval in seconds (default: 5.0)
        batch_threshold: Number of changes to trigger batch processing (default: 5)
        enable_real_time: Enable real-time monitoring
        enable_polling: Enable polling-based monitoring

    Returns:
        Dictionary with setup results and configuration
    """
    try:
        integration = await get_file_monitoring_integration()

        # Build monitoring configuration
        monitoring_config = {}
        if file_patterns is not None:
            monitoring_config["file_patterns"] = file_patterns
        if exclude_patterns is not None:
            monitoring_config["exclude_patterns"] = exclude_patterns
        if polling_interval is not None:
            monitoring_config["polling_interval"] = polling_interval
        if batch_threshold is not None:
            monitoring_config["batch_threshold"] = batch_threshold
        if enable_real_time is not None:
            monitoring_config["enable_real_time"] = enable_real_time
        if enable_polling is not None:
            monitoring_config["enable_polling"] = enable_polling

        # Set up monitoring
        config = await integration.setup_project_monitoring(
            project_name=project_name,
            root_directory=root_directory,
            auto_detect=auto_detect,
            monitoring_config=monitoring_config,
        )

        return {
            "success": True,
            "project_name": project_name,
            "root_directory": root_directory,
            "configuration": {
                "file_patterns": config.file_patterns,
                "exclude_patterns": config.exclude_patterns,
                "polling_interval": config.polling_interval,
                "batch_threshold": config.batch_threshold,
                "enable_real_time": config.enable_real_time,
                "enable_polling": config.enable_polling,
                "max_file_size": config.max_file_size,
            },
            "message": f"File monitoring set up successfully for project: {project_name}",
        }

    except Exception as e:
        logger.error(f"Failed to setup project monitoring: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to set up file monitoring for project: {project_name}",
        }


async def remove_project_monitoring_tool(project_name: str) -> dict[str, Any]:
    """
    Remove file monitoring for a project.

    Args:
        project_name: Name of the project to stop monitoring

    Returns:
        Dictionary with removal results
    """
    try:
        integration = await get_file_monitoring_integration()
        await integration.remove_project_monitoring(project_name)

        return {
            "success": True,
            "project_name": project_name,
            "message": f"File monitoring removed successfully for project: {project_name}",
        }

    except Exception as e:
        logger.error(f"Failed to remove project monitoring: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to remove file monitoring for project: {project_name}",
        }


async def get_monitoring_status_tool(project_name: str | None = None) -> dict[str, Any]:
    """
    Get file monitoring status for all projects or a specific project.

    Args:
        project_name: Optional project name to get specific status

    Returns:
        Dictionary with monitoring status information
    """
    try:
        integration = await get_file_monitoring_integration()
        status = await integration.get_integration_status()

        if project_name:
            # Filter for specific project
            project_status = None
            for project in status.get("monitored_projects", []):
                if project["project_name"] == project_name:
                    project_status = project
                    break

            if project_status:
                return {
                    "success": True,
                    "project_name": project_name,
                    "status": project_status,
                    "integration_status": {
                        "initialized": status["initialized"],
                        "monitoring_stats": status["monitoring_stats"],
                    },
                }
            else:
                return {
                    "success": False,
                    "project_name": project_name,
                    "message": f"Project {project_name} is not being monitored",
                }
        else:
            # Return full status
            return {
                "success": True,
                "integration_status": status,
                "monitored_projects_count": len(status.get("monitored_projects", [])),
            }

    except Exception as e:
        logger.error(f"Failed to get monitoring status: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to get monitoring status",
        }


async def trigger_manual_scan_tool(project_name: str) -> dict[str, Any]:
    """
    Trigger manual file system scan for a project.

    Args:
        project_name: Name of the project to scan

    Returns:
        Dictionary with scan results
    """
    try:
        integration = await get_file_monitoring_integration()
        results = await integration.trigger_manual_scan(project_name)

        return {
            "success": True,
            "project_name": project_name,
            "scan_results": results,
            "message": f"Manual scan completed for project: {project_name}",
        }

    except Exception as e:
        logger.error(f"Failed to trigger manual scan: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to trigger manual scan for project: {project_name}",
        }


async def configure_monitoring_mode_tool(mode: str) -> dict[str, Any]:
    """
    Configure global file monitoring mode.

    Args:
        mode: Monitoring mode ("polling", "hybrid", "disabled")

    Returns:
        Dictionary with configuration results
    """
    try:
        # Validate mode
        try:
            monitoring_mode = MonitoringMode(mode.lower())
        except ValueError:
            valid_modes = [m.value for m in MonitoringMode]
            return {
                "success": False,
                "error": f"Invalid monitoring mode: {mode}",
                "valid_modes": valid_modes,
                "message": f"Mode must be one of: {', '.join(valid_modes)}",
            }

        integration = await get_file_monitoring_integration()
        await integration.configure_monitoring_mode(monitoring_mode)

        return {
            "success": True,
            "mode": mode,
            "message": f"Monitoring mode set to: {mode}",
        }

    except Exception as e:
        logger.error(f"Failed to configure monitoring mode: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to configure monitoring mode: {mode}",
        }


async def update_project_config_tool(
    project_name: str,
    file_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    polling_interval: float | None = None,
    batch_threshold: int | None = None,
    enable_real_time: bool | None = None,
    enable_polling: bool | None = None,
    max_file_size: int | None = None,
) -> dict[str, Any]:
    """
    Update monitoring configuration for a project.

    Args:
        project_name: Name of the project to update
        file_patterns: File patterns to monitor
        exclude_patterns: File patterns to exclude
        polling_interval: Polling interval in seconds
        batch_threshold: Number of changes to trigger batch processing
        enable_real_time: Enable real-time monitoring
        enable_polling: Enable polling-based monitoring
        max_file_size: Maximum file size to monitor (in bytes)

    Returns:
        Dictionary with update results
    """
    try:
        integration = await get_file_monitoring_integration()

        # Build configuration updates
        config_updates = {}
        if file_patterns is not None:
            config_updates["file_patterns"] = file_patterns
        if exclude_patterns is not None:
            config_updates["exclude_patterns"] = exclude_patterns
        if polling_interval is not None:
            config_updates["polling_interval"] = polling_interval
        if batch_threshold is not None:
            config_updates["batch_threshold"] = batch_threshold
        if enable_real_time is not None:
            config_updates["enable_real_time"] = enable_real_time
        if enable_polling is not None:
            config_updates["enable_polling"] = enable_polling
        if max_file_size is not None:
            config_updates["max_file_size"] = max_file_size

        if not config_updates:
            return {
                "success": False,
                "message": "No configuration updates provided",
            }

        # Update configuration
        updated_config = await integration.update_project_config(project_name, config_updates)

        return {
            "success": True,
            "project_name": project_name,
            "updated_configuration": {
                "file_patterns": updated_config.file_patterns,
                "exclude_patterns": updated_config.exclude_patterns,
                "polling_interval": updated_config.polling_interval,
                "batch_threshold": updated_config.batch_threshold,
                "enable_real_time": updated_config.enable_real_time,
                "enable_polling": updated_config.enable_polling,
                "max_file_size": updated_config.max_file_size,
            },
            "message": f"Configuration updated successfully for project: {project_name}",
        }

    except Exception as e:
        logger.error(f"Failed to update project config: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to update configuration for project: {project_name}",
        }


async def trigger_file_invalidation_tool(file_path: str, project_name: str | None = None, force: bool = False) -> dict[str, Any]:
    """
    Manually trigger cache invalidation for a specific file.

    Args:
        file_path: Path to the file to invalidate
        project_name: Project name (auto-detected if not provided)
        force: Force invalidation even if file not monitored

    Returns:
        Dictionary with invalidation results
    """
    try:
        integration = await get_file_monitoring_integration()
        await integration.trigger_file_invalidation(file_path, project_name, force)

        return {
            "success": True,
            "file_path": file_path,
            "project_name": project_name,
            "force": force,
            "message": f"Cache invalidation triggered for file: {file_path}",
        }

    except Exception as e:
        logger.error(f"Failed to trigger file invalidation: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to trigger cache invalidation for file: {file_path}",
        }


def register_file_monitoring_tools(server) -> None:
    """
    Register file monitoring tools with the MCP server.

    Args:
        server: The MCP server instance
    """

    @server.tool()
    async def setup_project_monitoring(
        project_name: str,
        root_directory: str,
        auto_detect: bool = True,
        file_patterns: list[str] = None,
        exclude_patterns: list[str] = None,
        polling_interval: float = None,
        batch_threshold: int = None,
        enable_real_time: bool = True,
        enable_polling: bool = True,
    ) -> dict[str, Any]:
        """Set up file monitoring for a project with real-time cache invalidation.

        Args:
            project_name: Name of the project to monitor
            root_directory: Root directory of the project
            auto_detect: Whether to auto-detect project characteristics
            file_patterns: File patterns to monitor (e.g., ["*.py", "*.js"])
            exclude_patterns: File patterns to exclude (e.g., ["*.pyc", "node_modules/*"])
            polling_interval: Polling interval in seconds (default: 5.0)
            batch_threshold: Number of changes to trigger batch processing (default: 5)
            enable_real_time: Enable real-time monitoring
            enable_polling: Enable polling-based monitoring
        """
        return await setup_project_monitoring_tool(
            project_name=project_name,
            root_directory=root_directory,
            auto_detect=auto_detect,
            file_patterns=file_patterns,
            exclude_patterns=exclude_patterns,
            polling_interval=polling_interval,
            batch_threshold=batch_threshold,
            enable_real_time=enable_real_time,
            enable_polling=enable_polling,
        )

    @server.tool()
    async def remove_project_monitoring(project_name: str) -> dict[str, Any]:
        """Remove file monitoring for a project.

        Args:
            project_name: Name of the project to stop monitoring
        """
        return await remove_project_monitoring_tool(project_name)

    @server.tool()
    async def get_monitoring_status(project_name: str = None) -> dict[str, Any]:
        """Get file monitoring status for all projects or a specific project.

        Args:
            project_name: Optional project name to get specific status
        """
        return await get_monitoring_status_tool(project_name)

    @server.tool()
    async def trigger_manual_scan(project_name: str) -> dict[str, Any]:
        """Trigger manual file system scan for a project.

        Args:
            project_name: Name of the project to scan
        """
        return await trigger_manual_scan_tool(project_name)

    @server.tool()
    async def configure_monitoring_mode(mode: str) -> dict[str, Any]:
        """Configure global file monitoring mode.

        Args:
            mode: Monitoring mode ("polling", "hybrid", "disabled")
        """
        return await configure_monitoring_mode_tool(mode)

    @server.tool()
    async def update_project_monitoring_config(
        project_name: str,
        file_patterns: list[str] = None,
        exclude_patterns: list[str] = None,
        polling_interval: float = None,
        batch_threshold: int = None,
        enable_real_time: bool = None,
        enable_polling: bool = None,
        max_file_size: int = None,
    ) -> dict[str, Any]:
        """Update monitoring configuration for a project.

        Args:
            project_name: Name of the project to update
            file_patterns: File patterns to monitor
            exclude_patterns: File patterns to exclude
            polling_interval: Polling interval in seconds
            batch_threshold: Number of changes to trigger batch processing
            enable_real_time: Enable real-time monitoring
            enable_polling: Enable polling-based monitoring
            max_file_size: Maximum file size to monitor (in bytes)
        """
        return await update_project_config_tool(
            project_name=project_name,
            file_patterns=file_patterns,
            exclude_patterns=exclude_patterns,
            polling_interval=polling_interval,
            batch_threshold=batch_threshold,
            enable_real_time=enable_real_time,
            enable_polling=enable_polling,
            max_file_size=max_file_size,
        )

    @server.tool()
    async def trigger_file_invalidation(file_path: str, project_name: str = None, force: bool = False) -> dict[str, Any]:
        """Manually trigger cache invalidation for a specific file.

        Args:
            file_path: Path to the file to invalidate
            project_name: Project name (auto-detected if not provided)
            force: Force invalidation even if file not monitored
        """
        return await trigger_file_invalidation_tool(file_path, project_name, force)

    logger.info("Registered file monitoring MCP tools")
