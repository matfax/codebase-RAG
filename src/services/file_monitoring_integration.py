"""
File monitoring integration service.

This service manages the integration between file monitoring and cache invalidation,
ensuring that file system changes automatically trigger appropriate cache invalidations
with real-time responsiveness and optimized performance.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Union

from ..services.cache_invalidation_service import CacheInvalidationService, get_cache_invalidation_service
from ..services.file_monitoring_service import (
    FileMonitoringService,
    MonitoringMode,
    ProjectMonitoringConfig,
    get_file_monitoring_service,
)


class FileMonitoringIntegration:
    """
    Integration service for file monitoring and cache invalidation.

    This service provides:
    - Automatic setup of file monitoring for indexed projects
    - Seamless integration between monitoring and invalidation services
    - Configuration management for monitoring settings
    - Health monitoring and error recovery
    """

    def __init__(
        self,
        invalidation_service: CacheInvalidationService | None = None,
        monitoring_service: FileMonitoringService | None = None,
    ):
        """Initialize the file monitoring integration."""
        self.logger = logging.getLogger(__name__)
        self._invalidation_service = invalidation_service
        self._monitoring_service = monitoring_service
        self._is_initialized = False

    async def initialize(self) -> None:
        """Initialize the file monitoring integration."""
        try:
            # Get service instances
            if self._invalidation_service is None:
                self._invalidation_service = await get_cache_invalidation_service()

            if self._monitoring_service is None:
                self._monitoring_service = await get_file_monitoring_service()

            # Register integration
            self._invalidation_service.register_file_monitoring_integration(self._monitoring_service)

            self._is_initialized = True
            self.logger.info("File monitoring integration initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize file monitoring integration: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the file monitoring integration."""
        try:
            self._is_initialized = False
            self.logger.info("File monitoring integration shutdown successfully")

        except Exception as e:
            self.logger.error(f"Error during file monitoring integration shutdown: {e}")

    async def setup_project_monitoring(
        self,
        project_name: str,
        root_directory: str,
        auto_detect: bool = True,
        monitoring_config: dict[str, Any] | None = None,
    ) -> ProjectMonitoringConfig:
        """
        Set up file monitoring for a project.

        Args:
            project_name: Name of the project
            root_directory: Root directory of the project
            auto_detect: Whether to auto-detect project characteristics
            monitoring_config: Optional monitoring configuration overrides

        Returns:
            ProjectMonitoringConfig: The created monitoring configuration
        """
        if not self._is_initialized:
            raise RuntimeError("File monitoring integration not initialized")

        try:
            # Auto-detect project characteristics if requested
            if auto_detect:
                project_info = await self._detect_project_characteristics(root_directory)
                monitoring_config = self._merge_config(project_info, monitoring_config or {})

            # Create monitoring configuration
            config = self._monitoring_service.create_project_config(
                project_name=project_name, root_directory=root_directory, **(monitoring_config or {})
            )

            # Enable real-time invalidation
            await self._invalidation_service.enable_real_time_invalidation(project_name, enable=True)

            self.logger.info(f"Set up file monitoring for project: {project_name}")
            return config

        except Exception as e:
            self.logger.error(f"Failed to setup project monitoring for {project_name}: {e}")
            raise

    async def remove_project_monitoring(self, project_name: str) -> None:
        """
        Remove file monitoring for a project.

        Args:
            project_name: Name of the project
        """
        if not self._is_initialized:
            raise RuntimeError("File monitoring integration not initialized")

        try:
            # Disable real-time invalidation
            await self._invalidation_service.enable_real_time_invalidation(project_name, enable=False)

            # Remove monitoring
            self._monitoring_service.remove_project_monitoring(project_name)

            self.logger.info(f"Removed file monitoring for project: {project_name}")

        except Exception as e:
            self.logger.error(f"Failed to remove project monitoring for {project_name}: {e}")
            raise

    async def trigger_manual_scan(self, project_name: str) -> dict[str, Any]:
        """
        Trigger manual scan for a project.

        Args:
            project_name: Name of the project

        Returns:
            Dictionary with scan results
        """
        if not self._is_initialized:
            raise RuntimeError("File monitoring integration not initialized")

        try:
            # Perform manual scan
            changes = await self._monitoring_service.manual_project_scan(project_name)

            # Get scan results
            result = {
                "project_name": project_name,
                "changes_detected": changes.has_changes,
                "summary": changes.get_summary(),
                "files_to_reindex": changes.get_files_to_reindex(),
                "files_to_remove": changes.get_files_to_remove(),
            }

            self.logger.info(f"Manual scan completed for project {project_name}: {changes.get_summary()}")
            return result

        except Exception as e:
            self.logger.error(f"Failed manual scan for project {project_name}: {e}")
            raise

    async def get_integration_status(self) -> dict[str, Any]:
        """
        Get status of file monitoring integration.

        Returns:
            Dictionary with integration status
        """
        status = {
            "initialized": self._is_initialized,
            "invalidation_service_available": self._invalidation_service is not None,
            "monitoring_service_available": self._monitoring_service is not None,
            "monitored_projects": [],
            "monitoring_stats": None,
        }

        if self._is_initialized and self._monitoring_service:
            # Get monitoring stats
            monitoring_stats = self._monitoring_service.get_monitoring_stats()
            status["monitoring_stats"] = {
                "total_events": monitoring_stats.total_events,
                "invalidations_triggered": monitoring_stats.invalidations_triggered,
                "errors": monitoring_stats.errors,
                "last_event": monitoring_stats.last_event.isoformat() if monitoring_stats.last_event else None,
                "monitoring_start_time": (
                    monitoring_stats.monitoring_start_time.isoformat() if monitoring_stats.monitoring_start_time else None
                ),
            }

            # Get monitored projects
            configs = self._monitoring_service.get_project_configs()
            for project_name, config in configs.items():
                project_status = {
                    "project_name": project_name,
                    "root_directory": config.root_directory,
                    "polling_interval": config.polling_interval,
                    "batch_threshold": config.batch_threshold,
                    "enable_real_time": config.enable_real_time,
                    "enable_polling": config.enable_polling,
                }

                # Get real-time status from invalidation service
                if self._invalidation_service:
                    real_time_status = await self._invalidation_service.get_real_time_status(project_name)
                    project_status["real_time_status"] = real_time_status

                status["monitored_projects"].append(project_status)

        return status

    async def configure_monitoring_mode(self, mode: MonitoringMode) -> None:
        """
        Configure monitoring mode.

        Args:
            mode: Monitoring mode to set
        """
        if not self._is_initialized:
            raise RuntimeError("File monitoring integration not initialized")

        self._monitoring_service.set_monitoring_mode(mode)
        self.logger.info(f"Set monitoring mode to: {mode.value}")

    async def update_project_config(self, project_name: str, config_updates: dict[str, Any]) -> ProjectMonitoringConfig:
        """
        Update monitoring configuration for a project.

        Args:
            project_name: Name of the project
            config_updates: Configuration updates to apply

        Returns:
            Updated project monitoring configuration
        """
        if not self._is_initialized:
            raise RuntimeError("File monitoring integration not initialized")

        try:
            # Get current configuration
            configs = self._monitoring_service.get_project_configs()
            if project_name not in configs:
                raise ValueError(f"Project {project_name} is not being monitored")

            current_config = configs[project_name]

            # Create updated configuration
            updated_config = ProjectMonitoringConfig(
                project_name=current_config.project_name,
                root_directory=current_config.root_directory,
                file_patterns=config_updates.get("file_patterns", current_config.file_patterns),
                exclude_patterns=config_updates.get("exclude_patterns", current_config.exclude_patterns),
                polling_interval=config_updates.get("polling_interval", current_config.polling_interval),
                batch_threshold=config_updates.get("batch_threshold", current_config.batch_threshold),
                batch_timeout=config_updates.get("batch_timeout", current_config.batch_timeout),
                enable_real_time=config_updates.get("enable_real_time", current_config.enable_real_time),
                enable_polling=config_updates.get("enable_polling", current_config.enable_polling),
                invalidation_delay=config_updates.get("invalidation_delay", current_config.invalidation_delay),
                max_file_size=config_updates.get("max_file_size", current_config.max_file_size),
            )

            # Remove and re-add monitoring with new configuration
            self._monitoring_service.remove_project_monitoring(project_name)
            self._monitoring_service.add_project_monitoring(updated_config)

            self.logger.info(f"Updated monitoring configuration for project: {project_name}")
            return updated_config

        except Exception as e:
            self.logger.error(f"Failed to update project config for {project_name}: {e}")
            raise

    async def trigger_file_invalidation(self, file_path: str, project_name: str | None = None, force: bool = False) -> None:
        """
        Manually trigger cache invalidation for a file.

        Args:
            file_path: Path to the file
            project_name: Project name (auto-detected if None)
            force: Force invalidation even if file not monitored
        """
        if not self._is_initialized:
            raise RuntimeError("File monitoring integration not initialized")

        await self._monitoring_service.trigger_file_invalidation(file_path, project_name, force=force)

    async def _detect_project_characteristics(self, root_directory: str) -> dict[str, Any]:
        """
        Auto-detect project characteristics for optimized monitoring.

        Args:
            root_directory: Root directory of the project

        Returns:
            Dictionary with detected project characteristics
        """
        characteristics = {
            "file_patterns": [],
            "exclude_patterns": [],
            "polling_interval": 5.0,
            "batch_threshold": 5,
        }

        try:
            root_path = Path(root_directory)
            if not root_path.exists():
                return characteristics

            # Detect project type
            project_type = None
            if (root_path / "package.json").exists():
                project_type = "nodejs"
            elif (root_path / "requirements.txt").exists() or (root_path / "pyproject.toml").exists():
                project_type = "python"
            elif (root_path / "pom.xml").exists() or (root_path / "build.gradle").exists():
                project_type = "java"
            elif (root_path / "Cargo.toml").exists():
                project_type = "rust"
            elif (root_path / "go.mod").exists():
                project_type = "go"

            # Set project-specific configurations
            if project_type == "nodejs":
                characteristics["exclude_patterns"].extend(["node_modules/*", "dist/*", "build/*"])
                characteristics["polling_interval"] = 3.0  # Faster for web development
            elif project_type == "python":
                characteristics["exclude_patterns"].extend(["__pycache__/*", "*.pyc", ".venv/*", "venv/*"])
            elif project_type == "java":
                characteristics["exclude_patterns"].extend(["target/*", "build/*", "*.class"])
                characteristics["polling_interval"] = 10.0  # Slower for compiled languages
            elif project_type == "rust":
                characteristics["exclude_patterns"].extend(["target/*", "Cargo.lock"])
                characteristics["polling_interval"] = 8.0
            elif project_type == "go":
                characteristics["exclude_patterns"].extend(["vendor/*", "*.exe", "*.so"])

            # Detect project size for batch threshold adjustment
            file_count = sum(1 for _ in root_path.rglob("*") if _.is_file())
            if file_count > 1000:
                characteristics["batch_threshold"] = 10  # Higher threshold for large projects
            elif file_count < 100:
                characteristics["batch_threshold"] = 2  # Lower threshold for small projects

            self.logger.debug(f"Detected project type: {project_type}, file count: {file_count}")

        except Exception as e:
            self.logger.warning(f"Failed to detect project characteristics for {root_directory}: {e}")

        return characteristics

    def _merge_config(self, detected: dict[str, Any], provided: dict[str, Any]) -> dict[str, Any]:
        """
        Merge detected and provided configurations.

        Args:
            detected: Auto-detected configuration
            provided: User-provided configuration

        Returns:
            Merged configuration
        """
        merged = detected.copy()

        for key, value in provided.items():
            if key in ["file_patterns", "exclude_patterns"] and isinstance(value, list):
                # Extend lists for pattern-based configurations
                merged[key] = list(set(merged.get(key, []) + value))
            else:
                # Override for other configurations
                merged[key] = value

        return merged


# Global file monitoring integration instance
_file_monitoring_integration: FileMonitoringIntegration | None = None


async def get_file_monitoring_integration() -> FileMonitoringIntegration:
    """
    Get the global file monitoring integration instance.

    Returns:
        FileMonitoringIntegration: The global integration instance
    """
    global _file_monitoring_integration
    if _file_monitoring_integration is None:
        _file_monitoring_integration = FileMonitoringIntegration()
        await _file_monitoring_integration.initialize()
    return _file_monitoring_integration


async def shutdown_file_monitoring_integration() -> None:
    """Shutdown the global file monitoring integration."""
    global _file_monitoring_integration
    if _file_monitoring_integration:
        await _file_monitoring_integration.shutdown()
        _file_monitoring_integration = None
