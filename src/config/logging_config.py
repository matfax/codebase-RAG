"""
Centralized Logging Configuration for MCP Codebase RAG Server

This module provides comprehensive file-based logging configuration with
structured output, rotation, and debug capabilities for effective tracing.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

# Default log directory
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_LEVEL = "INFO"
DEBUG_LOG_LEVEL = "DEBUG"


class MCP_LoggingConfig:
    """Centralized logging configuration manager."""

    def __init__(self, log_dir: Path | None = None, enable_debug: bool = False):
        self.log_dir = log_dir or DEFAULT_LOG_DIR
        self.enable_debug = enable_debug or os.getenv("MCP_DEBUG_LEVEL", "").upper() == "DEBUG"
        self.request_tracking = os.getenv("MCP_ENABLE_REQUEST_TRACKING", "false").lower() == "true"

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup comprehensive logging configuration."""

        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG if self.enable_debug else logging.INFO)

        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        simple_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")

        # 1. Main application log (rotating file)
        main_log_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "mcp_server.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 10MB
        )
        main_log_handler.setLevel(logging.INFO)
        main_log_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(main_log_handler)

        # 2. Debug log (only when debug enabled)
        if self.enable_debug:
            debug_log_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "mcp_debug.log", maxBytes=20 * 1024 * 1024, backupCount=3, encoding="utf-8"  # 20MB
            )
            debug_log_handler.setLevel(logging.DEBUG)
            debug_log_handler.setFormatter(detailed_formatter)
            root_logger.addHandler(debug_log_handler)

        # 3. Error log (errors and above)
        error_log_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "mcp_errors.log", maxBytes=5 * 1024 * 1024, backupCount=10, encoding="utf-8"  # 5MB
        )
        error_log_handler.setLevel(logging.ERROR)
        error_log_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_log_handler)

        # 4. Performance log (for performance monitoring)
        performance_logger = logging.getLogger("mcp.performance")
        performance_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "mcp_performance.log", maxBytes=15 * 1024 * 1024, backupCount=7, encoding="utf-8"  # 15MB
        )
        performance_handler.setLevel(logging.INFO)
        performance_handler.setFormatter(detailed_formatter)
        performance_logger.addHandler(performance_handler)
        performance_logger.setLevel(logging.INFO)
        performance_logger.propagate = False  # Don't duplicate to root logger

        # 5. Multi-modal search debug log
        multimodal_logger = logging.getLogger("mcp.multi_modal")
        multimodal_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "mcp_multimodal.log", maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"  # 10MB
        )
        multimodal_handler.setLevel(logging.DEBUG if self.enable_debug else logging.INFO)
        multimodal_handler.setFormatter(detailed_formatter)
        multimodal_logger.addHandler(multimodal_handler)
        multimodal_logger.setLevel(logging.DEBUG if self.enable_debug else logging.INFO)
        multimodal_logger.propagate = False

        # 6. Service calls log (for debugging service interactions)
        service_calls_logger = logging.getLogger("mcp.service_calls")
        service_calls_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "mcp_service_calls.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"  # 5MB
        )
        service_calls_handler.setLevel(logging.DEBUG if self.enable_debug else logging.INFO)
        service_calls_handler.setFormatter(detailed_formatter)
        service_calls_logger.addHandler(service_calls_handler)
        service_calls_logger.setLevel(logging.DEBUG if self.enable_debug else logging.INFO)
        service_calls_logger.propagate = False

        # 7. Configuration log (for auto-config and feature toggles)
        config_logger = logging.getLogger("mcp.configuration")
        config_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "mcp_configuration.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"  # 5MB
        )
        config_handler.setLevel(logging.INFO)
        config_handler.setFormatter(detailed_formatter)
        config_logger.addHandler(config_handler)
        config_logger.setLevel(logging.INFO)
        config_logger.propagate = False

        # 8. Console output (optional, for development)
        if self.enable_debug or os.getenv("MCP_CONSOLE_LOGGING", "false").lower() == "true":
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(simple_formatter)
            root_logger.addHandler(console_handler)

        # Log initialization
        logger = logging.getLogger("mcp.logging_config")
        logger.info(f"Logging initialized - Debug: {self.enable_debug}, Request Tracking: {self.request_tracking}")
        logger.info(f"Log files location: {self.log_dir.absolute()}")
        logger.info(
            "Available log files: mcp_server.log, mcp_errors.log, mcp_performance.log, mcp_multimodal.log, mcp_service_calls.log, mcp_configuration.log"
        )
        if self.enable_debug:
            logger.info("Debug logging enabled - mcp_debug.log will contain detailed debugging information")

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name."""
        return logging.getLogger(f"mcp.{name}")

    def enable_request_tracking(self, enable: bool = True):
        """Enable or disable request tracking logging."""
        self.request_tracking = enable
        logger = logging.getLogger("mcp.logging_config")
        logger.info(f"Request tracking {'enabled' if enable else 'disabled'}")

    def set_debug_level(self, component: str, level: str):
        """Set debug level for a specific component."""
        logger = logging.getLogger(f"mcp.{component}")
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.info(f"Log level for {component} set to {level.upper()}")

    def list_log_files(self) -> list[Path]:
        """List all current log files."""
        return list(self.log_dir.glob("*.log"))

    def get_log_stats(self) -> dict:
        """Get statistics about current log files."""
        stats = {}
        for log_file in self.list_log_files():
            if log_file.exists():
                stats[log_file.name] = {"size_mb": round(log_file.stat().st_size / (1024 * 1024), 2), "modified": log_file.stat().st_mtime}
        return stats


# Global logging configuration instance
_logging_config: MCP_LoggingConfig | None = None


def init_logging(log_dir: Path | None = None, enable_debug: bool = None) -> MCP_LoggingConfig:
    """Initialize centralized logging configuration."""
    global _logging_config

    # Check environment variables if parameters not provided
    if enable_debug is None:
        enable_debug = os.getenv("MCP_DEBUG_LEVEL", "").upper() == "DEBUG"

    if log_dir is None:
        log_dir_env = os.getenv("MCP_LOG_DIR")
        if log_dir_env:
            log_dir = Path(log_dir_env)

    _logging_config = MCP_LoggingConfig(log_dir=log_dir, enable_debug=enable_debug)
    return _logging_config


def get_logging_config() -> MCP_LoggingConfig:
    """Get the current logging configuration, initializing if needed."""
    global _logging_config
    if _logging_config is None:
        _logging_config = init_logging()
    return _logging_config


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a properly configured logger."""
    config = get_logging_config()
    return config.get_logger(name)


# Convenience loggers for common components
def get_search_logger() -> logging.Logger:
    """Get logger for search tools."""
    return get_logger("search_tools")


def get_multimodal_logger() -> logging.Logger:
    """Get logger for multi-modal search."""
    return get_logger("multi_modal")


def get_performance_logger() -> logging.Logger:
    """Get logger for performance monitoring."""
    return get_logger("performance")


def get_service_logger() -> logging.Logger:
    """Get logger for service calls."""
    return get_logger("service_calls")


def get_config_logger() -> logging.Logger:
    """Get logger for configuration."""
    return get_logger("configuration")


# Auto-initialize logging when module is imported
# This ensures logging is available immediately
try:
    init_logging()
except Exception as e:
    # Fallback to basic console logging if file logging fails
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.getLogger("mcp.logging_config").warning(f"Failed to initialize file logging: {e}")
