#!/usr/bin/env python3
"""
Logging Initialization Script for MCP Codebase RAG Server

This script initializes the centralized logging system when the MCP server starts.
It should be imported early in the application startup process.
"""

import os
from pathlib import Path

from src.config.logging_config import get_logging_config, init_logging


def setup_mcp_logging():
    """Setup MCP logging system with environment-based configuration."""

    # Check environment variables for configuration
    debug_enabled = os.getenv("MCP_DEBUG_LEVEL", "").upper() == "DEBUG"
    log_dir = os.getenv("MCP_LOG_DIR", "logs")
    console_logging = os.getenv("MCP_CONSOLE_LOGGING", "false").lower() == "true"
    request_tracking = os.getenv("MCP_ENABLE_REQUEST_TRACKING", "false").lower() == "true"

    # Initialize logging configuration
    config = init_logging(log_dir=Path(log_dir), enable_debug=debug_enabled)

    # Enable request tracking if specified
    if request_tracking:
        config.enable_request_tracking(True)

    # Log startup information
    logger = config.get_logger("startup")
    logger.info("MCP Codebase RAG Server logging initialized")
    logger.info(f"Log directory: {config.log_dir.absolute()}")
    logger.info(f"Debug mode: {debug_enabled}")
    logger.info(f"Request tracking: {request_tracking}")
    logger.info(f"Console logging: {console_logging}")

    # Log available log files
    log_files = config.list_log_files()
    if log_files:
        logger.info(f"Active log files: {[f.name for f in log_files]}")

    # Log file stats
    stats = config.get_log_stats()
    if stats:
        logger.info(f"Log file sizes: {stats}")

    return config


# Auto-setup when imported
if __name__ == "__main__":
    setup_mcp_logging()
else:
    # Setup logging when this module is imported
    try:
        setup_mcp_logging()
    except Exception as e:
        # Fallback to basic logging if setup fails
        import logging

        logging.basicConfig(level=logging.INFO)
        logging.getLogger("mcp.startup").error(f"Failed to setup logging: {e}")
