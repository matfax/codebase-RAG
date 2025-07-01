"""MCP Tools Registry

This module manages the registration of all MCP tools.
"""

import logging
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_tools(mcp_app: FastMCP) -> None:
    """
    Register all MCP tools with the FastMCP application.
    
    Args:
        mcp_app: The FastMCP application instance
    """
    # Register MCP Prompts system first
    try:
        from prompts import register_prompts
        prompts_system = register_prompts(mcp_app)
        logger.info("MCP Prompts system registered successfully")
    except Exception as e:
        logger.error(f"Failed to register MCP Prompts system: {e}")
        # Continue without prompts if there's an error
    
    logger.info("Registering MCP Tools...")
    
    # TODO: Register core tools
    # TODO: Register indexing tools  
    # TODO: Register project tools
    # TODO: Register database tools
    
    logger.info("All MCP Tools registered successfully")