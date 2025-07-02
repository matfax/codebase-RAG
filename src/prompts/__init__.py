"""MCP Prompts System

This module provides the prompts system for the Codebase RAG MCP server.
"""

from .registry import MCPPromptsSystem

__all__ = ["MCPPromptsSystem"]


def register_prompts(app):
    """Register all prompts with the MCP app.

    Args:
        app: The FastMCP app instance

    Returns:
        MCPPromptsSystem: The initialized prompts system
    """
    prompts_system = MCPPromptsSystem(app)
    prompts_system.register_all_prompts()
    return prompts_system
