"""Understand Component Prompt Implementation.

This module implements the understand_component prompt for component analysis.
"""

from mcp.server.fastmcp import FastMCP

from ..base import BasePromptImplementation


class UnderstandComponentPrompt(BasePromptImplementation):
    """Implementation of the understand_component prompt."""

    def register(self, mcp_app: FastMCP) -> None:
        """Register the understand_component prompt with the MCP app."""
        # TODO: Migrate implementation from mcp_prompts.py
        pass
