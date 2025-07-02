"""Suggest Next Steps Prompt Implementation.

This module implements the suggest_next_steps prompt for providing recommendations.
"""

from mcp.server.fastmcp import FastMCP

from ..base import BasePromptImplementation


class SuggestNextStepsPrompt(BasePromptImplementation):
    """Implementation of the suggest_next_steps prompt."""

    def register(self, mcp_app: FastMCP) -> None:
        """Register the suggest_next_steps prompt with the MCP app."""
        # TODO: Migrate implementation from mcp_prompts.py
        pass
