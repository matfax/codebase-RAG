"""Suggest Next Steps Prompt Implementation.

This module implements the suggest_next_steps prompt for providing recommendations.
"""

import sys
from pathlib import Path

# Add src directory to path for absolute imports
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP
from prompts.base import BasePromptImplementation


class SuggestNextStepsPrompt(BasePromptImplementation):
    """Implementation of the suggest_next_steps prompt."""

    def register(self, mcp_app: FastMCP) -> None:
        """Register the suggest_next_steps prompt with the MCP app."""
        # TODO: Migrate implementation from mcp_prompts.py
        pass
