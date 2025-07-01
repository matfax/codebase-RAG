"""Backward compatibility wrapper for register_mcp_prompts.

This module provides the old import path for compatibility during migration.
"""

from prompts import register_prompts as register_mcp_prompts

__all__ = ["register_mcp_prompts"]