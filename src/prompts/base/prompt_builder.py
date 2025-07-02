"""Common prompt building utilities.

This module provides utilities for constructing and formatting prompts.
"""

from typing import Any

from mcp.server.fastmcp.prompts import base


def build_prompt_message(content: str, role: str = "user") -> base.Message:
    """Build a standard MCP prompt message.

    Args:
        content: The text content of the message
        role: The role of the message sender (default: "user")

    Returns:
        base.Message: A formatted MCP message
    """
    return base.Message(content=content, role=role)


def format_prompt_response(results: list[dict[str, Any]], title: str, include_summary: bool = True) -> str:
    """Format prompt results into a readable response.

    Args:
        results: List of result dictionaries
        title: Title for the response
        include_summary: Whether to include a summary section

    Returns:
        str: Formatted response text
    """
    response_parts = [f"# {title}\n"]

    if include_summary and results:
        response_parts.append("## Summary\n")
        response_parts.append(f"Found {len(results)} results\n\n")

    for i, result in enumerate(results, 1):
        response_parts.append(f"## Result {i}\n")
        for key, value in result.items():
            if isinstance(value, list):
                response_parts.append(f"**{key}**:\n")
                for item in value:
                    response_parts.append(f"  - {item}\n")
            elif isinstance(value, dict):
                response_parts.append(f"**{key}**:\n")
                for k, v in value.items():
                    response_parts.append(f"  - {k}: {v}\n")
            else:
                response_parts.append(f"**{key}**: {value}\n")
        response_parts.append("\n")

    return "".join(response_parts)
