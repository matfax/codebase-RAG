"""Base classes and utilities for MCP prompts.

This module provides the foundation for all MCP prompts.
"""

from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp.prompts import base


@dataclass
class PromptArgument:
    """Represents an argument for a prompt."""

    name: str
    type: type
    description: str
    default: Any = None
    required: bool = True


@dataclass
class PromptMessage:
    """Represents a message in a prompt response."""

    role: str
    content: str


class BasePrompt:
    """Base class for all MCP prompts."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.arguments: list[PromptArgument] = []

    def add_argument(self, argument: PromptArgument) -> None:
        """Add an argument to the prompt."""
        self.arguments.append(argument)

    def create_message(self, content: str, role: str = "user") -> base.Message:
        """Create a standard MCP message."""
        return base.Message(content=content, role=role)

    def create_messages(self, contents: list[str], role: str = "user") -> list[base.Message]:
        """Create multiple MCP messages."""
        return [self.create_message(content, role) for content in contents]
