"""Base implementation class for MCP prompts.

This module provides the base class that all prompt implementations inherit from.
"""

import logging
from typing import List, Optional, Any
from abc import ABC, abstractmethod

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base


class BasePromptImplementation(ABC):
    """Base class for all MCP prompt implementations."""
    
    def __init__(self, prompts_system):
        """Initialize the prompt implementation.
        
        Args:
            prompts_system: The MCPPromptsSystem instance
        """
        self.prompts_system = prompts_system
        self.logger = logging.getLogger(self.__class__.__name__)
        self.indexing_service = prompts_system.indexing_service
        self.analysis_service = prompts_system.analysis_service
        self.embedding_service = prompts_system.embedding_service
    
    @abstractmethod
    def register(self, mcp_app: FastMCP) -> None:
        """Register the prompt with the MCP app.
        
        Args:
            mcp_app: The FastMCP application instance
        """
        pass
    
    def create_message(self, content: str, role: str = "user") -> base.Message:
        """Create a standard MCP message.
        
        Args:
            content: The text content of the message
            role: The role of the message sender
            
        Returns:
            base.Message: A formatted MCP message
        """
        return base.Message(
            role=role,
            content=[base.TextContent(text=content)]
        )
    
    def create_error_message(self, error: Exception, context: str) -> List[base.Message]:
        """Create an error message for fallback scenarios.
        
        Args:
            error: The exception that occurred
            context: Context about where the error occurred
            
        Returns:
            List[base.Message]: Error message wrapped in a list
        """
        error_content = f"⚠️ Error in {context}: {str(error)}\n\n"
        error_content += "Falling back to basic functionality.\n"
        return [self.create_message(error_content)]