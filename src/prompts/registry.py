"""MCP Prompts System Registry

This module implements the main registry for MCP prompts.
"""

import logging

from mcp.server.fastmcp import FastMCP
from services.embedding_service import EmbeddingService
from services.indexing_service import IndexingService
from services.project_analysis_service import ProjectAnalysisService

from .advanced_search import AdvanceSearchPrompt

# Import prompt implementations
from .exploration import (
    ExploreProjectPrompt,
    FindEntryPointsPrompt,
    TraceFunctionalityPrompt,
    UnderstandComponentPrompt,
)
from .recommendation import OptimizeSearchPrompt, SuggestNextStepsPrompt

logger = logging.getLogger(__name__)


class MCPPromptsSystem:
    """
    Core MCP Prompts system providing intelligent workflow guidance.

    This system registers and manages intelligent prompts that help users
    explore codebases, understand components, trace functionality, and
    receive smart recommendations for next steps.
    """

    def __init__(self, mcp_app: FastMCP):
        self.mcp_app = mcp_app
        self.logger = logger
        self.indexing_service = None
        self.analysis_service = None
        self.embedding_service = None
        self._initialize_services()

    def _initialize_services(self):
        """Initialize required services for prompt operations."""
        try:
            self.indexing_service = IndexingService()
            self.analysis_service = ProjectAnalysisService()
            self.embedding_service = EmbeddingService()
            self.logger.info("MCP Prompts services initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP Prompts services: {e}")
            raise

    def register_all_prompts(self):
        """Register all MCP prompts with the FastMCP app."""
        self.logger.info("Registering MCP Prompts...")

        # Core exploration prompts
        self._register_prompt(ExploreProjectPrompt(self))
        self._register_prompt(UnderstandComponentPrompt(self))
        self._register_prompt(TraceFunctionalityPrompt(self))
        self._register_prompt(FindEntryPointsPrompt(self))

        # Smart recommendation prompts
        self._register_prompt(SuggestNextStepsPrompt(self))
        self._register_prompt(OptimizeSearchPrompt(self))

        # Advanced search prompts
        self._register_prompt(AdvanceSearchPrompt(self))

        self.logger.info("All MCP Prompts registered successfully")

    def _register_prompt(self, prompt_instance):
        """Register a single prompt with the MCP app.

        Args:
            prompt_instance: Instance of a prompt class
        """
        try:
            prompt_instance.register(self.mcp_app)
            self.logger.info(f"Registered prompt: {prompt_instance.__class__.__name__}")
        except Exception as e:
            self.logger.error(f"Failed to register prompt {prompt_instance.__class__.__name__}: {e}")
            raise
