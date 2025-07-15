"""Explore Project Prompt Implementation.

This module implements the explore_project prompt for comprehensive project analysis.
"""

import sys
from pathlib import Path

# Add src directory to path for absolute imports
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

from ..base import BasePromptImplementation


class ExploreProjectPrompt(BasePromptImplementation):
    """Implementation of the explore_project prompt."""

    def register(self, mcp_app: FastMCP) -> None:
        """Register the explore_project prompt with the MCP app."""

        @mcp_app.prompt()
        def explore_project(
            directory: str = ".",
            focus_area: str | None = None,
            detail_level: str = "overview",
        ) -> list[base.Message]:
            """
            Get a comprehensive guided exploration of project architecture and structure.

            This prompt analyzes the entire project to provide architectural insights,
            key module identification, dependency mapping, and strategic navigation guidance.
            Perfect for onboarding new developers or understanding unfamiliar codebases.

            Args:
                directory: Project directory to explore (default: current directory)
                focus_area: Specific area to focus on (e.g., "authentication", "data_layer", "api")
                detail_level: Level of detail - "overview", "detailed", or "comprehensive"
            """
            try:
                # Use enhanced project exploration service
                from services.project_exploration_service import (
                    ProjectExplorationService,
                )

                exploration_service = ProjectExplorationService()

                # Perform comprehensive project exploration
                exploration_result = exploration_service.explore_project(
                    project_path=directory,
                    focus_area=focus_area,
                    detail_level=detail_level,
                    include_dependencies=detail_level in ["detailed", "comprehensive"],
                    analyze_complexity=detail_level in ["detailed", "comprehensive"],
                )

                # Format the exploration results into a comprehensive prompt
                formatted_summary = exploration_service.format_exploration_summary(exploration_result, detail_level)

                # Create guided exploration prompt with rich analysis
                exploration_prompt = self._build_enhanced_exploration_prompt(
                    directory,
                    exploration_result,
                    focus_area,
                    detail_level,
                    formatted_summary,
                )

                return [self.create_message(exploration_prompt)]

            except Exception as e:
                self.logger.error(f"Error in explore_project prompt: {e}")
                # Fallback to basic exploration
                return self._create_fallback_exploration_prompt(directory, focus_area, detail_level, str(e))

    def _build_enhanced_exploration_prompt(
        self,
        directory: str,
        exploration_result,
        focus_area: str | None,
        detail_level: str,
        formatted_summary: str,
    ) -> str:
        """Build enhanced exploration prompt with comprehensive analysis."""
        base_prompt = f"""I need to explore and understand the codebase at '{directory}'. I've conducted a comprehensive analysis and here are the results:

{formatted_summary}

🔧 **First Steps - Check RAG Index Status:**
Before diving into exploration, please:
1. Use the `check_index_status` tool to see if this codebase is already indexed in the RAG system
2. If indexed, prioritize using `codebaseRAG:search` tool for efficient code discovery and analysis
3. If not indexed, consider indexing the codebase first for better exploration capabilities

🎯 **My Exploration Goals:**
I want a {detail_level} exploration that helps me understand this {exploration_result.project_type} project."""

        if focus_area:
            base_prompt += f"\n\n🔍 **Special Focus:** Please pay extra attention to '{focus_area}' related components and provide detailed insights about how this area fits into the overall architecture."

        base_prompt += """

🤖 **Please help me with:**

1. **Architecture Deep Dive:** Based on the analysis, explain how this project is structured and why this architecture was chosen
2. **Navigation Strategy:** Given the entry points and core modules identified, what's the most efficient way to explore this codebase?
3. **Key Insights:** What are the most important things I should understand about this project's design and implementation?
4. **Development Workflow:** How do the components work together? What's the typical data/request flow?
5. **Next Steps:** Based on what you see, what should I investigate next to build a solid understanding?

🚀 **Use the analysis results above to provide specific, actionable guidance that will help me quickly become productive with this project.**

Please search the codebase systematically using the identified entry points and core modules. **Prioritize using codebaseRAG:search tool** for efficient semantic search when the codebase is indexed, and provide insights that go beyond what the automated analysis discovered."""

        return base_prompt

    def _create_fallback_exploration_prompt(
        self,
        directory: str,
        focus_area: str | None,
        detail_level: str,
        error_msg: str,
    ) -> list[base.Message]:
        """Create fallback exploration prompt when enhanced analysis fails."""
        fallback_prompt = f"""I want to explore and understand the codebase at '{directory}' but encountered some analysis limitations: {error_msg}

🔧 **First Steps - Check RAG Index Status:**
Before diving into exploration, please:
1. Use the `check_index_status` tool to see if this codebase is already indexed in the RAG system
2. If indexed, prioritize using `codebaseRAG:search` tool for efficient code discovery and analysis
3. If not indexed, consider indexing the codebase first for better exploration capabilities

🎯 **Exploration Request:**
Please help me get a {detail_level} understanding of this project's structure and architecture."""

        if focus_area:
            fallback_prompt += f"\n\n🔍 **Focus Area:** I'm particularly interested in '{focus_area}' related functionality."

        fallback_prompt += """

📋 **Please help me by:**

1. **Project Structure Analysis:** Examine the directory structure and identify the main components
2. **Entry Point Discovery:** Find the main entry points (main.py, app.js, index files, etc.)
3. **Framework Detection:** Identify what frameworks and technologies are being used
4. **Architecture Pattern:** Determine the overall architecture pattern (MVC, microservices, etc.)
5. **Key Files Identification:** Point out the most important files to understand first
6. **Development Approach:** Suggest the best order to explore and learn this codebase

🚀 **Goal:** Provide me with actionable insights and a strategic approach to understanding this project quickly and effectively. **Prioritize using codebaseRAG:search tool** for efficient semantic search when the codebase is indexed."""

        return [self.create_message(fallback_prompt)]
