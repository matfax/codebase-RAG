"""Advanced Search Prompt Implementation.

This module implements the advance_search prompt for guided cross-project search operations.
"""

import sys
from pathlib import Path

# Add src directory to path for absolute imports
if str(Path(__file__).parent.parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

from ..base import BasePromptImplementation


class AdvanceSearchPrompt(BasePromptImplementation):
    """Implementation of the advance_search prompt."""

    def register(self, mcp_app: FastMCP) -> None:
        """Register the advance_search prompt with the MCP app."""

        @mcp_app.prompt()
        def advance_search(query: str, search_mode: str = "hybrid", detail_level: str = "standard") -> list[base.Message]:
            """
            Perform guided cross-project search with intelligent project selection.

            This prompt provides a user-friendly interface for advanced search operations,
            guiding users through project selection and executing optimized searches with
            clear project attribution and enhanced context.

            Args:
                query: The search query to execute
                search_mode: Search strategy - "semantic", "keyword", or "hybrid" (default: "hybrid")
                detail_level: Level of detail - "brief", "standard", or "comprehensive" (default: "standard")
            """
            try:
                # First, check if any projects are indexed
                project_discovery_result = self._discover_indexed_projects()

                if "error" in project_discovery_result:
                    return self._create_error_response(project_discovery_result["error"])

                if project_discovery_result["total_projects"] == 0:
                    return self._create_no_projects_response()

                # Build comprehensive search guidance prompt
                search_prompt = self._build_search_guidance_prompt(query, project_discovery_result, search_mode, detail_level)

                return [self.create_message(search_prompt)]

            except Exception as e:
                self.logger.error(f"Error in advance_search prompt: {e}")
                return self._create_fallback_response(query, search_mode, str(e))

    def _discover_indexed_projects(self) -> dict:
        """Discover all indexed projects and their information."""
        try:
            # Use the project utilities to list indexed projects
            from src.tools.project.project_utils import list_indexed_projects

            return list_indexed_projects()

        except Exception as e:
            self.logger.error(f"Failed to discover indexed projects: {e}")
            return {"error": str(e)}

    def _build_search_guidance_prompt(self, query: str, project_info: dict, search_mode: str, detail_level: str) -> str:
        """Build the main search guidance prompt."""

        total_projects = project_info["total_projects"]
        projects = project_info["projects"]

        # Calculate total indexed items across all projects
        total_items = sum(project["total_points"] for project in projects)

        # Create project summary
        project_summary = self._format_project_summary(projects, detail_level)

        prompt = f"""🔍 **Advanced Cross-Project Search Interface**

**Search Query:** "{query}"
**Search Mode:** {search_mode}
**Detail Level:** {detail_level}

📊 **Indexed Projects Overview:**
- **Total Projects:** {total_projects}
- **Total Indexed Items:** {total_items:,}
- **Content Types Available:** {self._get_available_content_types(projects)}

{project_summary}

🎯 **Search Strategy Recommendations:**

**Option 1: Current Project Search (Default)**
```
Use: search(query="{query}", search_mode="{search_mode}")
```
- Searches only within your current project context
- Fastest and most focused results
- Best for project-specific queries

**Option 2: Targeted Multi-Project Search**
```
Use: search(query="{query}", search_mode="{search_mode}", target_projects=["project1", "project2"])
```
- Search specific projects that are most relevant to your query
- Recommended projects for your query: {self._recommend_projects_for_query(query, projects)}
- Balances scope with relevance

**Option 3: Comprehensive Cross-Project Search**
```
Use: search(query="{query}", search_mode="{search_mode}", cross_project=true)
```
- Searches across all {total_projects} indexed projects
- Most comprehensive but may include irrelevant results
- Best for discovering patterns across your entire codebase ecosystem

🚀 **Next Steps:**

1. **Review Project List:** Examine the projects above and consider which are most relevant to your query
2. **Choose Search Scope:** Select one of the three search strategies based on your needs
3. **Execute Search:** Run the search command with your chosen parameters
4. **Refine if Needed:** Based on initial results, you can adjust project selection or search terms

💡 **Pro Tips:**
- Start with targeted multi-project search for best balance of scope and relevance
- Use semantic search mode for concept-based queries
- Use keyword search mode for specific function/class names
- Add `include_context=true` for more detailed code context in results

Would you like me to execute one of these search strategies, or would you prefer to customize the search parameters?"""

        return prompt

    def _format_project_summary(self, projects: list[dict], detail_level: str) -> str:
        """Format the project summary based on detail level."""
        if detail_level == "brief":
            return self._format_brief_project_list(projects)
        elif detail_level == "comprehensive":
            return self._format_comprehensive_project_list(projects)
        else:  # standard
            return self._format_standard_project_list(projects)

    def _format_brief_project_list(self, projects: list[dict]) -> str:
        """Format a brief project list."""
        project_names = [f"`{project['name']}`" for project in projects]
        return f"**Available Projects:** {', '.join(project_names)}"

    def _format_standard_project_list(self, projects: list[dict]) -> str:
        """Format a standard project list with key information."""
        lines = ["📁 **Available Projects:**"]

        for project in projects:
            content_types = project.get("collection_types", [])
            lines.append(
                f"- **{project['name']}** ({project['total_points']:,} items) "
                f"- Content: {', '.join(content_types) if content_types else 'code'}"
            )

        return "\n".join(lines)

    def _format_comprehensive_project_list(self, projects: list[dict]) -> str:
        """Format a comprehensive project list with detailed information."""
        lines = ["📁 **Available Projects (Detailed):**"]

        for project in projects:
            lines.append(f"\n**🔹 {project['name']}**")
            lines.append(f"  - Total Items: {project['total_points']:,}")
            lines.append(f"  - Collections: {len(project['collections'])}")

            # Show collection breakdown
            collection_details = project.get("collection_details", [])
            if collection_details:
                for detail in collection_details[:3]:  # Show first 3 collections
                    lines.append(f"    • {detail['type']}: {detail['points_count']:,} items")
                if len(collection_details) > 3:
                    lines.append(f"    • ... and {len(collection_details) - 3} more")

        return "\n".join(lines)

    def _get_available_content_types(self, projects: list[dict]) -> str:
        """Get a summary of available content types across all projects."""
        all_types = set()
        for project in projects:
            content_types = project.get("collection_types", [])
            all_types.update(content_types)

        return ", ".join(sorted(all_types)) if all_types else "code"

    def _recommend_projects_for_query(self, query: str, projects: list[dict]) -> str:
        """Provide simple project recommendations based on query content."""
        # Simple keyword-based recommendations
        query_lower = query.lower()
        recommendations = []

        # Look for obvious matches in project names
        for project in projects:
            project_name = project["name"].lower()
            if any(word in project_name for word in query_lower.split() if len(word) > 3):
                recommendations.append(project["name"])

        # If no obvious matches, recommend projects with most content
        if not recommendations:
            sorted_projects = sorted(projects, key=lambda p: p["total_points"], reverse=True)
            recommendations = [p["name"] for p in sorted_projects[:3]]

        # Limit to top 3 recommendations
        recommendations = recommendations[:3]

        return ", ".join(f"`{name}`" for name in recommendations)

    def _create_no_projects_response(self) -> list[base.Message]:
        """Create response when no projects are indexed."""
        content = """🔍 **Advanced Search - No Projects Found**

❌ **No indexed projects detected in your RAG system.**

To use advanced cross-project search, you need to index some projects first.

🚀 **Quick Setup:**

1. **Index your current project:**
   ```
   index_directory(directory=".", recursive=true)
   ```

2. **Index additional projects:**
   ```
   index_directory(directory="/path/to/other/project", recursive=true)
   ```

3. **Check indexing status:**
   ```
   check_index_status(directory=".")
   ```

Once you have indexed projects, return to `advance_search` for guided cross-project search capabilities.

💡 **Tip:** Start with indexing your most important projects to build a knowledge base for cross-project searches."""

        return [self.create_message(content)]

    def _create_error_response(self, error_msg: str) -> list[base.Message]:
        """Create error response."""
        content = f"""🔍 **Advanced Search - Error**

❌ **Error discovering indexed projects:** {error_msg}

🔧 **Troubleshooting:**
1. Ensure Qdrant is running and accessible
2. Check that you have indexed projects using `index_directory`
3. Verify your RAG system configuration

You can still perform basic searches using:
```
search(query="your query here")
```

For setup help, see the project documentation or use `health_check()` to diagnose issues."""

        return [self.create_message(content)]

    def _create_fallback_response(self, query: str, search_mode: str, error_msg: str) -> list[base.Message]:
        """Create fallback response when advanced features fail."""
        content = f"""🔍 **Advanced Search - Fallback Mode**

⚠️ **Advanced search features encountered an issue:** {error_msg}

**Falling back to basic search guidance for query:** "{query}"

🎯 **Basic Search Options:**

**Current Project Search:**
```
search(query="{query}", search_mode="{search_mode}")
```

**Cross-Project Search:**
```
search(query="{query}", search_mode="{search_mode}", cross_project=true)
```

**With Context:**
```
search(query="{query}", search_mode="{search_mode}", include_context=true, context_chunks=2)
```

💡 **Next Steps:**
1. Try one of the basic search commands above
2. Use `check_index_status()` to verify your indexing setup
3. Use `list_indexed_projects_tool()` to see available projects"""

        return [self.create_message(content)]
