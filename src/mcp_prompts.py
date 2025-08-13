"""
MCP Prompts System - Core Implementation

This module implements the MCP Prompts system for the Codebase RAG MCP Server,
providing intelligent guided workflows for AI agents and developers.

The system offers:
- Project exploration and analysis prompts
- Component understanding and tracing prompts
- Smart recommendation and optimization prompts
- Context-aware workflow orchestration
"""

import logging
from pathlib import Path
from typing import Union

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

from src.services.embedding_service import EmbeddingService
from src.services.indexing_service import IndexingService
from src.services.project_analysis_service import ProjectAnalysisService

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
        self._register_prompts()

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

    def _register_prompts(self):
        """Register all MCP prompts with the FastMCP app."""
        self.logger.info("Registering MCP Prompts...")

        # Core exploration prompts
        self._register_explore_project()
        self._register_understand_component()
        self._register_trace_functionality()
        self._register_find_entry_points()

        # Smart recommendation prompts
        self._register_suggest_next_steps()
        self._register_optimize_search()

        self.logger.info("All MCP Prompts registered successfully")

    def _register_explore_project(self):
        """Register the explore_project prompt."""

        @self.mcp_app.prompt()
        def explore_project(
            directory: str = ".",
            focus_area: str | None = None,
            detail_level: str = "overview",
        ) -> list[base.Message]:
            """
            Get a comprehensive guided exploration of project architecture and
            structure.

            This prompt analyzes the entire project to provide architectural
            insights, key module identification, dependency mapping, and strategic
            navigation guidance.
            Perfect for onboarding new developers or understanding unfamiliar codebases.

            Args:
                directory: Project directory to explore (default: current directory)
                focus_area: Specific area to focus on (e.g., "authentication",
                "data_layer", "api")
                detail_level: Level of detail - "overview", "detailed", or
                "comprehensive"
            """
            try:
                # Use enhanced project exploration service
                from src.services.project_exploration_service import (
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

                return [base.Message(content=exploration_prompt, role="user")]

            except Exception as e:
                self.logger.error(f"Error in explore_project prompt: {e}")
                # Fallback to basic exploration
                return self._create_fallback_exploration_prompt(directory, focus_area, detail_level, str(e))

    def _register_understand_component(self):
        """Register the understand_component prompt."""

        @self.mcp_app.prompt()
        def understand_component(
            component_name: str,
            component_type: str = "auto",
            include_dependencies: bool = True,
            include_usage_examples: bool = True,
        ) -> list[base.Message]:
            """
            Get in-depth understanding of a specific component, module, or function.

            This prompt provides detailed analysis of component responsibilities,
            interfaces, dependencies, and usage patterns. Ideal for understanding
            complex components before making modifications or integrating with them.

            Args:
                component_name: Name of component to analyze (class, function,
                module, file)
                component_type: Type hint - "class", "function", "module", "file",
                or "auto"
                include_dependencies: Whether to analyze dependencies and relationships
                include_usage_examples: Whether to provide usage examples and patterns
            """
            try:
                # Use enhanced component analysis service
                from src.services.component_analysis_service import ComponentAnalysisService

                analysis_service = ComponentAnalysisService()

                # Perform comprehensive component analysis
                analysis_result = analysis_service.analyze_component(
                    component_name=component_name,
                    project_path=".",
                    component_type=component_type,
                    include_dependencies=include_dependencies,
                    include_usage_examples=include_usage_examples,
                    analyze_quality=True,
                )

                # Format the analysis results into a comprehensive summary
                formatted_summary = analysis_service.format_analysis_summary(analysis_result, detail_level="detailed")

                # Create enhanced component understanding prompt
                understanding_prompt = self._build_enhanced_component_prompt(
                    component_name,
                    analysis_result,
                    include_dependencies,
                    include_usage_examples,
                    formatted_summary,
                )

                return [
                    base.Message(
                        role="user",
                        content=base.TextContent(type="text", text=understanding_prompt),
                    )
                ]

            except Exception as e:
                self.logger.error(f"Error in understand_component prompt: {e}")
                # Fallback to basic component analysis
                return self._create_fallback_component_prompt(
                    component_name,
                    component_type,
                    include_dependencies,
                    include_usage_examples,
                    str(e),
                )

    def _register_trace_functionality(self):
        """Register the trace_functionality prompt."""

        @self.mcp_app.prompt()
        def trace_functionality(
            functionality_description: str,
            trace_type: str = "full_flow",
            include_config: bool = True,
            include_data_flow: bool = True,
        ) -> list[base.Message]:
            """
            Trace the complete implementation path of specific functionality.

            This prompt traces functionality from API endpoints to database
            operations, identifying the complete call chain, configuration
            dependencies, and data flow.
            Essential for understanding complex feature implementations or debugging
            issues.

            Args:
                functionality_description: Description of functionality to trace
                (e.g., "user authentication", "file upload")
                trace_type: Type of trace - "full_flow", "api_to_db",
                "user_journey", or "data_pipeline"
                include_config: Whether to include configuration and environment
                dependencies
                include_data_flow: Whether to trace data transformation and flow
                patterns
            """
            try:
                # Use enhanced functionality tracing service
                from src.services.functionality_tracing_service import (
                    FunctionalityTracingService,
                )

                tracing_service = FunctionalityTracingService()

                # Perform comprehensive functionality tracing
                trace_result = tracing_service.trace_functionality(
                    functionality_description=functionality_description,
                    project_path=".",
                    trace_type=trace_type,
                    include_config=include_config,
                    include_data_flow=include_data_flow,
                    max_depth=10,
                )

                # Format the trace results into a comprehensive summary
                formatted_summary = tracing_service.format_trace_summary(trace_result, detail_level="detailed")

                # Create enhanced functionality tracing prompt
                tracing_prompt = self._build_enhanced_trace_prompt(
                    functionality_description,
                    trace_result,
                    trace_type,
                    include_config,
                    include_data_flow,
                    formatted_summary,
                )

                return [
                    base.Message(
                        role="user",
                        content=base.TextContent(type="text", text=tracing_prompt),
                    )
                ]

            except Exception as e:
                self.logger.error(f"Error in trace_functionality prompt: {e}")
                # Fallback to basic functionality tracing
                return self._create_fallback_trace_prompt(
                    functionality_description,
                    trace_type,
                    include_config,
                    include_data_flow,
                    str(e),
                )

    def _register_find_entry_points(self):
        """Register the find_entry_points prompt."""

        @self.mcp_app.prompt()
        def find_entry_points(
            entry_type: str = "all",
            learning_path: bool = True,
            include_examples: bool = True,
        ) -> list[base.Message]:
            """
            Identify and explain all main entry points into the application.

            This prompt discovers entry points like main functions, API routes,
            CLI commands, and other application starting points, providing a
            structured learning path for new developers to understand how the
            application is used and invoked.

            Args:
                entry_type: Type of entry points - "all", "api", "cli",
                "main_functions", or "scripts"
                learning_path: Whether to suggest an optimal learning/reading order
                include_examples: Whether to provide usage examples for each entry point
            """
            try:
                # Build entry points discovery prompt
                entry_points_prompt = self._build_entry_points_prompt(entry_type, learning_path, include_examples)

                return [
                    base.Message(
                        role="user",
                        content=base.TextContent(type="text", text=entry_points_prompt),
                    )
                ]

            except Exception as e:
                self.logger.error(f"Error in find_entry_points prompt: {e}")
                return [
                    base.Message(
                        content=(
                            "I need to find all the entry points for this application. "
                            "Can you help me identify main functions, API routes, "
                            "CLI commands, and other ways to start or interact with "
                            "this codebase?"
                        ),
                        role="user",
                    )
                ]

    def _register_suggest_next_steps(self):
        """Register the suggest_next_steps prompt."""

        @self.mcp_app.prompt()
        def suggest_next_steps(
            current_context: str,
            user_role: str = "developer",
            task_type: str = "exploration",
            difficulty_level: str = "intermediate",
        ) -> list[base.Message]:
            """
            Get intelligent recommendations for next actions based on current context.

            This prompt analyzes your current exploration state and provides
            personalized
            recommendations for next steps, considering your role, task objectives, and
            skill level. Helps maintain momentum and direction in complex codebase work.

            Args:
                current_context: Description of what you're currently working on or
                                 have discovered
                user_role: Your role - "developer", "architect", "reviewer",
                           "newcomer", or "debugger"
                task_type: Type of task - "exploration", "development", "refactoring",
                           "debugging", or "review"
                difficulty_level: Skill level - "beginner", "intermediate",
                                  or "advanced"
            """
            try:
                # Build next steps recommendation prompt
                next_steps_prompt = self._build_next_steps_prompt(current_context, user_role, task_type, difficulty_level)

                return [
                    base.Message(
                        role="user",
                        content=base.TextContent(type="text", text=next_steps_prompt),
                    )
                ]

            except Exception as e:
                self.logger.error(f"Error in suggest_next_steps prompt: {e}")
                return [
                    base.Message(
                        role="user",
                        content=base.TextContent(
                            text=(
                                f"Based on my current context: '{current_context}', "
                                f"can you suggest the best next steps for a {user_role} "
                                f"working on {task_type} tasks?"
                            ),
                        ),
                    )
                ]

    def _register_optimize_search(self):
        """Register the optimize_search prompt."""

        @self.mcp_app.prompt()
        def optimize_search(
            previous_searches: list[str],
            search_goal: str,
            refine_strategy: bool = True,
            suggest_alternatives: bool = True,
        ) -> list[base.Message]:
            """
            Optimize search strategies and suggest better approaches for finding
            information.

            This prompt analyzes your search patterns and results to recommend more
            effective search terms, strategies, and alternative approaches for finding
            the information you need in the codebase.

            Args:
                previous_searches: List of previous search queries you've tried
                search_goal: What you're ultimately trying to find or understand
                refine_strategy: Whether to suggest refined search terms and strategies
                suggest_alternatives: Whether to suggest alternative investigation
                                      approaches
            """
            try:
                # Build search optimization prompt
                search_optimization_prompt = self._build_search_optimization_prompt(
                    previous_searches,
                    search_goal,
                    refine_strategy,
                    suggest_alternatives,
                )

                return [
                    base.Message(
                        role="user",
                        content=base.TextContent(type="text", text=search_optimization_prompt),
                    )
                ]

            except Exception as e:
                self.logger.error(f"Error in optimize_search prompt: {e}")
                return [
                    base.Message(
                        role="user",
                        content=base.TextContent(
                            text=(
                                f"I've been searching for '{search_goal}' with these "
                                f"queries: {previous_searches}. Can you help me optimize "
                                f"my search strategy and suggest better approaches?"
                            ),
                        ),
                    )
                ]

    def _build_exploration_prompt(self, directory: str, stats: dict, focus_area: str | None, detail_level: str) -> str:
        """Build the project exploration prompt text."""
        base_prompt = (
            f"I need to explore and understand the codebase at '{directory}'. "
            f"Here's what I know about the project:\n\n"
            f"📊 **Project Statistics:**\n"
            f"- Total files: {stats['total_files']:,}\n"
            f"- Relevant code files: {stats['relevant_files']:,}\n"
            f"- Size: {stats['size_mb']} MB\n"
            f"- Complexity: {stats['complexity']}\n"
            f"- Languages: {', '.join(stats['languages'].keys()) if stats['languages'] else 'Not detected'}\n\n"
            f"🎯 **Exploration Goal:**\n"
            f"I want a {detail_level} exploration of this project's "
            f"architecture and structure."
        )

        if focus_area:
            base_prompt += f"\n\n🔍 **Focus Area:** Please pay special attention to '{focus_area}' " "related components and functionality."

        base_prompt += (
            "\n\n📋 **Please help me with:**\n"
            "1. **Architecture Overview:** What's the overall architecture pattern "
            "and project structure?\n"
            "2. **Key Components:** What are the most important modules, classes, "
            "and functions?\n"
            "3. **Entry Points:** How does the application start and what are the "
            "main interaction points?\n"
            "4. **Dependencies:** What are the key dependencies and how do "
            "components interact?\n"
            "5. **Navigation Strategy:** What's the best order to explore and "
            "understand this codebase?\n"
            "\n\n🚀 **Next Steps:** After this overview, suggest the most "
            "valuable next exploration steps based on what you discover.\n\n"
            "Please search the codebase systematically and provide actionable "
            "insights that will help me quickly become productive with this project."
        )

        return base_prompt

    def _build_component_understanding_prompt(
        self,
        component_name: str,
        component_type: str,
        include_deps: bool,
        include_examples: bool,
        search_query: str,
    ) -> str:
        """Build the component understanding prompt text."""
        base_prompt = (
            f"I need to deeply understand the component '{component_name}' in this codebase.\n\n"
            f"🎯 **Component to Analyze:** {component_name}\n"
            f"📝 **Component Type:** {component_type}\n\n"
            f"🔍 **Analysis Requirements:**\n"
            f"1. **Purpose & Responsibility:** What does this component do and why does it exist?\n"
            f"2. **Interface & API:** What are its public methods, parameters, and return types?\n"
            f"3. **Implementation Details:** How does it work internally?"
        )

        if include_deps:
            base_prompt += "\n4. **Dependencies:** What does it depend on and what depends on it?"

        if include_examples:
            base_prompt += "\n5. **Usage Examples:** How is it typically used? Show me real examples from the codebase."

        base_prompt += f"""

🔍 **Search Strategy:**
Please search for '{search_query}' and any related terms to find all relevant code.

📋 **Deliverables:**
- Clear explanation of the component's role in the system
- Documentation of all public interfaces
- Examples of how it's used throughout the codebase
- Recommendations for working with or modifying this component

Please search thoroughly and provide practical insights I can use immediately."""

        return base_prompt

    def _build_functionality_tracing_prompt(
        self,
        functionality: str,
        trace_type: str,
        include_config: bool,
        include_data_flow: bool,
    ) -> str:
        """Build the functionality tracing prompt text."""
        base_prompt = f"""I need to trace the complete implementation of '{functionality}' through this codebase.

🎯 **Functionality to Trace:** {functionality}
📊 **Trace Type:** {trace_type}

🔍 **Tracing Requirements:**
1. **Entry Points:** Where does this functionality start (API endpoints, user interfaces, etc.)?
2. **Call Chain:** What's the complete sequence of function/method calls?
3. **Core Logic:** Where is the main business logic implemented?
4. **Data Operations:** How does it interact with databases or external services?"""

        if include_config:
            base_prompt += "\n5. **Configuration:** What configuration settings and environment variables affect this functionality?"

        if include_data_flow:
            base_prompt += "\n6. **Data Flow:** How is data transformed as it flows through the system?"

        base_prompt += """

📋 **Trace Strategy:**
- Start from the most likely entry points
- Follow the execution path step by step
- Document key decision points and branches
- Identify error handling and edge cases

🎯 **Deliverables:**
- Complete execution flow diagram (in text/code format)
- Key files and functions involved
- Configuration dependencies
- Potential modification points

Please search systematically and map out the complete journey of this functionality."""

        return base_prompt

    def _build_entry_points_prompt(self, entry_type: str, learning_path: bool, include_examples: bool) -> str:
        """Build the entry points discovery prompt text."""
        base_prompt = f"""I need to discover all the entry points for this application.

🎯 **Entry Points to Find:** {entry_type}

🔍 **Discovery Requirements:**
1. **Main Functions:** Find main() functions, application startup code
2. **API Endpoints:** Discover REST APIs, GraphQL endpoints, or similar
3. **CLI Commands:** Find command-line interfaces and scripts
4. **Background Services:** Identify workers, schedulers, or daemon processes
5. **Web Interfaces:** Locate web server configurations and route definitions"""

        if learning_path:
            base_prompt += "\n\n📚 **Learning Path:** Please suggest the optimal order to explore these entry points for a new developer."

        if include_examples:
            base_prompt += "\n\n💡 **Usage Examples:** Provide examples of how to invoke each entry point."

        base_prompt += """

📋 **Search Strategy:**
- Look for common patterns: main(), app.py, server.py, cli.py
- Search for framework-specific patterns (Flask routes, FastAPI endpoints, etc.)
- Find configuration files that define entry points
- Identify startup scripts and service definitions

🎯 **Deliverables:**
- Complete list of all entry points with descriptions
- Instructions for running/invoking each entry point
- Suggested exploration order for new developers
- Dependencies and setup requirements

Please search comprehensively and provide a complete entry point guide."""

        return base_prompt

    def _build_next_steps_prompt(self, context: str, user_role: str, task_type: str, difficulty: str) -> str:
        """Build the next steps recommendation prompt text."""
        base_prompt = f"""Based on my current work context, I need intelligent recommendations for next steps.

🎯 **Current Context:** {context}
👤 **My Role:** {user_role}
📋 **Task Type:** {task_type}
🎚️ **Difficulty Level:** {difficulty}

🤔 **Help me decide what to do next by considering:**

1. **Progress Assessment:** What have I likely accomplished so far based on this context?
2. **Logical Next Steps:** What are the most logical next actions to take?
3. **Role-Specific Recommendations:** What would be most valuable for someone in my role?
4. **Task-Appropriate Actions:** What steps best serve my current task type?
5. **Skill-Level Matching:** What's appropriate for my difficulty level?

🎯 **Provide Recommendations For:**
- Immediate next actions (next 30 minutes)
- Short-term goals (next few hours)
- Learning opportunities based on what I've discovered
- Potential blockers or challenges to prepare for

💡 **Consider my context and suggest:**
- Specific searches to run
- Files or components to examine next
- Questions to investigate
- Skills or knowledge areas to develop

Please analyze my situation and provide actionable, prioritized recommendations."""

        return base_prompt

    def _build_search_optimization_prompt(
        self,
        previous_searches: list[str],
        goal: str,
        refine_strategy: bool,
        suggest_alternatives: bool,
    ) -> str:
        """Build the search optimization prompt text."""
        searches_text = "', '".join(previous_searches) if previous_searches else "none provided"

        base_prompt = f"""I need help optimizing my search strategy to find what I'm looking for.

🎯 **Search Goal:** {goal}
🔍 **Previous Searches:** ['{searches_text}']

📊 **Analyze my search approach:**
1. **Pattern Analysis:** What patterns do you see in my search terms?
2. **Gap Identification:** What might I be missing in my search strategy?
3. **Term Effectiveness:** Which terms are likely most/least effective?"""

        if refine_strategy:
            base_prompt += "\n4. **Strategy Refinement:** How can I improve my search terms and approach?"

        if suggest_alternatives:
            base_prompt += "\n5. **Alternative Approaches:** What other investigation methods might work better?"

        base_prompt += f"""

💡 **Please provide:**
- **Optimized Search Terms:** Better keywords and phrases to try
- **Search Strategies:** Different approaches (exact matches, wildcards, related terms)
- **Alternative Methods:** Other ways to find this information (file browsing, dependency analysis, etc.)
- **Context Clues:** What to look for that might lead to my goal

🔧 **Search Techniques to Consider:**
- Technical terminology vs. business terminology
- Implementation details vs. high-level concepts
- Configuration vs. code vs. documentation
- Current implementation vs. planned features

Help me find '{goal}' more effectively by improving my search approach."""

        return base_prompt

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

Please search the codebase systematically using the identified entry points and core modules, and provide insights that go beyond what the automated analysis discovered."""

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

🚀 **Goal:** Provide me with actionable insights and a strategic approach to understanding this project quickly and effectively."""

        return [base.Message(role="user", content=base.TextContent(type="text", text=fallback_prompt))]

    def _build_enhanced_component_prompt(
        self,
        component_name: str,
        analysis_result,
        include_dependencies: bool,
        include_usage_examples: bool,
        formatted_summary: str,
    ) -> str:
        """Build enhanced component understanding prompt with comprehensive analysis."""
        base_prompt = f"""I need to deeply understand the component '{component_name}' in this codebase. I've conducted a comprehensive analysis and here are the results:

{formatted_summary}

🎯 **My Understanding Goals:**
Based on the analysis, I want to fully comprehend how this {analysis_result.primary_component.component_type if analysis_result.primary_component else "component"} works and how to use it effectively."""

        if analysis_result.primary_component:
            base_prompt += f"""

🔍 **Key Information Discovered:**
- **Location**: {Path(analysis_result.primary_component.file_path).name}:{analysis_result.primary_component.start_line}
- **Type**: {analysis_result.primary_component.component_type}
- **Language**: {analysis_result.primary_component.language}"""

        if analysis_result.usage_patterns:
            usage_count = len(analysis_result.usage_patterns)
            base_prompt += f"\n- **Usage Examples Found**: {usage_count} different usage patterns"

        base_prompt += """

🤖 **Please help me understand:**

1. **Core Functionality**: Based on the analysis, explain what this component does and why it exists
2. **Interface Deep Dive**: Walk through the component's interface, parameters, and return values in detail
3. **Usage Mastery**: Show me the best ways to use this component, including common patterns and edge cases
4. **Integration Points**: How does this component fit into the larger system? What are its key relationships?
5. **Practical Application**: Give me specific guidance on when and how to use this component in real scenarios"""

        if include_dependencies and analysis_result.dependency_analysis:
            base_prompt += "\n6. **Dependency Analysis**: Explain the component's dependencies and what depends on it"

        if include_usage_examples and analysis_result.usage_patterns:
            base_prompt += "\n7. **Usage Examples**: Show me real examples from the codebase and explain different usage patterns"

        base_prompt += """

🚀 **Use the analysis results above to provide specific, actionable insights that will help me:**
- Understand the component's design and purpose
- Use it correctly and effectively
- Integrate it properly with other parts of the codebase
- Avoid common pitfalls and follow best practices

Please search the codebase to provide additional context and examples beyond what the automated analysis discovered."""

        return base_prompt

    def _create_fallback_component_prompt(
        self,
        component_name: str,
        component_type: str,
        include_dependencies: bool,
        include_usage_examples: bool,
        error_msg: str,
    ) -> list[base.Message]:
        """Create fallback component prompt when enhanced analysis fails."""
        fallback_prompt = f"""I need to understand the component '{component_name}' in this codebase but encountered some analysis limitations: {error_msg}

🎯 **Component Understanding Request:**
I want to deeply understand this {component_type if component_type != "auto" else "component"} and how to work with it effectively."""

        fallback_prompt += """

📋 **Please help me by:**

1. **Component Discovery**: Search for and identify the component in the codebase
2. **Purpose Analysis**: Explain what this component does and why it exists
3. **Interface Exploration**: Show me the component's methods, parameters, and return types
4. **Code Analysis**: Walk through the component's implementation and key logic"""

        if include_dependencies:
            fallback_prompt += "\n5. **Dependency Mapping**: Identify what this component depends on and what depends on it"

        if include_usage_examples:
            fallback_prompt += "\n6. **Usage Examples**: Find and explain real examples of how this component is used"

        fallback_prompt += """

🚀 **Goal**: Provide me with a comprehensive understanding of this component so I can:
- Use it correctly in my own code
- Understand its role in the overall system
- Modify or extend it if needed
- Follow the established patterns and conventions

Please search the codebase systematically to find and analyze this component."""

        return [base.Message(role="user", content=base.TextContent(type="text", text=fallback_prompt))]

    def _build_enhanced_trace_prompt(
        self,
        functionality_description: str,
        trace_result,
        trace_type: str,
        include_config: bool,
        include_data_flow: bool,
        formatted_summary: str,
    ) -> str:
        """Build enhanced trace prompt with comprehensive analysis."""
        base_prompt = f"""I need to trace the complete implementation of '{functionality_description}' through this codebase. I've conducted a comprehensive trace analysis and here are the results:

{formatted_summary}

🎯 **My Tracing Goals:**
I want to understand the complete {trace_type} implementation of this functionality and how all the pieces work together."""

        if trace_result.entry_points:
            entry_count = len(trace_result.entry_points)
            base_prompt += f"""

🔍 **Key Information Discovered:**
- **Entry Points Found**: {entry_count} different ways to access this functionality"""

            if trace_result.api_endpoints:
                api_count = len(trace_result.api_endpoints)
                base_prompt += f"\n- **API Endpoints**: {api_count} REST/API endpoints identified"

            if trace_result.execution_paths:
                path_count = len(trace_result.execution_paths)
                base_prompt += f"\n- **Execution Paths**: {path_count} different execution flows analyzed"

        base_prompt += """

🤖 **Please help me understand:**

1. **Complete Flow Analysis**: Based on the trace results, walk me through the complete execution flow from start to finish
2. **Critical Path**: What's the main execution path and where are the key decision points?
3. **Data Transformations**: How does data flow and transform as it moves through the system?
4. **Integration Points**: How does this functionality integrate with other parts of the system?
5. **Error Handling**: Where are the error handling points and what can go wrong?"""

        if include_config and trace_result.configuration_dependencies:
            config_count = len(trace_result.configuration_dependencies)
            base_prompt += f"\n6. **Configuration Dependencies**: Explain the {config_count} configuration dependencies and their impact"

        if include_data_flow and trace_result.data_flow_analysis:
            flow_count = len(trace_result.data_flow_analysis)
            base_prompt += f"\n7. **Data Flow Details**: Detail the {flow_count} data transformation steps and flow patterns"

        base_prompt += """

🚀 **Use the trace analysis above to provide specific, actionable insights that will help me:**
- Understand how this functionality is implemented end-to-end
- Identify potential modification points and their impact
- Debug issues when they occur
- Extend or modify this functionality safely

Please search the codebase to provide additional context and examples beyond what the automated trace discovered."""

        return base_prompt

    def _create_fallback_trace_prompt(
        self,
        functionality_description: str,
        trace_type: str,
        include_config: bool,
        include_data_flow: bool,
        error_msg: str,
    ) -> list[base.Message]:
        """Create fallback trace prompt when enhanced analysis fails."""
        fallback_prompt = f"""I need to trace the complete implementation of '{functionality_description}' in this codebase but encountered some analysis limitations: {error_msg}

🎯 **Functionality Tracing Request:**
I want to understand the complete {trace_type} implementation of this functionality from entry point to final execution."""

        fallback_prompt += """

📋 **Please help me by:**

1. **Entry Point Discovery**: Find all the ways this functionality can be triggered (API endpoints, user interfaces, etc.)
2. **Call Chain Analysis**: Trace the sequence of function/method calls from start to finish
3. **Core Logic Identification**: Locate where the main business logic is implemented
4. **Data Operations**: Identify how it interacts with databases, files, or external services"""

        if include_config:
            fallback_prompt += (
                "\n5. **Configuration Analysis**: Find configuration settings and environment variables that affect this functionality"
            )

        if include_data_flow:
            fallback_prompt += "\n6. **Data Flow Mapping**: Trace how data is transformed as it flows through the system"

        fallback_prompt += """

🚀 **Goal**: Provide me with a comprehensive understanding of this functionality so I can:
- Understand how it works end-to-end
- Identify potential issues or bottlenecks
- Modify or extend it safely
- Debug problems when they occur

Please search the codebase systematically to map out the complete implementation journey."""

        return [base.Message(role="user", content=base.TextContent(type="text", text=fallback_prompt))]


def register_mcp_prompts(mcp_app: FastMCP) -> MCPPromptsSystem:
    """
    Register MCP Prompts system with a FastMCP application.

    Args:
        mcp_app: FastMCP application instance
    Returns:
        MCPPromptsSystem: Initialized prompts system
    """
    try:
        prompts_system = MCPPromptsSystem(mcp_app)
        logger.info("MCP Prompts system registered successfully")
        return prompts_system
    except Exception as e:
        logger.error(f"Failed to register MCP Prompts system: {e}")
        raise
