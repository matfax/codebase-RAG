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

import os
import logging
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base

from services.indexing_service import IndexingService
from services.project_analysis_service import ProjectAnalysisService
from services.embedding_service import EmbeddingService
from utils.performance_monitor import MemoryMonitor
from models.prompt_context import PromptContext, PromptResult


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
            focus_area: Optional[str] = None,
            detail_level: str = "overview"
        ) -> List[base.Message]:
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
                # Get project analysis
                analysis = self.analysis_service.analyze_repository(directory)
                
                if "error" in analysis:
                    return [base.Message(
                        role="user",
                        content=[base.TextContent(
                            text=f"I need help exploring this project at {directory}, but encountered an error: {analysis['error']}. Can you help me understand what might be wrong and suggest alternative approaches?"
                        )]
                    )]
                
                # Build exploration context
                project_stats = {
                    "total_files": analysis.get("total_files", 0),
                    "relevant_files": analysis.get("relevant_files", 0), 
                    "languages": analysis.get("language_breakdown", {}),
                    "size_mb": analysis.get("size_analysis", {}).get("total_size_mb", 0),
                    "complexity": analysis.get("indexing_complexity", {}).get("level", "unknown")
                }
                
                # Create guided exploration prompt
                exploration_prompt = self._build_exploration_prompt(
                    directory, project_stats, focus_area, detail_level
                )
                
                return [base.Message(
                    role="user", 
                    content=[base.TextContent(text=exploration_prompt)]
                )]
                
            except Exception as e:
                self.logger.error(f"Error in explore_project prompt: {e}")
                return [base.Message(
                    role="user",
                    content=[base.TextContent(
                        text=f"I want to explore the project structure at {directory} but encountered an error. Can you help me understand the codebase architecture and suggest the best way to get started exploring this project?"
                    )]
                )]
    
    def _register_understand_component(self):
        """Register the understand_component prompt."""
        @self.mcp_app.prompt()
        def understand_component(
            component_name: str,
            component_type: str = "auto",
            include_dependencies: bool = True,
            include_usage_examples: bool = True
        ) -> List[base.Message]:
            """
            Get in-depth understanding of a specific component, module, or function.
            
            This prompt provides detailed analysis of component responsibilities, interfaces,
            dependencies, and usage patterns. Ideal for understanding complex components
            before making modifications or integrating with them.
            
            Args:
                component_name: Name of component to analyze (class, function, module, file)
                component_type: Type hint - "class", "function", "module", "file", or "auto" 
                include_dependencies: Whether to analyze dependencies and relationships
                include_usage_examples: Whether to provide usage examples and patterns
            """
            try:
                # Search for the component in indexed codebase
                search_query = f"{component_name} {component_type}" if component_type != "auto" else component_name
                
                # Build component understanding prompt
                understanding_prompt = self._build_component_understanding_prompt(
                    component_name, component_type, include_dependencies, include_usage_examples, search_query
                )
                
                return [base.Message(
                    role="user",
                    content=[base.TextContent(text=understanding_prompt)]
                )]
                
            except Exception as e:
                self.logger.error(f"Error in understand_component prompt: {e}")
                return [base.Message(
                    role="user",
                    content=[base.TextContent(
                        text=f"I need to understand the component '{component_name}' in this codebase. Can you help me analyze its purpose, interfaces, dependencies, and provide usage examples?"
                    )]
                )]
    
    def _register_trace_functionality(self):
        """Register the trace_functionality prompt."""
        @self.mcp_app.prompt()
        def trace_functionality(
            functionality_description: str,
            trace_type: str = "full_flow",
            include_config: bool = True,
            include_data_flow: bool = True
        ) -> List[base.Message]:
            """
            Trace the complete implementation path of specific functionality.
            
            This prompt traces functionality from API endpoints to database operations,
            identifying the complete call chain, configuration dependencies, and data flow.
            Essential for understanding complex feature implementations or debugging issues.
            
            Args:
                functionality_description: Description of functionality to trace (e.g., "user authentication", "file upload")
                trace_type: Type of trace - "full_flow", "api_to_db", "user_journey", or "data_pipeline"  
                include_config: Whether to include configuration and environment dependencies
                include_data_flow: Whether to trace data transformation and flow patterns
            """
            try:
                # Build functionality tracing prompt
                tracing_prompt = self._build_functionality_tracing_prompt(
                    functionality_description, trace_type, include_config, include_data_flow
                )
                
                return [base.Message(
                    role="user",
                    content=[base.TextContent(text=tracing_prompt)]
                )]
                
            except Exception as e:
                self.logger.error(f"Error in trace_functionality prompt: {e}")
                return [base.Message(
                    role="user",
                    content=[base.TextContent(
                        text=f"I need to trace how '{functionality_description}' is implemented in this codebase. Can you help me understand the complete flow from entry point to final execution?"
                    )]
                )]
    
    def _register_find_entry_points(self):
        """Register the find_entry_points prompt."""
        @self.mcp_app.prompt()
        def find_entry_points(
            entry_type: str = "all",
            learning_path: bool = True,
            include_examples: bool = True
        ) -> List[base.Message]:
            """
            Identify and explain all main entry points into the application.
            
            This prompt discovers entry points like main functions, API routes, CLI commands,
            and other application starting points, providing a structured learning path
            for new developers to understand how the application is used and invoked.
            
            Args:
                entry_type: Type of entry points - "all", "api", "cli", "main_functions", or "scripts"
                learning_path: Whether to suggest an optimal learning/reading order
                include_examples: Whether to provide usage examples for each entry point
            """
            try:
                # Build entry points discovery prompt
                entry_points_prompt = self._build_entry_points_prompt(
                    entry_type, learning_path, include_examples
                )
                
                return [base.Message(
                    role="user",
                    content=[base.TextContent(text=entry_points_prompt)]
                )]
                
            except Exception as e:
                self.logger.error(f"Error in find_entry_points prompt: {e}")
                return [base.Message(
                    role="user",
                    content=[base.TextContent(
                        text=f"I need to find all the entry points for this application. Can you help me identify main functions, API routes, CLI commands, and other ways to start or interact with this codebase?"
                    )]
                )]
    
    def _register_suggest_next_steps(self):
        """Register the suggest_next_steps prompt."""  
        @self.mcp_app.prompt()
        def suggest_next_steps(
            current_context: str,
            user_role: str = "developer",
            task_type: str = "exploration",
            difficulty_level: str = "intermediate"
        ) -> List[base.Message]:
            """
            Get intelligent recommendations for next actions based on current context.
            
            This prompt analyzes your current exploration state and provides personalized
            recommendations for next steps, considering your role, task objectives, and
            skill level. Helps maintain momentum and direction in complex codebase work.
            
            Args:
                current_context: Description of what you're currently working on or have discovered
                user_role: Your role - "developer", "architect", "reviewer", "newcomer", or "debugger"
                task_type: Type of task - "exploration", "development", "refactoring", "debugging", or "review"
                difficulty_level: Skill level - "beginner", "intermediate", or "advanced"
            """
            try:
                # Build next steps recommendation prompt
                next_steps_prompt = self._build_next_steps_prompt(
                    current_context, user_role, task_type, difficulty_level
                )
                
                return [base.Message(
                    role="user",
                    content=[base.TextContent(text=next_steps_prompt)]
                )]
                
            except Exception as e:
                self.logger.error(f"Error in suggest_next_steps prompt: {e}")
                return [base.Message(
                    role="user",
                    content=[base.TextContent(
                        text=f"Based on my current context: '{current_context}', can you suggest the best next steps for a {user_role} working on {task_type} tasks?"
                    )]
                )]
    
    def _register_optimize_search(self):
        """Register the optimize_search prompt."""
        @self.mcp_app.prompt()
        def optimize_search(
            previous_searches: List[str],
            search_goal: str,
            refine_strategy: bool = True,
            suggest_alternatives: bool = True  
        ) -> List[base.Message]:
            """
            Optimize search strategies and suggest better approaches for finding information.
            
            This prompt analyzes your search patterns and results to recommend more effective
            search terms, strategies, and alternative approaches for finding the information
            you need in the codebase.
            
            Args:
                previous_searches: List of previous search queries you've tried
                search_goal: What you're ultimately trying to find or understand
                refine_strategy: Whether to suggest refined search terms and strategies
                suggest_alternatives: Whether to suggest alternative investigation approaches
            """
            try:
                # Build search optimization prompt
                search_optimization_prompt = self._build_search_optimization_prompt(
                    previous_searches, search_goal, refine_strategy, suggest_alternatives
                )
                
                return [base.Message(
                    role="user",
                    content=[base.TextContent(text=search_optimization_prompt)]
                )]
                
            except Exception as e:
                self.logger.error(f"Error in optimize_search prompt: {e}")
                return [base.Message(
                    role="user",
                    content=[base.TextContent(
                        text=f"I've been searching for '{search_goal}' with these queries: {previous_searches}. Can you help me optimize my search strategy and suggest better approaches?"
                    )]
                )]
    
    def _build_exploration_prompt(self, directory: str, stats: Dict, focus_area: Optional[str], detail_level: str) -> str:
        """Build the project exploration prompt text."""
        base_prompt = f"""I need to explore and understand the codebase at '{directory}'. Here's what I know about the project:

📊 **Project Statistics:**
- Total files: {stats['total_files']:,}
- Relevant code files: {stats['relevant_files']:,}  
- Size: {stats['size_mb']} MB
- Complexity: {stats['complexity']}
- Languages: {', '.join(stats['languages'].keys()) if stats['languages'] else 'Not detected'}

🎯 **Exploration Goal:** 
I want a {detail_level} exploration of this project's architecture and structure."""

        if focus_area:
            base_prompt += f"\n\n🔍 **Focus Area:** Please pay special attention to '{focus_area}' related components and functionality."

        base_prompt += f"""

📋 **Please help me with:**
1. **Architecture Overview:** What's the overall architecture pattern and project structure?
2. **Key Components:** What are the most important modules, classes, and functions?
3. **Entry Points:** How does the application start and what are the main interaction points?
4. **Dependencies:** What are the key dependencies and how do components interact?
5. **Navigation Strategy:** What's the best order to explore and understand this codebase?

🚀 **Next Steps:** After this overview, suggest the most valuable next exploration steps based on what you discover.

Please search the codebase systematically and provide actionable insights that will help me quickly become productive with this project."""

        return base_prompt
    
    def _build_component_understanding_prompt(self, component_name: str, component_type: str, 
                                           include_deps: bool, include_examples: bool, search_query: str) -> str:
        """Build the component understanding prompt text."""
        base_prompt = f"""I need to deeply understand the component '{component_name}' in this codebase.

🎯 **Component to Analyze:** {component_name}
📝 **Component Type:** {component_type}

🔍 **Analysis Requirements:**
1. **Purpose & Responsibility:** What does this component do and why does it exist?
2. **Interface & API:** What are its public methods, parameters, and return types?
3. **Implementation Details:** How does it work internally?"""

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
    
    def _build_functionality_tracing_prompt(self, functionality: str, trace_type: str, 
                                          include_config: bool, include_data_flow: bool) -> str:
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

        base_prompt += f"""

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
    
    def _build_search_optimization_prompt(self, previous_searches: List[str], goal: str, 
                                        refine_strategy: bool, suggest_alternatives: bool) -> str:
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