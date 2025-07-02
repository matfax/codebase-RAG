# PRD: Advanced Cross-Project Search Feature

## Introduction/Overview

This feature enhances the existing codebase RAG search functionality to support targeted cross-project searches. Currently, users can only search within the current project (`cross_project=false`) or across all indexed projects (`cross_project=true`). This enhancement allows users to specify exactly which projects to search, providing granular control over search scope and improving search efficiency for multi-project codebases.

The feature includes two main components:
1. **Enhanced Search Tool Interface**: Modify the existing MCP search tool to accept a list of target projects
2. **Advanced Search Prompt**: Create a user-friendly prompt that guides users through cross-project search options

This addresses the need for agents to leverage indexed data as a knowledge base for cross-project integration and implementation tasks.

## Goals

1. **Granular Search Control**: Enable users to specify exactly which projects to search across
2. **Improved Agent Knowledge Integration**: Allow AI agents to use indexed data from multiple selected projects as knowledge base
3. **Enhanced User Experience**: Provide an intuitive prompt interface for advanced search operations
4. **Search Result Clarity**: Clearly identify the source project and path for each search result
5. **Robust Error Handling**: Provide helpful feedback when specified projects don't exist or lack indexed data

## User Stories

1. **As a developer working on microservices**, I want to search for authentication implementations across specific service projects (e.g., user-service, auth-service, api-gateway) without including unrelated projects, so that I can find relevant patterns efficiently.

2. **As an AI agent**, I want to search across a curated set of projects to gather context for cross-project integration tasks, so that I can provide better recommendations based on existing implementations.

3. **As a code architect**, I want to search for database connection patterns across only my backend projects (excluding frontend projects), so that I can ensure consistency in my architecture.

4. **As a developer**, I want to be guided through available project options when performing cross-project searches, so that I can easily discover and select relevant projects.

5. **As a user**, I want clear feedback when I specify non-existent or non-indexed projects, so that I can correct my search parameters.

## Functional Requirements

### Enhanced Search Tool (MCP Interface)

1. The search tool **must** accept a new optional parameter `target_projects: Optional[List[str]]`
2. When `target_projects` is provided, the system **must** search only within collections belonging to the specified projects
3. The system **must** support searching multiple projects simultaneously (e.g., `["project1", "project2", "project3"]`)
4. The system **must** validate that specified project names exist and have indexed data
5. Search results **must** include the source project name and project path for each result
6. The system **must** provide clear error messages when specified projects don't exist or lack indexed data
7. The system **must** maintain existing search functionality (search modes, context inclusion, result limiting)
8. The system **must** handle project name normalization (spaces/hyphens to underscores for collection matching)

### Advanced Search Prompt

9. The system **must** provide an `advance_search` prompt command that guides users through cross-project search options
10. The prompt **must** first ask users whether they want to perform cross-project search
11. If cross-project is selected, the prompt **must** list all available indexed projects with their paths
12. The prompt **must** allow users to select multiple projects or choose to search all projects
13. The prompt **must** display the total number of indexed projects when first invoked
14. The prompt **must** execute the enhanced search tool with appropriate parameters based on user selections
15. The prompt **must** format and present search results with clear project attribution

### Search Results Enhancement

16. Search results **must** display source project information including project name and path
17. Results **must** be sorted by relevance score (mixed display) while maintaining project attribution
18. The system **must** indicate the search scope in result metadata (which projects were searched)

## Non-Goals (Out of Scope)

1. **Search Preference Persistence**: Will not store user search preferences or project combinations
2. **Advanced Project Statistics**: Will not display detailed project statistics (file counts, last indexing time) in project listings
3. **Backward Compatibility**: Will not maintain the existing `cross_project` boolean parameter behavior
4. **Migration Guidance**: Will not provide migration instructions from old to new usage patterns
5. **Result Sorting by Project**: Will not implement project-level grouping or filtering of results
6. **Real-time Project Updates**: Will not automatically refresh project lists during search sessions

## Technical Considerations

1. **MCP Tool Modification**: The existing `mcp__codebase-rag-mcp__search` tool in `src/mcp_tools.py` requires parameter updates
2. **Collection Architecture**: Leverage existing `project_{name}_{type}` collection naming pattern
3. **Search Service Integration**: Modify core search functions in `src/tools/indexing/search_tools.py`
4. **Project Utilities**: Extend `src/tools/project/project_utils.py` for project discovery and validation
5. **Prompt System Integration**: Create new prompt in `src/prompts/` following existing architecture
6. **Error Handling**: Implement comprehensive validation for project existence and data availability
7. **Performance**: Ensure search performance is not significantly impacted by additional validation steps

## Success Metrics

1. **Enhanced Agent Capability**: AI agents can successfully use indexed data from selected projects as knowledge base for cross-project tasks
2. **Search Precision**: Users can limit search scope to relevant projects, reducing noise in results
3. **User Adoption**: The advance_search prompt is actively used for cross-project search scenarios
4. **Error Reduction**: Clear error messages reduce user confusion when specifying invalid projects
5. **Integration Success**: Cross-project search enables better implementation recommendations and code reuse across projects

## Open Questions

1. Should we implement any caching mechanism for project lists to improve performance?
2. How should we handle projects that are partially indexed (some collections exist, others don't)?
3. Should we provide any project similarity or recommendation features based on search patterns?
4. How should we handle very large numbers of indexed projects in the user interface?