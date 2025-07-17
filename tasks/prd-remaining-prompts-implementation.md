# PRD: Remaining Prompts Implementation - Complete MCP Prompts Guided Workflow System

## Introduction/Overview

The current MCP Prompts system has successfully implemented 2 out of 7 core prompts (`explore_project` and `advance_search`), leaving 5 critical prompts incomplete. This creates a fragmented user experience where AI agents and developers cannot access the full guided workflow system designed to accelerate codebase understanding and development tasks.

This feature will complete the remaining 5 prompts (`find_entry_points`, `understand_component`, `trace_functionality`, `suggest_next_steps`, `optimize_search`) by migrating them from the legacy implementation to the enhanced modular system, integrating RAG search capabilities, and establishing seamless workflows between all prompts.

The goal is to provide a complete, intelligent guided workflow system that leverages indexed codebase knowledge to enable rapid project understanding, effective debugging, and strategic development guidance.

## Goals

1. **Complete Prompt System**: Implement all 5 remaining prompts with full RAG integration and enhanced analysis capabilities
2. **Accelerate Learning Curve**: Enable new developers to understand specific functionality within 15 seconds through `trace_functionality`
3. **Enhance AI Agent Capabilities**: Provide intelligent next-step recommendations through `suggest_next_steps` based on contextual analysis
4. **Establish Seamless Workflows**: Create fluid transitions between prompts for comprehensive project exploration
5. **Leverage External Knowledge**: Enable users to incorporate external project knowledge through cross-project analysis workflows

## User Stories

### Primary User Stories

1. **As a new developer debugging a feature**, I want to use `trace_functionality` to follow the complete execution chain of a specific function, so that I can understand how the feature works and identify potential issues within 15 seconds.

2. **As an AI agent assisting with development**, I want to use `suggest_next_steps` to provide contextual recommendations based on the current analysis state, so that I can guide users through effective development workflows.

3. **As a developer exploring an unfamiliar codebase**, I want to use `find_entry_points` to quickly identify all application entry points, so that I can understand where to start my analysis.

4. **As a developer working on component integration**, I want to use `understand_component` to get detailed analysis of specific modules including dependencies and relationships, so that I can safely modify or extend the component.

5. **As a developer struggling with search results**, I want to use `optimize_search` to get suggestions for better search strategies and keywords, so that I can find relevant code more efficiently.

### Secondary User Stories

6. **As a developer using external project knowledge**, I want to combine `advance_search` with `suggest_next_steps` to understand indexed external projects and incorporate their patterns into my current work.

7. **As a team lead reviewing code**, I want to use the complete prompt workflow to quickly understand feature implementations and provide informed feedback.

## Functional Requirements

### Core Prompt Implementation Requirements

1. **The system must implement `find_entry_points` prompt** that:
   - Discovers all application entry points (main functions, CLI commands, API endpoints, web routes)
   - Categorizes entry points by type (CLI, API, web, scheduled tasks, etc.)
   - Provides usage examples and documentation for each entry point
   - Suggests optimal exploration paths based on user role and project type

2. **The system must implement `understand_component` prompt** that:
   - Analyzes specified components, modules, or classes in depth
   - Maps component dependencies and relationships using RAG search
   - Identifies component interfaces, public methods, and configuration requirements
   - Provides usage examples and integration patterns
   - Highlights potential modification risks and testing requirements

3. **The system must implement `trace_functionality` prompt** that:
   - Follows complete execution chains for specified functions or features
   - Performs 3-5 levels of tracing through maximum 5 RAG searches
   - Identifies database queries, API calls, and external service integrations
   - Reports partial results when complete execution chain cannot be found
   - Provides clear execution flow documentation with call hierarchy

4. **The system must implement `suggest_next_steps` prompt** that:
   - Analyzes current context and user exploration history
   - Provides specific tool recommendations with suggested keywords
   - Offers contextual guidance based on project analysis state
   - Handles insufficient context by requesting user clarification
   - Tailors recommendations to user skill level and current workflow stage

5. **The system must implement `optimize_search` prompt** that:
   - Analyzes user search patterns and result effectiveness
   - Suggests alternative search strategies and keywords
   - Recommends specific search modes (semantic, keyword, hybrid) based on query type
   - Provides cross-project search recommendations when applicable
   - Guides users toward more effective search approaches

### RAG Integration Requirements

6. **All prompts must actively call existing MCP tools** including:
   - `search` tool for semantic and keyword-based code discovery
   - `check_index_status` for validation before analysis
   - `list_indexed_projects` for cross-project awareness
   - Integration with existing services (IndexingService, ProjectAnalysisService, etc.)

7. **The system must handle unindexed projects** by:
   - Checking index status before analysis
   - Providing clear guidance to index the project first
   - Offering graceful degradation with limited static analysis when possible

8. **All prompts must complete analysis within 15 seconds** for:
   - 95% of requests on projects with up to 10,000 indexed chunks
   - Standard analysis depth with option for deeper investigation
   - Efficient RAG search strategies optimized for response time

### Workflow Integration Requirements

9. **The system must establish seamless workflow transitions** between:
   - `explore_project` → `find_entry_points` → `understand_component` → `trace_functionality`
   - `advance_search` → `suggest_next_steps` → [back to core workflow]
   - Each prompt recommending relevant next steps through `suggest_next_steps`

10. **The system must maintain consistent error handling** with:
    - Clear error messages for common failure scenarios
    - Graceful degradation when services are unavailable
    - Fallback mechanisms following existing patterns from `explore_project`

### Output Format Requirements

11. **All prompts must provide structured output** including:
    - Immediate analysis results with RAG-powered insights
    - Strategic guidance for next steps
    - Confidence indicators and source attribution
    - Clear formatting suitable for both AI agents and human developers

12. **The system must support progressive disclosure** allowing:
    - Summary results for quick understanding
    - Detailed analysis for deeper investigation
    - Follow-up prompt recommendations for extended exploration

## Non-Goals (Out of Scope)

1. **Real-time Code Execution**: Will not execute or run code during tracing, only analyze static and indexed content
2. **Automated Code Generation**: Will not generate code implementations, only provide analysis and guidance
3. **Version Control Integration**: Will not integrate with Git history or branch analysis
4. **Performance Profiling**: Will not provide runtime performance analysis or optimization
5. **Database Schema Analysis**: Will not perform deep database structure analysis beyond indexed code references
6. **Security Vulnerability Scanning**: Will not perform security audits or vulnerability assessments
7. **Backward Compatibility**: May break existing parameter interfaces in legacy mcp_prompts.py implementation

## Design Considerations

### User Experience Design
- **Progressive Workflow**: Each prompt builds upon previous analysis, creating natural exploration paths
- **Context Awareness**: Prompts consider user skill level, project type, and current analysis state
- **Clear Error Recovery**: Comprehensive error messages with actionable recovery steps
- **Consistent Interface**: All prompts follow the same input/output patterns established by `explore_project`

### Integration Architecture
- **Service Reuse**: Leverage existing enhanced services (ComponentAnalysisService, FunctionalityTracingService)
- **MCP Tool Integration**: Active use of existing MCP tools rather than passive recommendations
- **Modular Design**: Each prompt is independently useful but optimized for workflow integration
- **Fallback Mechanisms**: Graceful degradation when enhanced services are unavailable

## Technical Considerations

### Implementation Dependencies
- **Migration from Legacy System**: Complete migration from existing implementations in `src/mcp_prompts.py`
- **Service Integration**: Full integration with IndexingService, ProjectAnalysisService, EmbeddingService
- **MCP Framework**: Built on FastMCP with proper error handling and message formatting
- **RAG Search Integration**: Active use of existing search infrastructure with optimized query strategies

### Performance Constraints
- **Response Time**: 15-second maximum response time for 95% of requests
- **Memory Efficiency**: Operate within existing MCP server memory constraints
- **Concurrent Execution**: Support multiple simultaneous prompt executions
- **Search Optimization**: Efficient RAG search patterns to minimize token usage

### Architecture Requirements
- **Modular Structure**: Follow existing patterns from `src/prompts/` directory
- **Base Class Inheritance**: Use `BasePromptImplementation` for consistent behavior
- **Service Dependency Injection**: Integrate with MCPPromptsSystem for service access
- **Error Handling**: Comprehensive fallback mechanisms with user-friendly messages

## Success Metrics

### Primary Success Metrics
1. **Learning Acceleration**: New developers can understand specific functionality within 15 seconds using `trace_functionality`
2. **Workflow Completion**: 90% of users successfully complete guided workflows from `explore_project` to `trace_functionality`
3. **External Knowledge Integration**: Users can effectively incorporate external project knowledge through `advance_search` + `suggest_next_steps` workflows
4. **AI Agent Effectiveness**: AI agents provide relevant next-step recommendations in 85% of `suggest_next_steps` interactions

### Technical Performance Metrics
5. **Response Time**: 95% of prompt executions complete within 15 seconds
6. **System Stability**: 99.5% uptime with proper error handling and graceful degradation
7. **Search Efficiency**: Average of 3 RAG searches per `trace_functionality` execution
8. **Error Rate**: <5% of prompt executions result in unrecoverable errors

### User Experience Metrics
9. **Prompt Discovery**: 80% of users discover and use at least 3 different prompts within first session
10. **Workflow Satisfaction**: Positive feedback on prompt recommendations and next-step guidance
11. **Knowledge Transfer**: Successful application of external project patterns in development tasks

## Open Questions

1. **Caching Strategy**: Should we implement caching for frequently traced functions or analyzed components to improve response times?

2. **Context Persistence**: How long should we maintain user exploration history for `suggest_next_steps` personalization?

3. **Cross-Project Limitations**: Should we implement any restrictions on cross-project analysis to prevent information overload?

4. **Incremental Analysis**: Should `trace_functionality` support incremental deepening (3 levels initially, then 5 levels on user request)?

5. **Integration Testing**: What specific test scenarios should we implement to validate workflow transitions between prompts?

6. **Migration Strategy**: Should we maintain temporary backward compatibility with legacy prompt interfaces during migration?

7. **Performance Monitoring**: What metrics should we track to ensure the 15-second response time requirement is consistently met?

8. **Feedback Loop**: How should we collect and incorporate user feedback to improve prompt effectiveness over time?

---

**Document Version**: 1.0
**Created**: 2025-07-15
**Target Audience**: Junior to Senior Developers
**Implementation Complexity**: Medium-High
**Estimated Implementation Time**: 3-4 weeks
