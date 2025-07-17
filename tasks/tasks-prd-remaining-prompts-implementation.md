## Relevant Files

- `src/prompts/find_entry_points.py` - Entry point discovery prompt implementation
- `src/prompts/understand_component.py` - Component analysis prompt implementation
- `src/prompts/trace_functionality.py` - Functionality tracing prompt implementation
- `src/prompts/suggest_next_steps.py` - Next steps recommendation prompt implementation
- `src/prompts/optimize_search.py` - Search optimization prompt implementation
- `src/prompts/base.py` - Base prompt implementation class (may need updates)
- `src/prompts/registry.py` - Prompt registration system (may need updates)
- `src/mcp_prompts.py` - Legacy prompt implementations (source for migration)
- `src/services/component_analysis_service.py` - Component analysis service (may need creation)
- `src/services/functionality_tracing_service.py` - Functionality tracing service (may need creation)
- `src/services/next_steps_service.py` - Next steps recommendation service (may need creation)
- `src/services/search_optimization_service.py` - Search optimization service (may need creation)
- `src/tools/indexing/search_tools.py` - Search tool integration (may need updates)
- `tests/prompts/test_find_entry_points.py` - Unit tests for find_entry_points prompt
- `tests/prompts/test_understand_component.py` - Unit tests for understand_component prompt
- `tests/prompts/test_trace_functionality.py` - Unit tests for trace_functionality prompt
- `tests/prompts/test_suggest_next_steps.py` - Unit tests for suggest_next_steps prompt
- `tests/prompts/test_optimize_search.py` - Unit tests for optimize_search prompt
- `tests/integration/test_prompt_workflows.py` - Integration tests for prompt workflows

### Background Context

This task implements the completion of the MCP Prompts guided workflow system as defined in `prd-remaining-prompts-implementation.md`. The system currently has 2 of 7 prompts implemented (`explore_project` and `advance_search`), and this task will complete the remaining 5 prompts.

**Key Requirements:**
- All prompts must actively call existing MCP tools (search, check_index_status, list_indexed_projects)
- Response time must be ≤15 seconds for 95% of requests
- `trace_functionality` must perform 3-5 levels of tracing through maximum 5 RAG searches
- `suggest_next_steps` must provide specific tool recommendations with keywords
- All prompts must handle unindexed projects by suggesting indexing first
- Seamless workflow transitions between prompts must be established

**Architecture Pattern:**
All prompts follow the enhanced analysis → formatted summary → guided recommendations pattern established by `explore_project`, with:
- Service dependency injection through MCPPromptsSystem
- RAG search integration with existing tools
- Comprehensive error handling with graceful degradation
- Structured output with confidence indicators and source attribution

**Migration Context:**
Existing implementations in `src/mcp_prompts.py` need to be migrated to the new modular system in `src/prompts/` directory, enhanced with RAG integration and proper service architecture.

### Notes

- Use existing patterns from `src/prompts/explore_project.py` and `src/prompts/advance_search.py` as reference implementations
- All prompts use synchronous execution (no async/await in prompt functions)
- Services handle async operations internally
- Follow `BasePromptImplementation` class structure for consistency
- Implement comprehensive fallback mechanisms for service failures

## Tasks

- [ ] 1.0 Migrate Legacy Prompt Implementations to Modular System
  - [ ] 1.1 Analyze existing implementations in `src/mcp_prompts.py` for each of the 5 prompts
  - [ ] 1.2 Create skeleton implementations in `src/prompts/` directory following existing patterns
  - [ ] 1.3 Update prompt registration in `src/prompts/registry.py` to include all 5 new prompts
  - [ ] 1.4 Ensure proper inheritance from `BasePromptImplementation` for all new prompts
  - [ ] 1.5 Implement basic parameter validation and error handling structure

- [ ] 2.0 Implement Core Prompt Functionality with RAG Integration
  - [ ] 2.1 Implement `find_entry_points` prompt with entry point discovery, categorization, and exploration path suggestions
  - [ ] 2.2 Implement `understand_component` prompt with component analysis, dependency mapping, and usage examples
  - [ ] 2.3 Implement `trace_functionality` prompt with 3-5 level execution chain tracing and maximum 5 RAG searches
  - [ ] 2.4 Implement `suggest_next_steps` prompt with context analysis and tool recommendations with keywords
  - [ ] 2.5 Implement `optimize_search` prompt with search pattern analysis and strategy recommendations
  - [ ] 2.6 Integrate active MCP tool calling (search, check_index_status, list_indexed_projects) in all prompts
  - [ ] 2.7 Implement structured output format with confidence indicators and source attribution

- [ ] 3.0 Establish Service Integration and Workflow Transitions
  - [ ] 3.1 Create or enhance `ComponentAnalysisService` for component analysis capabilities
  - [ ] 3.2 Create or enhance `FunctionalityTracingService` for execution chain tracing
  - [ ] 3.3 Create or enhance `NextStepsService` for contextual recommendations
  - [ ] 3.4 Create or enhance `SearchOptimizationService` for search strategy analysis
  - [ ] 3.5 Implement service dependency injection through MCPPromptsSystem
  - [ ] 3.6 Establish seamless workflow transitions between prompts with cross-referencing
  - [ ] 3.7 Implement context persistence for `suggest_next_steps` recommendations

- [ ] 4.0 Implement Error Handling and Performance Optimization
  - [ ] 4.1 Implement comprehensive error handling with graceful degradation for all prompts
  - [ ] 4.2 Add index status validation and guidance for unindexed projects
  - [ ] 4.3 Optimize RAG search strategies to meet 15-second response time requirement
  - [ ] 4.4 Implement proper fallback mechanisms when services are unavailable
  - [ ] 4.5 Add performance monitoring and logging for response time tracking
  - [ ] 4.6 Implement efficient search query optimization for minimal token usage
  - [ ] 4.7 Add memory management for concurrent prompt executions

- [ ] 5.0 Testing and Validation Framework
  - [ ] 5.1 Create unit tests for all 5 prompt implementations
  - [ ] 5.2 Implement integration tests for prompt workflow transitions
  - [ ] 5.3 Create performance tests to validate 15-second response time requirement
  - [ ] 5.4 Implement error handling tests for various failure scenarios
  - [ ] 5.5 Test RAG integration with existing search infrastructure
  - [ ] 5.6 Validate workflow completeness from `explore_project` to `trace_functionality`
  - [ ] 5.7 Test cross-project analysis workflows with `advance_search` + `suggest_next_steps`
