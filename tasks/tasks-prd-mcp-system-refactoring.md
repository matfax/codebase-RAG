## Relevant Files

- `src/prompts/__init__.py` - Main prompts module export interface
- `src/prompts/base/__init__.py` - Base prompt utilities module interface
- `src/prompts/base/prompt_base.py` - Base prompt classes and shared utilities
- `src/prompts/base/prompt_builder.py` - Common prompt building functions
- `src/prompts/exploration/__init__.py` - Exploration prompts module interface
- `src/prompts/exploration/explore_project.py` - Project exploration prompt implementation
- `src/prompts/exploration/understand_component.py` - Component analysis prompt implementation
- `src/prompts/exploration/trace_functionality.py` - Functionality tracing prompt implementation
- `src/prompts/exploration/find_entry_points.py` - Entry points discovery prompt implementation
- `src/prompts/recommendation/__init__.py` - Recommendation prompts module interface
- `src/prompts/recommendation/suggest_next_steps.py` - Next steps suggestion prompt implementation
- `src/prompts/recommendation/optimize_search.py` - Search optimization prompt implementation
- `src/prompts/registry.py` - Main MCPPromptsSystem class and prompt registration logic
- `src/tools/__init__.py` - Main tools module export interface
- `src/tools/core/__init__.py` - Core tools module interface
- `src/tools/core/health.py` - Health check and system status tools
- `src/tools/core/memory_utils.py` - Memory management utilities and tools
- `src/tools/indexing/__init__.py` - Indexing tools module interface
- `src/tools/indexing/index_tools.py` - Directory indexing and analysis tools
- `src/tools/indexing/search_tools.py` - Search and query tools
- `src/tools/indexing/analysis_tools.py` - Repository analysis tools
- `src/tools/project/__init__.py` - Project tools module interface
- `src/tools/project/project_tools.py` - Project management and configuration tools
- `src/tools/project/file_tools.py` - File operations and metadata tools
- `src/tools/database/__init__.py` - Database tools module interface
- `src/tools/database/qdrant_tools.py` - Qdrant client and connection tools
- `src/tools/database/collection_tools.py` - Collection management and cleanup tools
- `src/tools/registry.py` - Main tool registration logic and MCP app configuration
- `src/main.py` - Updated main module imports
- `src/mcp_prompts.test.py` - Updated test imports for prompts
- Additional test files as needed

### Notes

- Use `.venv/bin/pytest tests/` to run all tests
- Maintain backward compatibility during refactoring
- All existing MCP tool and prompt registrations must continue to work unchanged
- Ensure no circular import dependencies between modules
- Each module should have clear, single responsibility
- Import paths must be updated systematically across all affected files

## Tasks

- [x] 1.0 Design and Create Base Module Structure
  - [x] 1.1 Create src/prompts/ directory structure with __init__.py files
  - [x] 1.2 Create src/tools/ directory structure with __init__.py files
  - [x] 1.3 Analyze current mcp_prompts.py to identify shared utilities and dependencies
  - [x] 1.4 Analyze current mcp_tools.py to identify shared utilities and dependencies
  - [x] 1.5 Design import hierarchy to prevent circular dependencies
  - [x] 1.6 Create base prompt utilities and shared classes
  - [x] 1.7 Create shared tool utilities and error handling

- [x] 2.0 Refactor MCP Prompts System into Modular Architecture
  - [x] 2.1 Extract and migrate explore_project prompt to src/prompts/exploration/
  - [x] 2.2 Extract and migrate understand_component prompt to src/prompts/exploration/
  - [x] 2.3 Extract and migrate trace_functionality prompt to src/prompts/exploration/
  - [x] 2.4 Extract and migrate find_entry_points prompt to src/prompts/exploration/
  - [x] 2.5 Extract and migrate suggest_next_steps prompt to src/prompts/recommendation/
  - [x] 2.6 Extract and migrate optimize_search prompt to src/prompts/recommendation/
  - [x] 2.7 Create src/prompts/registry.py with MCPPromptsSystem class
  - [x] 2.8 Implement proper module imports and exports in __init__.py files
  - [x] 2.9 Update prompt builder utilities and shared helper functions

- [x] 3.0 Refactor MCP Tools System into Modular Architecture
  - [x] 3.1 Extract health_check tool to src/tools/core/health.py
  - [x] 3.2 Extract memory management utilities to src/tools/core/memory_utils.py
  - [x] 3.3 Extract index_directory and related tools to src/tools/indexing/index_tools.py
  - [x] 3.4 Extract search tool and related functions to src/tools/indexing/search_tools.py
  - [x] 3.5 Extract analyze_repository tools to src/tools/indexing/analysis_tools.py
  - [x] 3.6 Extract project management tools to src/tools/project/project_tools.py
  - [x] 3.7 Extract file operation tools to src/tools/project/file_tools.py
  - [x] 3.8 Extract Qdrant client tools to src/tools/database/qdrant_tools.py
  - [x] 3.9 Extract collection management tools to src/tools/database/collection_tools.py
  - [x] 3.10 Create src/tools/registry.py with main tool registration logic
  - [x] 3.11 Implement proper module imports and exports in __init__.py files

- [x] 4.0 Update Main Application and Integration Points
  - [x] 4.1 Update src/main.py imports to use new modular structure
  - [x] 4.2 Update src/run_mcp.py if needed for new imports
  - [x] 4.3 Create backward compatibility layer for existing imports
  - [x] 4.4 Update any other files that import from mcp_prompts or mcp_tools
  - [x] 4.5 Verify MCP app registration works with new structure
  - [x] 4.6 Test FastMCP prompt and tool discovery with new modules

- [x] 5.0 Validate and Test Refactored System
  - [x] 5.1 Update src/mcp_prompts.test.py imports for new module structure
  - [x] 5.2 Run existing test suite to ensure no regressions
  - [x] 5.3 Create additional integration tests for module loading
  - [x] 5.4 Test MCP server startup and tool/prompt registration
  - [x] 5.5 Validate all prompts work correctly with new structure
  - [x] 5.6 Validate all tools work correctly with new structure
  - [x] 5.7 Performance test to ensure no degradation
  - [x] 5.8 Clean up original mcp_prompts.py and mcp_tools.py files
  - [x] 5.9 Update documentation and CLAUDE.md if needed
