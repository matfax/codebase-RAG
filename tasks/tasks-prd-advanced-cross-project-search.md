## Relevant Files

- `src/mcp_tools.py` - Contains the main MCP tool definitions including the search tool that needs enhancement
- `src/tools/indexing/search_tools.py` - Core search implementation that handles collection filtering and search execution
- `src/tools/project/project_utils.py` - Project discovery and validation utilities
- `src/services/qdrant_service.py` - Vector database service for collection management
- `src/prompts/advanced_search/` - New directory for advanced search prompt implementation
- `src/prompts/advanced_search/__init__.py` - Module initialization for advanced search prompts
- `src/prompts/advanced_search/advance_search.py` - Main advanced search prompt implementation
- `src/prompts/registry.py` - Prompt registration system to register the new prompt
- `tests/test_advanced_search_tool.py` - Unit tests for enhanced search tool functionality
- `tests/test_advance_search_prompt.py` - Unit tests for advanced search prompt

### Notes

- Tests should be placed in the `tests/` directory following the existing test structure
- Use `pytest tests/` to run all tests or `pytest tests/test_specific.py` for individual test files
- The prompt system follows the existing architecture in `src/prompts/` with base classes and registration

## Tasks

- [x] 1.0 Enhance MCP Search Tool Interface
  - [x] 1.1 Add `target_projects` parameter to search tool schema in `src/mcp_tools.py`
  - [x] 1.2 Update search tool parameter validation and documentation
  - [x] 1.3 Modify search tool handler to process target_projects parameter
  - [x] 1.4 Add project name normalization for collection matching
  - [x] 1.5 Implement error handling for invalid project specifications

- [x] 2.0 Implement Target Projects Logic in Search Services
  - [x] 2.1 Update `search_sync()` function in `src/tools/indexing/search_tools.py` to accept target_projects
  - [x] 2.2 Create `get_target_project_collections()` helper function for collection filtering
  - [x] 2.3 Modify collection selection logic to support project-specific filtering
  - [x] 2.4 Update search result metadata to include project information
  - [x] 2.5 Add project path information to search results

- [x] 3.0 Create Project Discovery and Validation Utilities
  - [x] 3.1 Add `list_indexed_projects()` function to `src/tools/project/project_utils.py`
  - [x] 3.2 Implement `validate_project_exists()` function for project existence checking
  - [x] 3.3 Create `get_project_collections()` function to map projects to collections
  - [x] 3.4 Add `normalize_project_name()` utility for consistent naming
  - [x] 3.5 Implement project metadata extraction (name, path, collection info)

- [x] 4.0 Implement Advanced Search Prompt
  - [x] 4.1 Create `src/prompts/advanced_search/` directory structure
  - [x] 4.2 Implement base advanced search prompt class
  - [x] 4.3 Add cross-project confirmation and project listing logic
  - [x] 4.4 Implement project selection interface and validation
  - [x] 4.5 Add search execution and result formatting
  - [x] 4.6 Register advance_search prompt in prompt registry

- [ ] 5.0 Add Comprehensive Testing and Documentation
  - [ ] 5.1 Create unit tests for enhanced search tool functionality
  - [ ] 5.2 Add tests for project discovery and validation utilities
  - [ ] 5.3 Implement integration tests for advanced search prompt
  - [ ] 5.4 Update MCP tool documentation with new parameters
  - [ ] 5.5 Add usage examples and error handling documentation