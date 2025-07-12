## Relevant Files

- `src/utils/git_context_service.py` - New service for reading Git information from .git directory
- `src/utils/git_context_service.test.py` - Unit tests for GitContextService
- `src/models/file_metadata.py` - Extend FileMetadata model with Git-related fields
- `src/models/file_metadata.test.py` - Unit tests for FileMetadata Git extensions
- `src/services/indexing_service.py` - Integrate Git information collection during indexing
- `src/services/indexing_service.test.py` - Unit tests for indexing with Git information
- `src/services/project_analysis_service.py` - Add Git repository detection capabilities
- `src/services/project_analysis_service.test.py` - Unit tests for Git detection
- `src/tools/search.py` - Add version filtering parameters to search functionality
- `src/tools/search.test.py` - Unit tests for version-filtered search
- `src/tools/project/list_indexed_projects.py` - Display Git version information in project listings
- `src/tools/project/list_indexed_projects.test.py` - Unit tests for project listing with Git info

### Notes

- Unit tests should typically be placed alongside the code files they are testing (e.g., `MyComponent.py` and `MyComponent.test.py` in the same directory).
- Use `uv run pytest [optional/path/to/test/file]` to run tests. Running without a path executes all tests found by the pytest configuration.

## Tasks

- [ ] 1.0 Create Git Context Detection Service
  - [ ] 1.1 Create `src/utils/git_context_service.py` with GitContextService class
  - [ ] 1.2 Implement `.git/HEAD` reading to detect current branch or detached HEAD state
  - [ ] 1.3 Implement commit hash resolution from branch refs (`.git/refs/heads/*`)
  - [ ] 1.4 Implement repository URL extraction from `.git/config`
  - [ ] 1.5 Implement Git worktree detection and handling (`.git` file vs directory)
  - [ ] 1.6 Implement error handling for corrupted or inaccessible Git repositories
  - [ ] 1.7 Add commit timestamp extraction from Git objects
  - [ ] 1.8 Create comprehensive unit tests for all Git states and edge cases

- [ ] 2.0 Extend Data Models for Git Information
  - [ ] 2.1 Add Git-related fields to FileMetadata dataclass (git_commit_hash, git_branch, git_repository_url, git_commit_timestamp)
  - [ ] 2.2 Update FileMetadata.to_dict() method to include Git fields
  - [ ] 2.3 Update FileMetadata.from_dict() method to handle Git fields with backward compatibility
  - [ ] 2.4 Update FileMetadata.__str__() to display Git information when available
  - [ ] 2.5 Add validation for Git field formats (commit hash length, URL format)
  - [ ] 2.6 Create unit tests for FileMetadata Git field serialization/deserialization

- [ ] 3.0 Integrate Git Information into Indexing Pipeline
  - [ ] 3.1 Import and initialize GitContextService in IndexingService
  - [ ] 3.2 Add Git context detection at the start of process_codebase_for_indexing()
  - [ ] 3.3 Pass Git information to FileMetadata creation in file processing
  - [ ] 3.4 Update chunk metadata to include Git information from FileMetadata
  - [ ] 3.5 Ensure Git information is preserved in incremental indexing mode
  - [ ] 3.6 Add logging for Git detection results and any errors
  - [ ] 3.7 Update indexing progress reporting to show Git status
  - [ ] 3.8 Create integration tests for indexing with various Git states

- [ ] 4.0 Enhance Search Functionality with Version Filtering
  - [ ] 4.1 Add git_branch and git_commit optional parameters to search tool
  - [ ] 4.2 Implement metadata filtering in QdrantService for Git parameters
  - [ ] 4.3 Update search query building to include Git filters when provided
  - [ ] 4.4 Ensure Git information is included in search result metadata
  - [ ] 4.5 Add validation for Git parameter formats (branch names, commit hashes)
  - [ ] 4.6 Update search tool documentation with Git filtering examples
  - [ ] 4.7 Create unit tests for version-filtered searches
  - [ ] 4.8 Test search performance with Git filters on large datasets

- [ ] 5.0 Update Project Management Tools for Version Display
  - [ ] 5.1 Modify list_indexed_projects to query Git metadata from collections
  - [ ] 5.2 Format Git information display as "project_name@branch:short_hash"
  - [ ] 5.3 Add special formatting for detached HEAD states ("detached@abc12345")
  - [ ] 5.4 Add clear indication for projects without version control ("No version control")
  - [ ] 5.5 Group or sort projects by Git status for better organization
  - [ ] 5.6 Update tool documentation to explain Git information display
  - [ ] 5.7 Create unit tests for project listing with various Git states
  - [ ] 5.8 Add integration tests for end-to-end Git tracking workflow
