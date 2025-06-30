## Relevant Files

- `src/services/file_metadata_service.py` - New service for managing file metadata and change detection
- `src/services/change_detector_service.py` - Service for detecting file changes since last indexing
- `src/services/indexing_service.py` - Enhanced to support incremental indexing mode
- `src/mcp_tools.py` - Updated to include incremental indexing parameter and time estimation
- `manual_indexing.py` - New standalone script for manual indexing operations
- `src/utils/time_estimator.py` - Utility for estimating indexing time and providing user recommendations
- `tests/test_file_metadata_service.py` - Unit tests for file metadata service
- `tests/test_change_detector_service.py` - Unit tests for change detection service
- `tests/test_incremental_indexing.py` - Integration tests for incremental indexing functionality
- `tests/test_manual_indexing.py` - Tests for manual indexing script
- `README.md` - Updated documentation with incremental indexing setup and usage instructions
- `CLAUDE.md` - Updated with new development commands and architecture information

### Notes

- Unit tests should be placed in the `tests/` directory following existing patterns
- Use `.venv/bin/pytest tests/` to run all tests
- The manual indexing script should be executable from the project root
- Qdrant container must be running for all indexing operations

## Tasks

- [x] 1.0 Implement File Metadata and Change Detection Infrastructure
  - [x] 1.1 Create FileMetadata model with fields for file_path, mtime, content_hash, file_size, indexed_at
  - [x] 1.2 Implement FileMetadataService for storing and retrieving file metadata in Qdrant
  - [x] 1.3 Create ChangeDetectorService to compare current file state with stored metadata
  - [x] 1.4 Implement file content hashing (SHA256) for change verification
  - [x] 1.5 Add metadata collection management in QdrantService
  - [x] 1.6 Create utility functions for file system operations (mtime, size, existence checks)

- [x] 2.0 Create Manual Indexing Tool
  - [x] 2.1 Create manual_indexing.py script with command-line argument parsing
  - [x] 2.2 Implement directory and mode parameter validation
  - [x] 2.3 Add progress reporting and status feedback for manual operations
  - [x] 2.4 Integrate with existing indexing services for actual processing
  - [x] 2.5 Add error handling and user-friendly error messages
  - [x] 2.6 Create help documentation and usage examples

- [x] 3.0 Enhance Existing Indexing Service for Incremental Mode
  - [x] 3.1 Add incremental_mode parameter to IndexingService.process_codebase_for_indexing()
  - [x] 3.2 Implement selective file processing based on change detection results
  - [x] 3.3 Add logic to handle file deletions (remove from vector database)
  - [x] 3.4 Implement file move/rename detection and metadata updates
  - [x] 3.5 Add change summary generation (files added, modified, deleted counts)
  - [x] 3.6 Ensure backward compatibility with existing full indexing workflow

- [x] 4.0 Implement Time Estimation and User Recommendations
  - [x] 4.1 Create TimeEstimatorService to calculate expected indexing duration
  - [x] 4.2 Implement repository analysis for file count and size estimation
  - [x] 4.3 Add logic to detect when indexing might exceed 5 minutes
  - [x] 4.4 Create user recommendation system with manual tool suggestions
  - [x] 4.5 Add clear instructions for manual indexing tool usage
  - [x] 4.6 Integrate time estimation into MCP tool responses

- [x] 5.0 Update MCP Tools and API Integration
  - [x] 5.1 Add incremental parameter to index_directory MCP tool
  - [x] 5.2 Integrate time estimation and recommendation logic into MCP workflows
  - [x] 5.3 Add change summary to MCP tool responses
  - [x] 5.4 Update tool documentation and parameter descriptions
  - [x] 5.5 Ensure proper error handling and fallback to full indexing when needed
  - [x] 5.6 Add comprehensive logging for debugging incremental operations