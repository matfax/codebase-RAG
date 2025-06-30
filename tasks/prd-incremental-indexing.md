# Product Requirements Document: Incremental Indexing

## 1. Introduction/Overview

This document outlines the requirements for implementing incremental indexing functionality in the Codebase RAG MCP Server. The feature addresses the performance bottleneck of indexing large codebases by only processing changed files since the last indexing operation, rather than reprocessing the entire codebase.

Currently, users with large codebases (1000+ files) face lengthy indexing times that can exceed 5 minutes, creating a poor user experience. This feature will introduce intelligent change detection and selective reprocessing to dramatically reduce indexing time for typical development workflows.

## 2. Goals

- **G-1:** Reduce indexing time for large codebases by 80%+ for typical development scenarios (5-50 changed files)
- **G-2:** Provide a standalone manual indexing tool for heavy indexing operations to avoid blocking conversational workflows
- **G-3:** Automatically detect and process only changed files since the last indexing operation
- **G-4:** Maintain backward compatibility with existing indexed collections
- **G-5:** Improve user experience by providing intelligent recommendations for indexing approach based on estimated processing time

## 3. User Stories

- **U-1:** As a developer working on a large codebase, I want to update my index after changing 5 files in a 1000-file codebase without waiting for full reindexing, so that I can quickly get updated code analysis.

- **U-2:** As an AI assistant, I want to know which files changed since the last indexing so I can provide accurate code analysis and focus on recently modified areas.

- **U-3:** As a developer with a large repository, I want to be warned when indexing might take over 5 minutes and be offered an alternative manual indexing approach, so I can choose the best workflow for my situation.

- **U-4:** As a developer, I want the system to automatically detect when files have been moved or renamed and update the index accordingly, so I don't lose context when refactoring code structure.

- **U-5:** As a user, I want clear setup instructions for the required Qdrant container so I can quickly get the system running locally.

## 4. Functional Requirements

### Core Incremental Indexing
- **F-1:** The system must detect file changes since the last indexing operation by comparing file modification timestamps and content hashes
- **F-2:** The system must only reprocess files that have been added, modified, or renamed since the last indexing
- **F-3:** The system must remove stale entries from the vector database when files are deleted
- **F-4:** The system must handle file moves and renames by updating metadata while preserving embeddings when content is unchanged
- **F-5:** The system must store file metadata (modification time, content hash, file size) to enable accurate change detection

### Manual Indexing Tool
- **F-6:** The system must provide a standalone script `manual_indexing.py` that can be executed independently
- **F-7:** The manual indexing tool must accept command-line parameters:
  - `-d, --directory`: Target directory path (required)
  - `-m, --mode`: Indexing mode with options `clear_existing` or `incremental` (required)
- **F-8:** The manual indexing tool must provide progress feedback and completion status
- **F-9:** The manual indexing tool must work with the same Qdrant configuration as the MCP server

### Intelligent Recommendations
- **F-10:** The system must estimate indexing time based on file count and repository size
- **F-11:** When estimated indexing time exceeds 5 minutes, the system must recommend using the manual indexing tool
- **F-12:** The system must provide clear instructions for using the manual indexing tool when recommended

### MCP Integration
- **F-13:** The existing `index_directory` MCP tool must support an `incremental` parameter to enable incremental indexing mode
- **F-14:** The system must provide a summary of changes processed during incremental indexing (files added, modified, deleted)
- **F-15:** The system must maintain the existing `clear_existing` functionality for full reindexing when needed

## 5. Non-Goals (Out of Scope)

- **NG-1:** Handling cases where embedding models change between indexing operations
- **NG-2:** Complex edge case handling (e.g., symbolic links, special file types)
- **NG-3:** Rollback capabilities for failed incremental updates
- **NG-4:** Cross-platform compatibility beyond current system support
- **NG-5:** Performance targets beyond general improvement (specific metrics will be evaluated post-implementation)

## 6. Design Considerations

### File Change Detection
- Use file modification timestamps (`mtime`) as the primary change indicator
- Implement content hashing (SHA256) as a secondary verification method
- Store metadata in a dedicated Qdrant collection for efficient querying

### Backward Compatibility
- Existing collections without metadata will fall back to full reindexing
- Gradual migration strategy for existing users
- Maintain existing MCP tool interfaces

### User Experience
- Clear progress indication during incremental updates
- Informative error messages with fallback suggestions
- Detailed change summaries for transparency

## 7. Technical Considerations

### Qdrant Integration
- Must work with the current Qdrant setup and configuration
- Utilize existing collection structure and naming conventions
- Implement efficient metadata queries for change detection

### Performance Optimization
- Batch processing for multiple file changes
- Minimize vector database operations
- Optimize memory usage during incremental updates

### Error Handling
- Graceful fallback to full reindexing on metadata corruption
- Comprehensive logging for debugging incremental operations
- Clear error messages for common failure scenarios

## 8. Success Metrics

- **M-1:** Indexing time reduction: Measure average time improvement for typical development scenarios (5-50 changed files in 1000+ file codebases)
- **M-2:** User adoption: Track usage of incremental indexing vs. full reindexing
- **M-3:** Change detection accuracy: Verify that no file changes are missed and no unchanged files are unnecessarily reprocessed
- **M-4:** Error rate: Monitor fallback to full reindexing due to incremental indexing failures
- **M-5:** User satisfaction: Collect feedback on improved workflow experience

## 9. Setup Requirements

### Qdrant Container Setup
Users must have a local Qdrant instance running. The system will provide clear documentation:

```bash
# Start Qdrant container
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```

### Environment Configuration
- Existing `.env` file configuration will be maintained
- No additional setup required beyond current system requirements

## 10. Open Questions

- **Q-1:** Should the system provide options for different change detection strategies (timestamp-only vs. content-hash verification)?
- **Q-2:** How should the system handle concurrent indexing operations (e.g., manual tool running while MCP server is active)?
- **Q-3:** Should there be a configurable threshold for the 5-minute warning, or should it be dynamic based on system performance?
- **Q-4:** How should the system handle partial failures during incremental updates (e.g., some files process successfully, others fail)?