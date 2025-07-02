# PRD: Large Codebase Indexing Performance Optimization

## Introduction/Overview

The current Codebase RAG MCP Server experiences timeout issues when indexing large codebases, causing incomplete indexing with only partial data (e.g., single points) appearing in Qdrant. This feature will optimize the indexing performance to handle large repositories efficiently while maintaining all existing functionality.

The goal is to transform the current sequential, synchronous indexing process into a high-performance, asynchronous system that can handle large codebases without timeouts while providing real-time feedback to users.

## Goals

1. Eliminate timeout issues for large codebase indexing operations
2. Implement asynchronous background indexing that doesn't block search operations
3. Provide real-time progress tracking and logging for indexing operations
4. Maintain backward compatibility with existing MCP tools and search functionality
5. Add intelligent codebase analysis tools to help users assess indexing complexity
6. Implement incremental indexing to enable resumable operations

## User Stories

1. **As a developer**, I want to index large codebases (10,000+ files) without encountering timeouts, so that I can use RAG search across my entire project.

2. **As a developer**, I want indexing to run in the background while I continue using search functionality, so that my workflow isn't interrupted by long indexing operations.

3. **As a developer**, I want to see indexing progress and estimated completion time, so that I can plan my work accordingly.

4. **As a developer**, I want to quickly assess if a repository is large before indexing, so that I can decide whether to index the full repository or specific subdirectories.

5. **As a developer**, I want failed indexing operations to be resumable, so that I don't lose progress when encountering errors or interruptions.

6. **As a developer**, I want memory usage warnings in logs, so that I can understand system resource usage during indexing.

## Functional Requirements

1. **Parallel Processing**: The system must process multiple files concurrently using configurable thread pools for file I/O operations.

2. **Batch Embedding Generation**: The system must group multiple file contents into batches (configurable 10-50 files) for single Ollama API calls to reduce network latency.

3. **Asynchronous Indexing**: The system must support background indexing operations that don't block search functionality.

4. **Progress Tracking**: The system must provide real-time progress updates showing files processed, remaining files, and estimated completion time.

5. **Streaming Database Operations**: The system must insert embeddings into Qdrant in configurable batches (500-1000 points) rather than accumulating all in memory.

6. **Repository Analysis Tools**: The system must provide commands to analyze repository size, file count, and complexity before indexing.

7. **Incremental Indexing**: The system must track file modification times and only reprocess changed files during subsequent indexing operations.

8. **Memory Management**: The system must implement memory cleanup between batches and provide memory usage logging.

9. **Status Reporting**: Search responses must indicate when indexing operations are in progress for the queried collections.

10. **Enhanced Logging**: The system must provide detailed logging for each indexing stage (file reading, embedding generation, database insertion) to help identify bottlenecks.

11. **Graceful Error Handling**: The system must handle partial failures and continue processing remaining files while logging specific errors.

12. **Configuration Options**: The system must allow users to configure batch sizes, concurrency limits, and memory thresholds through environment variables.

## Non-Goals (Out of Scope)

1. Changing the core Ollama + Qdrant architecture
2. Breaking existing MCP tool interfaces or search functionality
3. Implementing real-time API monitoring endpoints (simple logging is sufficient)
4. Supporting different file types with specialized processing (current approach maintained)
5. Backward compatibility with existing configuration files (new installations only)
6. Integration with external embedding services beyond Ollama in this phase

## Design Considerations

1. **Asynchronous Architecture**: Implement a task queue system for background indexing operations with status tracking.

2. **Memory-Efficient Processing**: Use streaming approaches where files are processed in configurable batches (e.g., 100 files at a time) with memory cleanup between batches.

3. **Progress UI**: Enhance logging output to include progress indicators that can be monitored through stderr output.

4. **Configuration**: Add new environment variables for performance tuning:
   - `INDEXING_BATCH_SIZE` (default: 20)
   - `INDEXING_CONCURRENCY` (default: 4)
   - `QDRANT_BATCH_SIZE` (default: 500)
   - `MEMORY_WARNING_THRESHOLD_MB` (default: 1000)

## Technical Considerations

1. **Thread Safety**: Ensure all concurrent operations are thread-safe, especially Qdrant client operations.

2. **Resource Management**: Implement proper resource cleanup for file handles and database connections in concurrent environments.

3. **Error Recovery**: Design the system to handle partial failures gracefully and provide clear error messages for debugging.

4. **Backward Compatibility**: Maintain existing MCP tool signatures while adding optional parameters for new functionality.

5. **Testing Strategy**: Ensure comprehensive testing with large synthetic codebases to validate performance improvements.

## Success Metrics

1. **Performance**: Successfully index test repositories that currently timeout (target: completion without errors)
2. **Concurrency**: Enable search operations while indexing is in progress
3. **Progress Visibility**: Provide progress updates at least every 100 processed files
4. **Memory Efficiency**: Process large codebases within reasonable memory limits (configurable threshold)
5. **Resumability**: Successfully resume interrupted indexing operations with <5% duplicate work
6. **User Experience**: Reduce user wait time by enabling background processing

## Open Questions

1. What is the exact file count and size of your problematic repository? (pending test command results)
2. Should the system auto-detect optimal batch sizes based on available system resources?
3. Do you prefer a simple task queue or more sophisticated job management system?
4. Should there be limits on maximum concurrent indexing operations?
5. How should the system handle very large individual files (>10MB)?

## Implementation Priority

**Phase 1 (High Priority)**:
- Parallel file processing
- Batch embedding generation
- Basic progress logging

**Phase 2 (Medium Priority)**:
- Asynchronous background indexing
- Incremental indexing with change detection
- Repository analysis tools

**Phase 3 (Nice to Have)**:
- Advanced memory monitoring
- Performance auto-tuning
- Enhanced error recovery
