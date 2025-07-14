# PRD: Point ID Database Fix

## Introduction/Overview

The Qdrant vector database is rejecting Point IDs during the file metadata storage phase of indexing because the current implementation generates 16-character hex strings (e.g., "fedfda3b08808e5f") instead of valid UUIDs or unsigned integers. This causes 100% failure rate for metadata storage, breaking the incremental indexing capability. The goal is to fix Point ID generation to ensure successful database writes while maintaining deterministic ID generation for consistent file tracking.

## Goals

1. Achieve 100% success rate for Qdrant database writes by generating valid Point IDs
2. Maintain deterministic Point ID generation (same file path always generates same ID)
3. Preserve indexing performance at minimum 50% of current speed
4. Provide early validation with clear error messages before attempting database operations
5. Enable both manual indexing and MCP users to successfully store metadata

## User Stories

1. As a developer using manual indexing, I want all chunks to be successfully stored in Qdrant so that my indexing process completes without errors
2. As an MCP user, I want reliable database writes so that my indexed data is properly stored and searchable
3. As a developer, I want early validation of Point IDs so that I'm warned about issues before the indexing process starts
4. As a user with existing invalid Point IDs, I want clear guidance on how to resolve the issue using clear_existing mode

## Functional Requirements

1. The system must generate Point IDs that are valid UUIDs (format: "a1b2c3d4-e5f6-7890-abcd-ef1234567890")
2. The system must use deterministic UUID generation based on file paths (using uuid5 with namespace)
3. The system must validate Point ID format before attempting database insertion
4. The system must detect existing invalid Point IDs during pre-indexing validation
5. The system must display a warning message when invalid Point IDs are detected, suggesting clear_existing mode
6. The system must maintain or improve current indexing performance (minimum 50% of original speed)
7. The system must apply the fix to both FileMetadataService and any other services generating Point IDs
8. The system must generate consistent Point IDs across different runs for the same file path

## Non-Goals (Out of Scope)

1. This feature will NOT migrate existing invalid Point IDs in the database
2. This feature will NOT maintain backward compatibility with hex string Point IDs
3. This feature will NOT change the Point ID generation strategy for non-metadata collections
4. This feature will NOT modify the Qdrant database schema or configuration
5. This feature will NOT implement automatic cleanup of invalid Point IDs

## Technical Considerations

1. Use Python's uuid.uuid5() with a consistent namespace for deterministic UUID generation
2. Consider using relative file paths as input to uuid5() for consistent IDs across different absolute paths
3. Implement Point ID validation using regex pattern matching for UUID format
4. Add validation early in the indexing pipeline to fail fast on Point ID issues
5. Ensure the fix is applied consistently across all Point ID generation locations
6. Consider performance impact of UUID generation vs current hex string generation

## Success Metrics

1. 100% success rate for Qdrant database writes (no Point ID format errors)
2. Indexing performance maintains at least 50% of pre-fix speed
3. Zero Point ID validation errors during normal operation
4. Clear error messages displayed when validation fails
5. Successful storage of file metadata for both manual indexing and MCP operations

## Open Questions

1. Should we use a project-specific namespace for uuid5 or a global namespace?
2. What specific warning message format would be most helpful for users?
3. Should we log the mapping between old hex IDs and new UUIDs for debugging?
4. Is there a preferred performance benchmark tool for measuring the 50% speed requirement?
