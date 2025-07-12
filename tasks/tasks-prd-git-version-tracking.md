# Product Requirements Document: Git Version Tracking for Agentic-RAG

## Introduction/Overview

This feature enhances the Agentic-RAG indexing system by adding Git version information tracking capabilities. Currently, the system indexes codebases without any awareness of version control state, making it impossible to distinguish between different versions or branches of the same code. This enhancement will enable developers to search and compare code across different Git branches and commits, providing crucial context for understanding code evolution and branch-specific implementations.

## Goals

1. **Automatic Git Information Collection**: Automatically detect and collect Git commit hash, branch name, repository URL, and commit timestamp during indexing
2. **Version-Aware Search**: Enable code search filtering by Git branch or specific commit
3. **Version Status Visibility**: Display version control information in project listings
4. **Comparison Foundation**: Establish the infrastructure for future cross-version code comparison features

## User Stories

1. **As a developer**, I want to search for function implementations in a specific branch, so that I can understand feature-branch-specific implementation details without interference from other branches.

2. **As a developer**, I want to know which commit version the indexed code comes from, so that I can trace code changes and ensure I'm looking at the correct version.

3. **As a developer**, I want to see version control status when listing indexed projects, so that I can quickly identify which versions have been indexed.

4. **As a developer**, I want to index multiple versions of the same project simultaneously, so that I can compare implementations across different branches or commits.

5. **As a team lead**, I want to search for code in the main branch only, so that I can review production code without seeing experimental changes.

## Functional Requirements

1. **Git Information Detection and Collection**
   - The system must automatically detect if a directory is under Git version control
   - The system must collect: full commit hash (40 characters), branch name, repository URL, and commit timestamp
   - The system must handle special Git states: detached HEAD, Git worktrees, and corrupted repositories
   - The system must gracefully handle non-Git directories by setting Git fields to null/None

2. **Data Model Extension**
   - The system must extend FileMetadata model with Git-related fields while maintaining backward compatibility
   - The system must store Git information as part of chunk metadata in Qdrant collections
   - The system must preserve existing indexed data without Git information (backward compatibility)

3. **Version-Filtered Search**
   - The system must support optional `git_branch` parameter in search queries
   - The system must support optional `git_commit` parameter in search queries
   - The system must return Git information as part of search results metadata
   - The system must handle searches across projects with mixed Git/non-Git status

4. **Project Listing Enhancement**
   - The system must display Git branch and commit information in `list_indexed_projects` output
   - The system must clearly indicate projects without version control
   - The system must show special Git states (detached HEAD, worktree) in a user-friendly format

5. **Special State Handling**
   - The system must detect and properly label detached HEAD states as "detached@{short_hash}"
   - The system must support Git worktrees by detecting and following `.git` file references
   - The system must treat corrupted Git repositories as non-version-controlled
   - The system must handle permission errors when accessing `.git` directory

## Non-Goals (Out of Scope)

1. **Git History Analysis**: This feature will NOT analyze or store Git commit history beyond the current state
2. **Branch Merge Tracking**: This feature will NOT track branch relationships or merge history
3. **Remote Repository Sync**: This feature will NOT sync with or fetch from remote repositories
4. **Commit Diff Calculation**: This feature will NOT calculate or store diffs between commits
5. **Git Operations**: This feature will NOT perform any Git operations (checkout, pull, commit, etc.)
6. **Author Information**: This feature will NOT collect or store Git author/committer information

## Design Considerations

### Data Architecture
- Collection naming remains unchanged to avoid collection proliferation
- Git information stored purely as metadata within existing collections
- Version filtering implemented through metadata queries rather than separate collections

### User Interface
- Git information displayed in a concise, readable format: `project_name@branch:short_hash`
- Special states clearly indicated: "No version control", "Detached HEAD", "Worktree"
- Search results include Git context in metadata section

## Technical Considerations

1. **New Service Layer**: Create `GitContextService` to encapsulate all Git information reading logic
2. **Direct File Reading**: Read Git information directly from `.git/` directory structure rather than using git commands for better performance and reliability
3. **Caching Strategy**: Cache Git information during indexing session to avoid repeated file system access
4. **Error Handling**: Implement comprehensive error handling for various Git states and edge cases
5. **Performance Impact**: Git information reading adds minimal overhead (<50ms per project)
6. **Storage Impact**: Additional metadata fields add approximately 200 bytes per file entry

## Success Metrics

1. **Functional Success**
   - 100% of Git-controlled projects correctly identified and Git information collected
   - Version-filtered searches return only results from specified branch/commit
   - All special Git states (detached HEAD, worktrees) properly handled

2. **Performance Metrics**
   - Git information collection adds less than 5% to total indexing time
   - Search performance with version filters maintains sub-second response times
   - No impact on search performance for queries without version filters

3. **User Experience**
   - Developers can successfully search within specific branches
   - Version information clearly visible in project listings
   - Intuitive handling of edge cases (non-Git projects, special states)

4. **Reliability**
   - System continues to function normally for non-Git projects
   - Corrupted Git repositories don't cause indexing failures
   - Backward compatibility maintained for existing indexed data

## Open Questions

1. **Multi-Version Storage Strategy**: Should we limit the number of versions indexed per project to prevent unlimited growth?
2. **Version Comparison UI**: What's the most intuitive way to present cross-version comparison results in future iterations?
3. **Worktree Naming**: Should worktrees be treated as separate projects or variants of the main project?
4. **Branch Switching**: How should the system handle when a user switches branches in an already-indexed directory?
5. **Partial Indexing**: Should we support indexing only specific files that changed between versions?
