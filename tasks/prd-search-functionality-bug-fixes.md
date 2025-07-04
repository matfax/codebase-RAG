# PRD: Search Functionality Bug Fixes

## Introduction/Overview

This PRD addresses critical bugs in the MCP search functionality that are affecting search result accuracy and user experience. Through analysis of the search implementation, we have identified two primary issues that need immediate resolution:

1. **n_results Parameter Multiplication Bug**: The search function incorrectly multiplies the requested number of results by the number of collections being searched, leading to excessive results.
2. **Empty Content in Search Results**: Some search results return empty content despite having valid metadata, indicating incomplete data storage in the vector database.

These issues directly impact the reliability and usability of the codebase RAG search system, affecting both direct user queries and AI agent interactions.

## Goals

1. **Fix Result Count Accuracy**: Ensure search results respect the n_results parameter exactly
2. **Eliminate Empty Content Results**: Resolve issues causing empty content in search results
3. **Improve Search Reliability**: Enhance the overall consistency and predictability of search behavior
4. **Maintain Backward Compatibility**: Ensure fixes don't break existing functionality
5. **Add Robust Testing**: Implement comprehensive tests to prevent regression

## User Stories

1. **As a developer using the search tool**, I want to receive exactly the number of results I request (e.g., n_results=5), so that I can get focused, manageable results.

2. **As an AI agent**, I want search results to always contain actual content, so that I can provide meaningful responses based on the retrieved information.

3. **As a system user**, I want consistent search behavior across different collections and projects, so that I can rely on the search functionality for my workflows.

4. **As a developer**, I want clear error handling when search results are malformed or incomplete, so that I can understand and resolve issues quickly.

5. **As a maintainer**, I want comprehensive test coverage for search functionality, so that I can prevent similar bugs in the future.

## Functional Requirements

### n_results Parameter Fix

1. The search function **must** return exactly the number of results specified by the n_results parameter
2. The system **must** aggregate results from multiple collections and then limit to n_results, not multiply by collection count
3. The system **must** maintain proper result ranking across collections when limiting results
4. The system **must** handle edge cases where fewer results are available than requested
5. The system **must** preserve existing search mode functionality (semantic, keyword, hybrid)

### Empty Content Resolution

6. The search function **must** filter out results with empty content before returning to users
7. The system **must** identify and report files with empty content in the vector database
8. The system **must** provide a mechanism to re-index files that have empty content
9. The system **must** validate content integrity during the indexing process
10. The system **must** log warnings when empty content is detected during search

### Error Handling and Diagnostics

11. The system **must** provide clear error messages when search results are malformed
12. The system **must** include diagnostic information about result count discrepancies
13. The system **must** log detailed information about search operations for debugging
14. The system **must** provide utilities to diagnose and repair vector database inconsistencies

### Testing and Validation

15. The system **must** include unit tests for the n_results parameter behavior
16. The system **must** include integration tests for multi-collection search scenarios
17. The system **must** include tests for empty content detection and handling
18. The system **must** include performance tests to ensure fixes don't degrade search speed

## Non-Goals (Out of Scope)

1. **Search Algorithm Optimization**: Will not modify the underlying search algorithms or ranking mechanisms
2. **New Search Features**: Will not add new search modes or capabilities
3. **Database Schema Changes**: Will not modify the vector database schema or collection structure
4. **UI/UX Improvements**: Will not change the search interface or result presentation
5. **Performance Optimization**: Will not focus on search speed improvements beyond maintaining current performance

## Technical Considerations

### Primary Code Locations

1. **Search Function Fix**: `src/tools/indexing/search_tools.py` line 346 - modify result limiting logic
2. **Content Validation**: `src/tools/indexing/search_tools.py` - add content filtering
3. **Collection Handling**: `src/tools/indexing/search_tools.py` - improve multi-collection aggregation
4. **Error Handling**: `src/tools/core/error_utils.py` - enhance search error reporting

### Implementation Details

1. **Result Limiting Logic**: Change from `n_results * len(search_collections)` to `n_results`
2. **Content Filtering**: Add validation to filter out empty content before returning results
3. **Aggregation Strategy**: Sort all results by score before limiting, not per-collection
4. **Diagnostic Tools**: Add utilities to identify and report database inconsistencies

### Testing Strategy

1. **Unit Tests**: Test search function behavior with various n_results values and collection counts
2. **Integration Tests**: Test end-to-end search scenarios with real data
3. **Edge Case Tests**: Test with empty collections, malformed data, and boundary conditions
4. **Performance Tests**: Ensure fixes don't significantly impact search performance

## Success Metrics

1. **Result Count Accuracy**: Search function returns exactly n_results items in 100% of cases
2. **Content Completeness**: Zero search results with empty content after fixes
3. **Test Coverage**: Achieve 95%+ test coverage for search functionality
4. **Performance Maintenance**: Search performance remains within 10% of current benchmarks
5. **Bug Reduction**: Zero reported issues related to result count or empty content after deployment

## Current Bug Analysis

### Bug 1: n_results Multiplication
- **Location**: `src/tools/indexing/search_tools.py:346`
- **Issue**: `return all_results[: n_results * len(search_collections)]`
- **Impact**: Requesting 5 results with 3 collections returns 15 results
- **Fix**: `return all_results[:n_results]`

### Bug 2: Empty Content
- **Location**: Vector database storage/retrieval
- **Issue**: Some chunks stored with empty content field
- **Impact**: Search results show metadata but no actual content
- **Fix**: Add content validation and filtering

## Open Questions

1. Should we implement automatic re-indexing for files with empty content?
2. How should we handle partial results when some collections have issues?
3. Should we add metrics to track search result quality over time?
4. How should we communicate these fixes to existing users of the system?

## Dependencies

- Qdrant vector database
- Tree-sitter parsers for content extraction
- Existing indexing pipeline
- Current test infrastructure
