# Product Requirements Document: ChunkType FILE Attribute Error Fix

## 1. Introduction/Overview

The Agentic-RAG codebase is experiencing critical errors during manual indexing of TypeScript/JSX files (.tsx). The error logs show two primary issues:
1. `type object 'ChunkType' has no attribute 'FILE'`
2. `CodeChunk.__init__() missing 3 required positional arguments: 'chunk_id', 'start_byte', and 'end_byte'`

These errors are preventing proper indexing of .tsx files, resulting in incomplete data storage (e.g., only 39 lines indexed instead of 225 lines for input.tsx). The system was originally designed to fall back to whole-file chunking when intelligent parsing fails, but the current implementation has inconsistent enum references and constructor calls.

## 2. Goals

### Primary Goals
- Eliminate `ChunkType.FILE` attribute errors during .tsx file processing
- Fix CodeChunk constructor calls missing required arguments
- Ensure complete indexing of .tsx files (full line count preservation)
- Maintain backward compatibility with existing indexed data

### Secondary Goals
- Improve error handling for syntax errors across all supported languages
- Enhance test coverage for fallback mechanisms
- Update documentation to reflect correct ChunkType usage

## 3. User Stories

**As a developer using the manual indexing tool:**
- I want .tsx files to be indexed completely without errors
- I want to see accurate line counts in Qdrant matching the actual file content
- I want the system to gracefully handle parsing failures with proper fallback

**As a system administrator:**
- I want to re-index affected files that were previously corrupted
- I want confidence that the indexing system won't fail on syntax errors
- I want clear error reporting when issues occur

**As a codebase maintainer:**
- I want consistent enum usage throughout the codebase
- I want proper constructor patterns for CodeChunk instances
- I want comprehensive test coverage for edge cases

## 4. Functional Requirements

1. **ChunkType Enum Consistency**: The system must use consistent ChunkType enum values throughout the codebase, replacing any `ChunkType.FILE` references with `ChunkType.WHOLE_FILE`.

2. **CodeChunk Constructor Validation**: All CodeChunk instantiations must include the required parameters: `chunk_id`, `start_byte`, and `end_byte`.

3. **Fallback Mechanism Enhancement**: When intelligent parsing fails, the system must successfully create whole-file chunks using `ChunkType.WHOLE_FILE`.

4. **Complete File Indexing**: The system must index the complete content of .tsx files, preserving accurate line counts and file size metadata.

5. **Error Recovery**: The system must handle parsing failures gracefully without stopping the entire indexing process.

6. **Cross-Language Support**: The fix must not break existing functionality for other supported languages (Python, JavaScript, TypeScript, Go, Rust, Java, C++).

7. **Incremental Re-indexing**: The system must support re-indexing of previously corrupted files without affecting correctly indexed files.

8. **Test Coverage**: All chunk type handling and constructor patterns must have corresponding unit tests.

## 5. Non-Goals (Out of Scope)

- Adding new ChunkType enum values beyond fixing the existing FILE/WHOLE_FILE inconsistency
- Changing the fundamental architecture of the intelligent chunking system
- Modifying the Tree-sitter parser integration
- Adding support for new programming languages
- Performance optimizations (focus is on correctness)
- UI/UX improvements to the manual indexing tool
- Backwards compatibility with very old data formats

## 6. Technical Considerations

- **Existing Data**: Must preserve existing Qdrant collections that use `ChunkType.WHOLE_FILE.value`
- **Import Dependencies**: Changes should not require updating import statements across the codebase
- **Testing Framework**: Use existing pytest framework for unit tests
- **Error Logging**: Maintain existing logging patterns for debugging
- **AST Extraction Service**: Ensure compatibility with existing AST extraction logic

## 7. Success Metrics

- **Zero ChunkType.FILE Errors**: No more "has no attribute 'FILE'" errors in logs
- **Zero Constructor Errors**: No more "missing positional arguments" errors for CodeChunk
- **Complete File Indexing**: .tsx files show correct line counts in Qdrant (225 lines vs previous 39 lines)
- **Test Pass Rate**: 100% test pass rate for updated chunk handling logic
- **Re-indexing Success**: Affected files can be successfully re-indexed with complete content

## 8. Open Questions

- Should we add a migration script to update any existing database entries that might have incorrect chunk types?
- Do we need to audit other file types beyond .tsx for similar issues?
- Should we implement additional validation in the CodeChunk constructor to prevent future similar issues?
- Are there other parts of the codebase that might be using hardcoded "file" string values instead of enum references?
