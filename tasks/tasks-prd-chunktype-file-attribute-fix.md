# Tasks: ChunkType FILE Attribute Error Fix

## Relevant Files

- `src/models/code_chunk.py` - Contains ChunkType enum definition, verified WHOLE_FILE exists ✅
- `src/services/chunking_strategies.py` - Fixed ChunkType.FILE references to ChunkType.WHOLE_FILE ✅
- `src/services/ast_extraction_service.py` - Fixed CodeChunk constructor calls with missing parameters ✅
- `src/services/code_parser_service.py` - Fixed CodeSyntaxError constructor calls ✅
- `trees/documentation-modernization-1/src/services/chunking_strategies.py` - Fixed ChunkType.FILE references ✅
- `manual_indexing.py` - Manual indexing tool tested with TSX files ✅
- `test_tsx_fallback.py` - Created custom test script for TSX fallback chunking ✅
- `test_sample.tsx` - Created test TSX file for verification ✅
- `test_dir/textarea.tsx` - Created test TSX file for verification ✅
- `test_dir/select.tsx` - Created test TSX file for verification ✅

### Notes

- The ChunkType enum in `code_chunk.py` already has `WHOLE_FILE = "whole_file"` (line 54)
- Current tests use `ChunkType.WHOLE_FILE` correctly, confirming this is the right enum value
- Focus on finding and fixing `ChunkType.FILE` references and CodeChunk constructor issues
- Test with `.tsx` files specifically since that's where the errors occur

## Tasks

- [x] 1.0 Identify and Fix ChunkType.FILE References
  - [x] 1.1 Search for all ChunkType.FILE references in codebase using grep
  - [x] 1.2 Update ChunkType.FILE to ChunkType.WHOLE_FILE in src/services/chunking_strategies.py (lines 254, 452)
  - [x] 1.3 Update ChunkType.FILE to ChunkType.WHOLE_FILE in trees/documentation-modernization-1/src/services/chunking_strategies.py
  - [x] 1.4 Verify no other ChunkType.FILE references exist in the codebase

- [x] 2.0 Fix CodeChunk Constructor Missing Arguments
  - [x] 2.1 Locate CodeChunk instantiations missing chunk_id, start_byte, end_byte parameters
  - [x] 2.2 Fix FallbackChunkingStrategy._extract_chunks method to include missing parameters
  - [x] 2.3 Fix StructuredFileChunkingStrategy._extract_single_chunk method to include missing parameters
  - [x] 2.4 Add proper chunk_id generation using file path and content hash
  - [x] 2.5 Calculate start_byte and end_byte values for whole-file chunks

- [x] 3.0 Update Tests and Verify Functionality
  - [x] 3.1 Run existing tests to ensure no regressions from ChunkType.WHOLE_FILE changes
  - [x] 3.2 Update any tests that might reference ChunkType.FILE
  - [x] 3.3 Add specific test case for .tsx file fallback chunking
  - [x] 3.4 Verify test_fallback_chunking passes with updated constructor calls
  - [x] 3.5 Run pytest tests/test_code_parser_service.py and tests/test_syntax_error_handling.py

- [x] 4.0 Test Manual Indexing with TSX Files
  - [x] 4.1 Test manual indexing with the problematic /Users/jeff/Documents/personal/claudia/src/components/ui/input.tsx file
  - [x] 4.2 Verify complete file indexing (225 lines instead of 39 lines)
  - [x] 4.3 Check Qdrant for correct chunk storage and metadata
  - [x] 4.4 Test other .tsx files (textarea.tsx, select.tsx) from the error log
  - [x] 4.5 Verify no ChunkType.FILE or constructor errors in logs

- [x] 5.0 Documentation and Cleanup
  - [x] 5.1 Update any documentation that references ChunkType.FILE
  - [x] 5.2 Add comments explaining the fallback mechanism in chunking strategies
  - [x] 5.3 Update INTELLIGENT_CHUNKING_GUIDE.md if needed
  - [x] 5.4 Run final verification with uvx --directory . run pytest tests/ to ensure all tests pass
