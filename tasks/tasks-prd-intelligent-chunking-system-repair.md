## Relevant Files

- `src/services/code_parser_service.py` - Core service for Tree-sitter parsing and intelligent chunking
- `src/services/code_parser_service.test.py` - Unit tests for code parser service
- `src/services/indexing_service.py` - Main indexing service that uses code parser
- `src/services/indexing_service.test.py` - Unit tests for indexing service  
- `src/models/code_chunk.py` - Data models for different chunk types
- `src/models/code_chunk.test.py` - Unit tests for chunk models
- `requirements.txt` - Python dependencies including Tree-sitter packages
- `pyproject.toml` - Poetry dependencies configuration
- `src/utils/tree_sitter_manager.py` - New utility for managing Tree-sitter dependencies
- `src/utils/tree_sitter_manager.test.py` - Unit tests for Tree-sitter manager
- `src/utils/language_registry.py` - New centralized language support registry
- `src/utils/language_registry.test.py` - Unit tests for language registry
- `manual_indexing.py` - Manual indexing tool that needs chunking validation
- `src/mcp_tools.py` - MCP tools that trigger indexing workflows

### Notes

- Unit tests should typically be placed alongside the code files they are testing (e.g., `code_parser_service.py` and `code_parser_service.test.py` in the same directory).
- Use `npx jest [optional/path/to/test/file]` to run tests. Running without a path executes all tests found by the Jest configuration.

## Tasks

- [x] 1.0 Diagnose and Fix Tree-sitter Dependencies
  - [x] 1.1 Investigate Tree-sitter version compatibility issues and `PyCapsule` errors
  - [x] 1.2 Update Tree-sitter dependencies to compatible versions in requirements.txt and pyproject.toml
  - [x] 1.3 Create TreeSitterManager utility class for robust parser initialization
  - [x] 1.4 Fix Language object validation in CodeParserService._initialize_parsers()
  - [x] 1.5 Add comprehensive error logging for Tree-sitter initialization failures
  - [x] 1.6 Test parser initialization for all supported languages (Python, JS, TS, Go, Rust, Java)

- [ ] 2.0 Restore Intelligent Chunking for Existing Languages
  - [x] 2.1 Verify and fix AST node mappings for Python (functions, classes, methods, imports)
  - [x] 2.2 Verify and fix AST node mappings for JavaScript/TypeScript (functions, classes, interfaces, exports)
  - [x] 2.3 Verify and fix AST node mappings for Java (methods, classes, interfaces, imports)
  - [x] 2.4 Verify and fix AST node mappings for Go (functions, structs, interfaces, imports)
  - [x] 2.5 Verify and fix AST node mappings for Rust (functions, structs, enums, modules)
  - [ ] 2.6 Test intelligent chunking end-to-end with sample files for each language
  - [ ] 2.7 Validate that chunk metadata includes proper semantic information (signatures, docstrings, etc.)

- [ ] 3.0 Add C++ Language Support
  - [ ] 3.1 Add tree-sitter-cpp dependency to requirements.txt and pyproject.toml
  - [ ] 3.2 Extend CodeParserService._supported_languages to include C++ variants (cpp, c, hpp, h)
  - [ ] 3.3 Define AST node mappings for C++ (functions, classes, namespaces, includes, templates)
  - [ ] 3.4 Update language detection in IndexingService to properly map C++ file extensions
  - [ ] 3.5 Create comprehensive test cases for C++ header (.hpp) and source (.cpp) files
  - [ ] 3.6 Validate C++ template and namespace parsing works correctly

- [ ] 4.0 Enhance Error Handling and Fallback Mechanisms
  - [ ] 4.1 Create LanguageRegistry utility to centralize language support information
  - [ ] 4.2 Improve syntax error recovery in CodeParserService.parse_file()
  - [ ] 4.3 Add detailed logging for parse failures with specific error types and locations
  - [ ] 4.4 Implement chunking quality validation (ensure chunks have valid boundaries)
  - [ ] 4.5 Add metrics tracking for chunking success rates per language
  - [ ] 4.6 Create diagnostic tools to verify Tree-sitter parser health
  - [ ] 4.7 Update fallback logic to only use whole-file chunking when absolutely necessary

- [ ] 5.0 System Cleanup and Data Migration
  - [ ] 5.1 Create data migration script to identify and remove invalid whole-file chunks
  - [ ] 5.2 Implement reindexing workflow for existing codebases with new intelligent chunking
  - [ ] 5.3 Add validation tools to verify chunk quality after migration
  - [ ] 5.4 Update manual_indexing.py to report chunking statistics and quality metrics
  - [ ] 5.5 Create system health check for ongoing monitoring of chunking performance
  - [ ] 5.6 Document migration procedures and rollback plans