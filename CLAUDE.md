# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

## Architecture Overview

This is a **Codebase RAG (Retrieval-Augmented Generation) MCP Server** that enables AI agents to understand and query codebases using natural language with **function-level precision** through intelligent syntax-aware code chunking.

### Project Structure

```
src/
├── main.py                    # MCP server entry point
├── run_mcp.py                 # Server startup script
├── models/                    # Data models and structures
│   ├── code_chunk.py         # Intelligent chunk representations
│   └── file_metadata.py      # File tracking and metadata
├── services/                  # Core business logic
│   ├── code_parser_service.py    # AST parsing and chunking
│   ├── indexing_service.py       # Orchestration and processing
│   ├── embedding_service.py      # Ollama integration
│   ├── qdrant_service.py         # Vector database operations
│   └── project_analysis_service.py # Repository analysis
├── tools/                     # MCP tool implementations
│   ├── core/                 # Error handling and utilities
│   ├── indexing/             # Parsing and chunking tools
│   └── project/              # Project management tools
├── utils/                     # Shared utilities
│   ├── language_registry.py     # Language support definitions
│   ├── tree_sitter_manager.py   # Parser management
│   └── performance_monitor.py   # Progress tracking
└── prompts/                   # Advanced query prompts (future)

Root Files:
├── manual_indexing.py         # Standalone indexing tool
├── pyproject.toml            # uv/Python configuration
└── docs/                     # Documentation (referenced)

6. **Data Flow**

   **Full Indexing Flow with Intelligent Chunking:**
   ```
   Source Code → File Discovery → AST Parsing → Intelligent Chunking →
   Function/Class Extraction → Batch Embedding → Streaming DB Storage → Metadata Storage
   ```

   **Incremental Indexing Flow:**
   ```
   File Discovery → Change Detection → Selective Processing →
   AST Re-parsing → Smart Chunking → Embedding → DB Updates → Metadata Updates
   ```

   **Query Flow with Precision Results:**
   ```
   Natural Language → Embedding → Vector Search → Function-Level Matches →
   Context Enhancement → Ranked Results with Breadcrumbs
   ```

   **⚠️ IMPORTANT: Async/Await Search Chain**
   The search pipeline uses async/await throughout. Key async functions:
   - `search_sync()` in `search_tools.py` - Main entry point (despite name, is async)
   - `SearchService.search()` - Async search orchestration
   - `EmbeddingService.generate_embeddings()` - Returns async generator
   - `CodeParserService.parse_file()` - Async file parsing

   Common async issues and fixes:
   ```python
   # ❌ WRONG - Missing await causes "coroutine was never awaited"
   embeddings = embedding_service.generate_embeddings([query])

   # ✅ CORRECT - Properly await async calls
   embeddings = await embedding_service.generate_embeddings([query])

   # ❌ WRONG - Can't iterate coroutine directly
   for chunk in parser.parse_file(path):

   # ✅ CORRECT - Await async function first
   chunks = await parser.parse_file(path)
   for chunk in chunks:
   ```

   **Manual Tool Flow:**
   ```
   CLI Input → Validation → Pre-analysis → Progress Tracking →
   AST Processing → Core Processing → Syntax Error Reporting
   ```

### Configuration

Environment variables (`.env` file):

### MCP Tools Available

#### `index_directory(directory, patterns, recursive, clear_existing, incremental, project_name)`
#### `search(query, n_results, cross_project, search_mode, include_context, context_chunks, target_projects)`

## Intelligent Code Chunking System

### Overview
The system uses **Tree-sitter** parsers to perform syntax-aware code analysis, breaking down source code into semantically meaningful chunks (functions, classes, methods) instead of processing entire files as single units.

### Supported Languages and Chunk Types

**Fully Implemented Languages:**
- **Python (.py, .pyw, .pyi)**: Functions, classes, methods, constants, docstrings, decorators
- **JavaScript (.js, .jsx, .mjs, .cjs)**: Functions, classes, modules, arrow functions
- **TypeScript (.ts)**: Interfaces, types, classes, functions, generics, annotations
- **TypeScript JSX (.tsx)**: React components, interfaces, types, functions
- **Go (.go)**: Functions, structs, interfaces, methods, packages
- **Rust (.rs)**: Functions, structs, impl blocks, traits, modules, macros
- **Java (.java)**: Classes, methods, interfaces, annotations, generics
- **C++ (.cpp, .cxx, .cc, .c, .hpp, .hxx, .hh, .h)**: Functions, classes, structs, namespaces, templates

**Structured Files:**
- **JSON/YAML**: Object-level chunking (e.g., `scripts`, `dependencies` as separate chunks)
- **Markdown**: Header-based hierarchical chunking with section organization
- **Configuration Files**: Section-based parsing with semantic grouping

### Chunk Metadata Schema

Each intelligent chunk includes rich metadata:
```python
{
    "content": str,              # Actual code content
    "file_path": str,            # Source file path
    "chunk_type": str,           # function|class|method|interface|constant
    "name": str,                 # Function/class name
    "signature": str,            # Function signature or class inheritance
    "start_line": int,           # Starting line number
    "end_line": int,             # Ending line number
    "language": str,             # Programming language
    "docstring": str,            # Extracted documentation
    "access_modifier": str,      # public|private|protected
    "parent_class": str,         # Parent class for methods
    "has_syntax_errors": bool,   # Syntax error flag
    "error_details": str,        # Error description if any
    "chunk_id": str,             # Unique identifier
    "content_hash": str,         # Content hash for change detection
}
```

### Incremental Indexing Workflow

1. **Initial Indexing**: Full codebase processing with metadata storage
2. **Change Detection**: Compare file modification times and content hashes
3. **Selective Processing**: Only reprocess files with detected changes
4. **Metadata Updates**: Update file metadata after successful processing
5. **Collection Management**: Automatic cleanup of stale entries

### Collection Architecture

**Content Collections** (store intelligent chunks with embeddings):
- `project_{name}_code`: **Intelligent code chunks** - functions, classes, methods from (.py, .js, .java, etc.)
- `project_{name}_config`: **Structured config chunks** - JSON/YAML objects, configuration sections
- `project_{name}_documentation`: **Document chunks** - Markdown headers, documentation sections

**Metadata Collection** (tracks file states):
- `project_{name}_file_metadata`: File change tracking for incremental indexing
  - Stores: file_path, mtime, content_hash, file_size, indexed_at, syntax_error_count
  - Used for: change detection, incremental processing, progress tracking, error monitoring

## Critical Implementation Notes

### Async/Await Patterns in Search Pipeline

The entire search pipeline is **async** and requires proper await handling:

1. **Search Entry Points**:
   ```python
   # search_tools.py - Main MCP tool entry
   async def search_sync(...):  # Despite name, this is async!
       search_service = SearchService(...)
       results = await search_service.search(...)  # Must await
   ```

2. **Common Async Chain Issues**:
   - `RuntimeWarning: coroutine 'X' was never awaited` - Missing await
   - `'coroutine' object has no attribute 'Y'` - Trying to use coroutine without await
   - `TypeError: 'async_generator' object is not iterable` - Need async for

3. **Key Async Functions**:
   - `EmbeddingService.generate_embeddings()` - Async embedding generation
   - `SearchService.search()` - Main search orchestration
   - `SearchCacheService.get_cached_results()` - Cache retrieval
   - `CodeParserService.parse_file()` - File parsing (if used in search)
   - `QdrantService.search()` - Vector database queries

4. **Debugging Async Issues**:
   ```python
   # Add these debug prints to trace async flow:
   print(f"Before await: {type(result)}")  # Shows <class 'coroutine'>
   result = await some_async_function()
   print(f"After await: {type(result)}")   # Shows actual result type
   ```
