# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup

**Quick Start:**
- `uv sync` - Install dependencies and create virtual environment
- `uvx --directory . run python src/run_mcp.py` - Start the MCP server to test installation

**Development Setup:**
- `uv add "mcp[cli]"` - Add MCP CLI support (if needed)
- `uv lock` - Update lock file if pyproject.toml changes
- `cp .env.example .env` - Copy environment configuration and customize as needed

### Registering with Claude Code

To use this MCP server with Claude Code:

```bash
claude mcp add codebase-rag-mcp \
  --command "uvx" \
  --args "--directory" \
  --args "." \
  --args "run" \
  --args "python" \
  --args "src/run_mcp.py"
```

This registers the server with Claude Code for use in conversations.

### Testing
- `uvx --directory . run pytest tests/` - Run all tests
- `uvx --directory . run pytest tests/test_specific.py` - Run specific test file
- `uvx --directory . run pytest tests/test_code_parser_service.py` - Test intelligent chunking service
- `uvx --directory . run pytest tests/test_intelligent_chunking.py` - Test chunking integration
- `uvx --directory . run python test_full_functionality.py` - Test basic MCP functionality
- `uvx --directory . run python test_mcp_stdio.py` - Test stdio communication
- `uvx --directory . run python demo_mcp_usage.py` - Run usage demo

### Manual Indexing Tool
- `uvx --directory . run python manual_indexing.py -d /path/to/repo -m clear_existing` - Full indexing
- `uvx --directory . run python manual_indexing.py -d /path/to/repo -m incremental` - Incremental indexing
- `uvx --directory . run python manual_indexing.py -d /path/to/repo -m incremental --verbose` - Verbose output
- `uvx --directory . run python manual_indexing.py -d /path/to/repo -m clear_existing --no-confirm` - Skip prompts

### Performance Testing and Validation
- Manual tool provides pre-indexing analysis with file count and time estimates
- Use `--verbose` flag for detailed logging and performance metrics
- Monitor memory usage and processing rates in logs

### External Dependencies
- **Qdrant**: `docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant`
- **Ollama**: Must be running locally with embedding models (e.g., `ollama pull nomic-embed-text`)
- **Tree-sitter**: Language parsers are automatically installed via uv dependencies for intelligent code chunking

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

   **Manual Tool Flow:**
   ```
   CLI Input → Validation → Pre-analysis → Progress Tracking →
   AST Processing → Core Processing → Syntax Error Reporting
   ```

### Key Architectural Decisions

- **Vector Database**: Qdrant for storing and searching embeddings
- **Embeddings**: Ollama with configurable models (default: `nomic-embed-text`)
- **🎯 Chunking Strategy**: **INTELLIGENT SYNTAX-AWARE CHUNKING** using Tree-sitter AST parsing
  - Function-level and class-level granular chunks instead of whole files
  - Supports 10+ programming languages with dedicated parsers
  - Graceful fallback to whole-file processing for syntax errors
  - Rich metadata extraction (signatures, docstrings, breadcrumbs)
- **Collection Organization**: Separate collections for code, config, documentation, and file metadata
- **Project Context**: Automatically detects project boundaries using `.git`, `pyproject.toml` markers
- **Incremental Processing**: File change detection using modification times and content hashes
- **Parallel Processing**: Multi-threaded file processing with configurable concurrency
- **Memory Management**: Streaming operations with memory monitoring and cleanup
- **Batch Optimization**: Batched operations for embeddings and database insertions
- **Error Tolerance**: Smart handling of syntax errors with detailed reporting and recovery mechanisms

### Configuration

Environment variables (`.env` file):

**Basic Configuration:**
- `OLLAMA_HOST`: Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_DEFAULT_EMBEDDING_MODEL`: Default embedding model (default: `nomic-embed-text`)
- `QDRANT_HOST`: Qdrant host (default: `localhost`)
- `QDRANT_PORT`: Qdrant port (default: `6333`)

**Performance Tuning:**
- `INDEXING_CONCURRENCY`: Parallel processing workers (default: `4`)
- `INDEXING_BATCH_SIZE`: Files per processing batch (default: `20`)
- `EMBEDDING_BATCH_SIZE`: Texts per embedding API call (default: `10`)
- `QDRANT_BATCH_SIZE`: Points per database batch (default: `500`)
- `MEMORY_WARNING_THRESHOLD_MB`: Memory usage warning threshold (default: `1000`)
- `MAX_FILE_SIZE_MB`: Maximum file size to process (default: `5`)
- `MAX_DIRECTORY_DEPTH`: Maximum directory traversal depth (default: `20`)

**Development Settings:**
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `FOLLOW_SYMLINKS`: Follow symbolic links (default: `false`)

### Primary MCP Tool

#### `search(query, n_results, cross_project, search_mode, include_context, context_chunks)`
- **🔍 Function-Level Precision**: Returns specific functions, classes, and methods instead of entire files
- **Natural Language**: Search indexed content using natural language queries
- **Project Scoping**: Search within current project or across all projects
- **Context Enhancement**: Include surrounding code context in results with breadcrumb navigation
- **Multiple Search Modes**: Hybrid, semantic, and keyword search options
- **Rich Metadata**: Results include function signatures, docstrings, and AST-derived information

**Example Usage:**
- "Find functions that handle file uploads"
- "Show me React components that use hooks"
- "Find error handling patterns in Python"

### Additional MCP Tools

For advanced users and developers, additional tools are available:
- `index_directory()`: Index a codebase for searching
- `health_check()`: Verify server connectivity and status
- `analyze_repository_tool()`: Get repository statistics and analysis

See `docs/MCP_TOOLS.md` for comprehensive tool documentation and `docs/BEST_PRACTICES.md` for optimization guides and workflows.

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

### Error Handling and Recovery

**Syntax Error Tolerance:**
- **ERROR Node Detection**: Identifies Tree-sitter ERROR nodes in AST
- **Partial Processing**: Preserves correct code sections even with syntax errors
- **Smart Recovery**: Includes surrounding context for better understanding
- **Graceful Fallback**: Automatically falls back to whole-file processing when AST parsing fails completely

**Error Classification:**
- **Minor Errors**: Small syntax issues that don't affect overall structure
- **Major Errors**: Significant parsing failures requiring fallback
- **Recoverable Errors**: Errors where partial content can still be extracted

**Error Reporting:**
- Detailed error statistics in manual indexing tool output
- Per-file error counts and locations
- Syntax error impact assessment
- Recovery strategy recommendations

### Performance Characteristics

**AST Parsing Performance:**
- **Single File Parsing**: < 100ms for typical files (99th percentile)
- **Memory Usage**: < 50MB additional per 1000 files
- **Total Processing Impact**: < 20% increase over whole-file approach

**Caching and Optimization:**
- **Parser Caching**: Tree-sitter parsers cached per language
- **AST Result Caching**: Based on file content hash for incremental updates
- **Memory Management**: Automatic cleanup between processing batches

### File Organization Patterns

- **Smart Filtering**: Uses `.ragignore` files for excluding directories/files from indexing
- **Automatic Categorization**: Files categorized by type (code/config/documentation/metadata)
- **Collection Naming**: Deterministic collection names: `project_{name}_{type}`
- **Binary Detection**: Automatically skips binary files and large files
- **Language Detection**: Identifies programming languages for better categorization
- **Gitignore Integration**: Respects `.gitignore` patterns for relevant file detection

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

**Chunk-Level Benefits:**
- **Precise Retrieval**: Search returns specific functions instead of entire files
- **Better Embeddings**: Each vector represents a complete semantic unit
- **Rich Context**: Function signatures, docstrings, and breadcrumb navigation
- **Error Isolation**: Syntax errors in one function don't affect others

### Performance Optimization Features

- **Parallel Processing**: Multi-threaded file reading and AST parsing
- **Intelligent Chunking**: Syntax-aware code analysis with Tree-sitter parsers
- **Parser Caching**: Cached language parsers for improved performance
- **Batch Operations**: Grouped embedding generation and database insertions
- **Streaming Architecture**: Memory-efficient processing for large codebases
- **Progress Monitoring**: Real-time progress tracking with ETA calculation and syntax error reporting
- **Memory Management**: Automatic cleanup and garbage collection with AST memory optimization
- **Adaptive Batching**: Dynamic batch size adjustment based on memory usage
- **Retry Logic**: Exponential backoff for failed operations with graceful fallback mechanisms
- **Early Filtering**: Skip excluded directories before processing
- **Error Recovery**: Smart handling of syntax errors with partial content preservation
