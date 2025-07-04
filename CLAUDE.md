# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv (modern Python package manager)
uv sync

# Run tests
uv run pytest tests/

# Run specific test module
uv run pytest tests/test_intelligent_chunking.py -v

# Run code quality checks
uv run ruff check src/
uv run black --check src/
```

### MCP Server Operations
```bash
# Register with Claude Code (recommended)
./register_mcp.sh

# Manual registration alternative
claude mcp add codebase-rag-mcp "$(pwd)/mcp_server"

# Test MCP server startup
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "1.0.0", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}}}' | uv run python src/run_mcp.py
```

### Manual Indexing Tool
```bash
# Full indexing with progress tracking
uv run python manual_indexing.py -d /path/to/repo -m clear_existing

# Incremental indexing (only changed files)
uv run python manual_indexing.py -d /path/to/repo -m incremental

# With verbose output and no confirmation prompts
uv run python manual_indexing.py -d /path/to/repo -m incremental --verbose --no-confirm
```

### Development Dependencies
- **Docker**: For running Qdrant vector database
- **Ollama**: For embedding generation (`ollama pull nomic-embed-text`)
- **uv**: Modern Python package manager

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

### Configuration

Environment variables (`.env` file):
```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_DEFAULT_EMBEDDING_MODEL=nomic-embed-text

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Performance Tuning
INDEXING_CONCURRENCY=4
INDEXING_BATCH_SIZE=20
EMBEDDING_BATCH_SIZE=10
QDRANT_BATCH_SIZE=500
MEMORY_WARNING_THRESHOLD_MB=1000
MAX_FILE_SIZE_MB=5
MAX_DIRECTORY_DEPTH=20

# Logging
LOG_LEVEL=INFO
```

### Core Service Architecture

**Primary Services:**
- **`IndexingService`**: Orchestrates parallel processing pipeline with batch optimization
- **`CodeParserService`**: Implements Tree-sitter AST parsing for intelligent code chunking
- **`QdrantService`**: Manages vector database operations with streaming and batch insertion
- **`EmbeddingService`**: Handles Ollama integration with automatic batching and retry logic
- **`ProjectAnalysisService`**: Analyzes project structure, respects .gitignore, filters relevant files

**Supporting Services:**
- **`FileMetadataService`**: Tracks file states for incremental indexing
- **`ChangeDetectorService`**: Detects file changes using modification times and content hashes
- **`IndexingReporter`**: Comprehensive reporting with progress tracking and ETA estimation
- **`SearchStrategies`**: Multiple search algorithms (semantic, keyword, hybrid)
- **`PerformanceMonitor`**: Real-time progress monitoring with memory usage tracking

### Data Flow Architecture

```
Source Code → File Discovery → AST Parsing → Intelligent Chunking →
Function/Class Extraction → Batch Embedding → Vector Storage → Metadata Tracking
```

### MCP Tools Available

#### `index_directory(directory, patterns, recursive, clear_existing, incremental, project_name)`
Index a directory with intelligent recommendations and time estimation.

#### `search(query, n_results, cross_project, search_mode, include_context, context_chunks, target_projects)`
Search indexed content using natural language with function-level precision.

#### Additional Tools
- **`health_check`**: Verify Qdrant and Ollama connectivity
- **`analyze_repository_tool`**: Get detailed repository statistics
- **`get_chunking_metrics_tool`**: Performance metrics for intelligent chunking
- **`diagnose_parser_health_tool`**: Tree-sitter parser diagnostics

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

## Key Development Patterns

### Service-Oriented Architecture
- Clear separation of concerns with specialized services
- Async/await throughout the codebase for concurrent operations
- Rich error handling with automatic retry and exponential backoff

### Performance Optimizations
- **Incremental Indexing**: 80%+ time savings by only processing changed files
- **Parallel Processing**: Configurable concurrency for large codebases
- **Batch Operations**: Optimized batch sizes for memory and network efficiency
- **Memory Management**: Automatic garbage collection with configurable thresholds

### Testing Strategy
```bash
# Run all tests
uv run pytest tests/

# Test specific components
uv run pytest tests/test_intelligent_chunking.py
uv run pytest tests/test_code_parser_service.py
uv run pytest tests/test_embedding_service.py

# End-to-end workflow tests
uv run pytest tests/test_end_to_end_workflow.py

# Performance benchmarks
uv run pytest tests/test_performance_benchmarks.py
```

### Code Quality Standards
- **Ruff**: Linting with E, F, W, I, N, UP, YTT, BLE, B, A, C4, T20 rules
- **Black**: Code formatting with 140 character line length
- **Type Hints**: Required for all new code
- **Error Handling**: Comprehensive error recovery and reporting

### Development Workflow
1. **Setup**: `uv sync` to install dependencies
2. **Test**: Run relevant test suite before changes
3. **Code**: Follow existing patterns and conventions
4. **Validate**: Run tests and quality checks
5. **Document**: Update CLAUDE.md if architecture changes

### File Processing Conventions
- **Supported Languages**: Python, JavaScript/TypeScript, Go, Rust, Java, C++
- **Configuration Files**: JSON/YAML with object-level chunking
- **Documentation**: Markdown with header-based chunking
- **Error Tolerance**: Graceful handling of syntax errors with smart fallback

### Integration Points
- **Tree-sitter**: For syntax-aware parsing and intelligent chunking
- **Qdrant**: Vector database for semantic search capabilities
- **Ollama**: Local LLM for embedding generation
- **FastMCP**: Model-Context-Provider integration framework
