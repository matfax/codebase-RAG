# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Essential Setup
```bash
# Activate virtual environment (required before any development)
source .venv/bin/activate  # or use uv venv to create if not exists

# Install dependencies
uv sync

# Run tests
uv run pytest tests/
uv run pytest src/tests/  # Unit tests in src directory
uv run pytest src/tests/test_specific_file.py  # Single test file
測試前請記得啟用 .venv

# Code quality
uv run ruff check .  # Linting
uv run ruff check --fix .  # Auto-fix issues
uv run black .  # Code formatting
uv run pre-commit run --all-files  # Pre-commit hooks
```

### Running the MCP Server
```bash
# Development mode - MCP server in stdio mode
./mcp_server

# Manual indexing for large codebases
python manual_indexing.py -d /path/to/repo -m clear_existing
python manual_indexing.py -d /path/to/repo -m incremental  # Only changed files
```

### Cache Services (Redis/Qdrant)
```bash
# Start cache services with Docker Compose
docker-compose -f docker-compose.cache.yml up -d

# Stop services
docker-compose -f docker-compose.cache.yml down
```

## Architecture Overview

This is a **Codebase RAG (Retrieval-Augmented Generation) MCP Server** that enables AI agents to understand and query codebases using natural language with **function-level precision** through intelligent syntax-aware code chunking and advanced **Graph RAG capabilities**.

### Core Architecture

The system operates on a **two-phase architecture**:

1. **Indexing Phase**: Parse source code using Tree-sitter → Extract semantic chunks (functions, classes, methods) → Generate embeddings → Store in Qdrant vector database
2. **Query Phase**: Process natural language queries → Semantic search → Graph relationship analysis → Return contextual results

### Key Components

#### MCP Tools Layer (`src/tools/`)
- **Core Tools**: `indexing/`, `project/`, `cache/` - Basic MCP operations
- **Graph RAG Tools**: `graph_rag/` - Advanced code relationship analysis
  - `structure_analysis.py` - Analyze code component relationships
  - `function_chain_analysis.py` - Trace execution flows between functions
  - `function_path_finding.py` - Find optimal paths between code components
  - `project_chain_analysis.py` - Project-wide architectural analysis
  - `pattern_identification.py` - Detect architectural patterns
  - `similar_implementations.py` - Cross-project similarity analysis

#### Services Layer (`src/services/`)
- **Parsing & Chunking**: `code_parser_service.py`, `chunking_strategies.py`, `ast_extraction_service.py`
- **Storage & Search**: `qdrant_service.py`, `embedding_service.py`, `hybrid_search_service.py`
- **Graph RAG**: `graph_rag_service.py`, `structure_relationship_builder.py`, `implementation_chain_service.py`
- **Performance**: `graph_performance_optimizer.py`, `graph_rag_cache_service.py`

#### Data Models (`src/models/`)
- **CodeChunk**: Enhanced with `breadcrumb` (hierarchical location), `imports_used` (dependencies), rich metadata
- **FileMetadata**: Change tracking for incremental indexing

### Intelligent Code Chunking System

Uses **Tree-sitter parsers** for syntax-aware parsing across 8+ languages:
- **Python**: Functions, classes, methods, async functions, decorators
- **JavaScript/TypeScript**: Functions, classes, React components, interfaces
- **Go/Rust/Java/C++**: Language-specific constructs

Each chunk includes:
- **Breadcrumb**: Hierarchical identifier (e.g., `module.class.method`)
- **Imports Used**: Dependency tracking for graph building
- **Rich Metadata**: Signatures, docstrings, access modifiers, type hints

### Graph RAG Architecture

**Two-Phase Design**:
1. **Indexing**: Extract breadcrumbs and import dependencies, store in vector DB
2. **Graph Building**: On-demand relationship graph construction from stored data

**Relationship Types**:
- **Import Dependencies**: Based on `imports_used` metadata
- **Hierarchical**: Parent-child relationships (class → method)
- **Function Calls**: Runtime execution relationships (planned feature)

**Caching Strategy**:
- **L1**: File-level parsing cache
- **L2**: Breadcrumb resolution cache
- **L3**: Complete relationship graphs
- **TTL**: Based on file modification times

### Collection Organization

**Qdrant Collections**:
- `project_{name}_code`: Source code chunks with embeddings
- `project_{name}_config`: Configuration files (JSON, YAML)
- `project_{name}_documentation`: Documentation (Markdown)
- `project_{name}_file_metadata`: Change tracking for incremental updates

### Async/Await Patterns

The search pipeline is fully async. Key patterns:
```python
# Common async issues to avoid:
# ❌ Missing await
embeddings = embedding_service.generate_embeddings([query])

# ✅ Proper await
embeddings = await embedding_service.generate_embeddings([query])

# ❌ Can't iterate coroutine
for chunk in parser.parse_file(path):

# ✅ Await first, then iterate
chunks = await parser.parse_file(path)
for chunk in chunks:
```

### Configuration

Environment variables (`.env` file):
- **QDRANT_HOST**, **QDRANT_PORT**: Vector database connection
- **OLLAMA_HOST**: Embedding service endpoint
- **REDIS_HOST**, **REDIS_PORT**: Cache configuration
- **ENABLE_PERFORMANCE_MONITORING**: Performance tracking toggle

### Graph RAG vs Standard RAG

**Standard RAG**: Query → Vector Search → Return chunks
**Graph RAG**: Query → Vector Search + Graph Traversal → Return chunks with relationships

Graph RAG enables:
- Understanding component dependencies and impact analysis
- Tracing function execution flows across files
- Architectural pattern recognition
- Cross-project similarity analysis
