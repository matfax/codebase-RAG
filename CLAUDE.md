# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup
- `.venv/bin/poetry install` - Install dependencies
- `.venv/bin/poetry add "mcp[cli]"` - Add MCP CLI support
- `.venv/bin/poetry lock` - Update lock file if pyproject.toml changes

### Running the MCP Server
- `.venv/bin/python src/run_mcp.py` - Start the MCP server in stdio mode
- `./register_mcp.sh` - Register with Claude Code (creates configuration)
- Server communicates via stdio, not HTTP
- Logs are output to stderr, JSON-RPC communication via stdin/stdout

### Testing
- `.venv/bin/pytest tests/` - Run all tests
- `.venv/bin/pytest tests/test_specific.py` - Run specific test file

### External Dependencies
- **Qdrant**: `docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant`
- **Ollama**: Must be running locally with embedding models (e.g., `ollama pull nomic-embed-text`)

## Architecture Overview

This is a **Codebase RAG (Retrieval-Augmented Generation) MCP Server** that enables AI agents to understand and query codebases using natural language.

### Core Components

1. **MCP Server** (`src/main.py`, `src/mcp_tools.py`)
   - FastMCP-based server exposing indexing and search tools
   - Registers MCP tools for codebase operations
   - Uses environment variables from `.env` file

2. **Services Layer** (`src/services/`)
   - `indexing_service.py`: Processes codebases for embedding generation
   - `qdrant_service.py`: Vector database operations and collection management
   - `embedding_service.py`: Ollama integration for embeddings

3. **Data Flow**
   - Index: Source code → Chunking → Embedding → Qdrant storage
   - Query: Natural language → Embedding → Vector search → Ranked results

### Key Architectural Decisions

- **Vector Database**: Qdrant for storing and searching embeddings
- **Embeddings**: Ollama with configurable models (default: `nomic-embed-text`)
- **Chunking Strategy**: Currently whole-file chunks, metadata includes language detection
- **Collection Organization**: Separate collections for code, config, and documentation
- **Project Context**: Automatically detects project boundaries using `.git`, `pyproject.toml` markers

### Configuration

Environment variables (`.env` file):
- `OLLAMA_HOST`: Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_DEFAULT_EMBEDDING_MODEL`: Default embedding model (default: `nomic-embed-text`)
- `QDRANT_HOST`: Qdrant host (default: `localhost`)
- `QDRANT_PORT`: Qdrant port (default: `6333`)

### MCP Tools Available

- `index_directory()`: Index files in a directory with automatic project detection
- `search()`: Search indexed content with project-scoped or cross-project queries
- `health_check()`: Verify server status

### File Organization Patterns

- Uses `.ragignore` files for excluding directories/files from indexing
- Automatically categorizes files by type (code/config/documentation)
- Generates deterministic collection names based on project context