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
- `python test_full_functionality.py` - Test basic MCP functionality
- `python test_mcp_stdio.py` - Test stdio communication
- `python demo_mcp_usage.py` - Run usage demo

### Manual Indexing Tool
- `python manual_indexing.py -d /path/to/repo -m clear_existing` - Full indexing
- `python manual_indexing.py -d /path/to/repo -m incremental` - Incremental indexing
- `python manual_indexing.py -d /path/to/repo -m incremental --verbose` - Verbose output
- `python manual_indexing.py -d /path/to/repo -m clear_existing --no-confirm` - Skip prompts

### Performance Testing and Validation
- Manual tool provides pre-indexing analysis with file count and time estimates
- Use `--verbose` flag for detailed logging and performance metrics
- Monitor memory usage and processing rates in logs

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
   - `indexing_service.py`: Processes codebases with parallel processing and batch optimization
   - `qdrant_service.py`: Vector database operations with streaming and retry logic
   - `embedding_service.py`: Ollama integration with batch embedding generation
   - `project_analysis_service.py`: Repository analysis and file filtering
   - `file_metadata_service.py`: File change tracking for incremental indexing
   - `change_detector_service.py`: Change detection and selective reprocessing logic

3. **Models Layer** (`src/models/`)
   - `file_metadata.py`: File metadata tracking with content hashing and change detection

4. **Utilities Layer** (`src/utils/`)
   - `performance_monitor.py`: Progress tracking, ETA estimation, and memory monitoring
   - `stage_logger.py`: Detailed timing and performance logging

5. **Manual Indexing Tool** (`manual_indexing.py`)
   - Standalone script for heavy indexing operations
   - Command-line interface with validation and progress reporting
   - Integration with all core services for consistent behavior

3. **Data Flow**
   
   **Full Indexing Flow:**
   ```
   Source Code → File Discovery → Parallel Processing → Chunking → 
   Batch Embedding → Streaming DB Storage → Metadata Storage
   ```
   
   **Incremental Indexing Flow:**
   ```
   File Discovery → Change Detection → Selective Processing → 
   Changed Files Only → Embedding → DB Updates → Metadata Updates
   ```
   
   **Query Flow:**
   ```
   Natural Language → Embedding → Vector Search → 
   Context Enhancement → Ranked Results
   ```
   
   **Manual Tool Flow:**
   ```
   CLI Input → Validation → Pre-analysis → Progress Tracking → 
   Core Processing → Status Reporting
   ```

### Key Architectural Decisions

- **Vector Database**: Qdrant for storing and searching embeddings
- **Embeddings**: Ollama with configurable models (default: `nomic-embed-text`)
- **Chunking Strategy**: Currently whole-file chunks, metadata includes language detection
- **Collection Organization**: Separate collections for code, config, documentation, and file metadata
- **Project Context**: Automatically detects project boundaries using `.git`, `pyproject.toml` markers
- **Incremental Processing**: File change detection using modification times and content hashes
- **Parallel Processing**: Multi-threaded file processing with configurable concurrency
- **Memory Management**: Streaming operations with memory monitoring and cleanup
- **Batch Optimization**: Batched operations for embeddings and database insertions

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

### MCP Tools Available

#### `index_directory(directory, patterns, recursive, clear_existing, incremental)`
- **Smart Indexing**: Automatically detects existing data and provides recommendations
- **Incremental Mode**: `incremental=true` processes only changed files
- **Time Estimation**: Provides processing time estimates and manual tool recommendations
- **Batch Processing**: Optimized for large codebases with parallel processing
- **Progress Tracking**: Real-time progress reporting with ETA

#### `search(query, n_results, cross_project, search_mode, include_context, context_chunks)`
- **Natural Language**: Search indexed content using natural language queries
- **Project Scoping**: Search within current project or across all projects
- **Context Enhancement**: Include surrounding code context in results
- **Multiple Search Modes**: Hybrid, semantic, and keyword search options

#### `health_check()`
- **Server Status**: Verify MCP server health and connectivity
- **Dependency Check**: Validate Qdrant and Ollama service availability
- **Performance Metrics**: Basic system status and response times

#### Additional MCP Tools
- `analyze_repository_tool()`: Repository structure analysis and statistics
- `get_file_filtering_stats_tool()`: File filtering analysis for optimization
- `check_index_status()`: Check existing index status with recommendations
- `get_indexing_progress()`: Real-time progress monitoring for ongoing operations

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

**Content Collections** (store actual embeddings):
- `project_{name}_code`: Source code files (.py, .js, .java, etc.)
- `project_{name}_config`: Configuration files (.json, .yaml, .toml, etc.)
- `project_{name}_documentation`: Documentation files (.md, .rst, .txt, etc.)

**Metadata Collection** (tracks file states):
- `project_{name}_file_metadata`: File change tracking for incremental indexing
  - Stores: file_path, mtime, content_hash, file_size, indexed_at
  - Used for: change detection, incremental processing, progress tracking

### Performance Optimization Features

- **Parallel Processing**: Multi-threaded file reading and processing
- **Batch Operations**: Grouped embedding generation and database insertions
- **Streaming Architecture**: Memory-efficient processing for large codebases
- **Progress Monitoring**: Real-time progress tracking with ETA calculation
- **Memory Management**: Automatic cleanup and garbage collection
- **Adaptive Batching**: Dynamic batch size adjustment based on memory usage
- **Retry Logic**: Exponential backoff for failed operations
- **Early Filtering**: Skip excluded directories before processing