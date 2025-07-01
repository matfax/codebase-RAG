# Codebase RAG MCP Server

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) Model-Controller-Provider (MCP) server designed to assist AI agents and junior developers in understanding and navigating codebases. It leverages Qdrant as its vector database for storing codebase embeddings and integrates with Ollama for generating these embeddings. The server provides API endpoints for indexing local directories or GitHub repositories and for querying the indexed codebases using natural language.

## Features

### Core Functionality
- **Codebase Indexing**: Index local project directories or remote GitHub repositories
- **Automatic Project Analysis**: Intelligently identifies project types and relevant source code files, respecting `.gitignore` rules
- **Flexible Embedding Models**: Supports using various embedding models available through Ollama
- **Metal Performance Shaders (MPS) Acceleration**: Automatically utilizes MPS on macOS for faster embedding generation
- **Natural Language Querying**: Ask questions about your codebase in natural language and retrieve relevant code snippets

## Intelligent Code Chunking

This system implements state-of-the-art **syntax-aware code chunking** using Tree-sitter parsers to break down source code into meaningful, semantically coherent chunks instead of processing entire files as single units.

### Key Benefits

- **🎯 Precise Retrieval**: Find specific functions, classes, or methods instead of entire files
- **📈 Better Embeddings**: Each chunk represents a complete semantic unit, improving vector quality
- **🔍 Function-Level Search**: Query for specific functionality and get exact matches
- **📚 Rich Metadata**: Each chunk includes detailed information like function signatures, docstrings, and context

### Supported Languages

**Phase 1 (Fully Supported):**
- Python (.py) - Functions, classes, methods, constants
- JavaScript (.js, .jsx) - Functions, objects, modules
- TypeScript (.ts, .tsx) - Interfaces, types, classes, functions

**Phase 2 (Extended Support):**
- Go (.go) - Functions, structs, interfaces, methods
- Rust (.rs) - Functions, structs, impl blocks, traits
- Java (.java) - Classes, methods, interfaces
- C/C++ (.c, .cpp, .h) - Functions, structs, classes

**Structured Files:**
- JSON/YAML - Object-level chunking (e.g., separate chunks for `scripts`, `dependencies`)
- Markdown - Header-based hierarchical chunking
- Configuration files - Section-based parsing

### How It Works

1. **AST Parsing**: Uses Tree-sitter to generate syntax trees for each source file
2. **Semantic Extraction**: Identifies functions, classes, methods, and other semantic units
3. **Context Enhancement**: Preserves surrounding context (imports, docstrings, inline comments)
4. **Metadata Enrichment**: Extracts signatures, parameter types, access modifiers, and relationships
5. **Error Handling**: Gracefully handles syntax errors with smart fallback to whole-file processing

### Chunk Types

- `function` - Standalone functions with their complete implementation
- `class` - Class definitions including properties and documentation
- `method` - Individual methods within classes
- `interface` - TypeScript/Java interfaces and type definitions
- `constant` - Important constants and configuration objects
- `config_object` - Structured configuration sections

### Advanced Features
- **🎯 Intelligent Code Chunking**: Function-level and class-level intelligent chunking using Tree-sitter for precise code understanding and retrieval
- **🚀 Incremental Indexing**: Only process changed files for dramatically faster re-indexing (80%+ time savings)
- **🔧 Manual Indexing Tool**: Standalone script for heavy indexing operations that don't block conversational workflows
- **⚡ Large Codebase Optimization**: Parallel processing, batch operations, and streaming for handling 10,000+ file repositories
- **🧠 Intelligent Recommendations**: Automatic detection of existing indexes with smart time estimates and workflow suggestions
- **📊 Progress Tracking**: Real-time progress monitoring with ETA estimation and memory usage tracking
- **🗂️ Smart Collection Management**: Automatic categorization into code, config, documentation, and metadata collections
- **🌐 Multi-Language Support**: Support for 10+ programming languages with syntax-aware parsing
- **🛡️ Error Tolerance**: Graceful handling of syntax errors with smart fallback mechanisms

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+**: The project is developed with Python.
- **Poetry**: Used for dependency management. If you don't have it, you can install it via `pipx install poetry` or `pip install poetry` (preferably in a virtual environment).
- **Docker (Recommended for Qdrant)**: Qdrant is best run as a Docker container.
- **Ollama**: For running local language models and embedding models. Download from [ollama.com](https://ollama.com/).
- **Tree-sitter Language Parsers**: Automatically installed with the project dependencies for intelligent code chunking support.

## Configuration

This project uses environment variables for configuration, loaded from a `.env` file in the project root.

Example `.env` file:

```bash
# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_DEFAULT_EMBEDDING_MODEL=nomic-embed-text

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Performance Tuning
INDEXING_CONCURRENCY=4                    # Parallel file processing workers
INDEXING_BATCH_SIZE=20                    # Files processed per batch
EMBEDDING_BATCH_SIZE=10                   # Texts per embedding API call
QDRANT_BATCH_SIZE=500                     # Points per database insertion
MEMORY_WARNING_THRESHOLD_MB=1000          # Memory usage warning threshold
MAX_FILE_SIZE_MB=5                       # Skip files larger than this
MAX_DIRECTORY_DEPTH=20                   # Maximum directory traversal depth

# Logging
LOG_LEVEL=INFO                           # DEBUG, INFO, WARNING, ERROR
```

### Configuration Options

#### Basic Settings
- `OLLAMA_HOST`: Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_DEFAULT_EMBEDDING_MODEL`: Default embedding model (default: `nomic-embed-text`)
- `QDRANT_HOST`: Qdrant host (default: `localhost`)
- `QDRANT_PORT`: Qdrant port (default: `6333`)

#### Performance Settings
- `INDEXING_CONCURRENCY`: Number of parallel workers for file processing (default: `4`)
- `INDEXING_BATCH_SIZE`: Files processed in each batch (default: `20`)
- `EMBEDDING_BATCH_SIZE`: Texts sent to embedding API per call (default: `10`)
- `QDRANT_BATCH_SIZE`: Points inserted to database per batch (default: `500`)
- `MEMORY_WARNING_THRESHOLD_MB`: Memory usage threshold for warnings (default: `1000`)

#### File Processing Settings
- `MAX_FILE_SIZE_MB`: Skip files larger than this size (default: `5`)
- `MAX_DIRECTORY_DEPTH`: Maximum directory traversal depth (default: `20`)
- `FOLLOW_SYMLINKS`: Whether to follow symbolic links (default: `false`)

## Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd codebase-rag-mcp # Or your project directory name
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install project dependencies**:
    ```bash
    .venv/bin/poetry install
    ```
    *Note: If `poetry install` fails due to `pyproject.toml` changes, run `.venv/bin/poetry lock` first, then `.venv/bin/poetry install` again. Also, ensure you have Python 3.10 or higher installed and activated in your virtual environment.*

4.  **Add MCP to your project dependencies**:
    ```bash
    .venv/bin/poetry add "mcp[cli]"
    ```

## Running Qdrant

It's recommended to run Qdrant using Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```
This command will start Qdrant and map its default ports (6333 for gRPC, 6334 for HTTP) to your host. Data will be persisted in a `qdrant_data` directory in your current working directory.

## Running Ollama

1.  **Start the Ollama server**: Follow the instructions on [ollama.com](https://ollama.com/) to download and run Ollama.

2.  **Pull an embedding model**: This project uses `nomic-embed-text` by default for examples. You can pull it using:
    ```bash
    ollama pull nomic-embed-text
    ```
    You can use any other embedding model available in Ollama, just ensure you specify its name correctly when making API calls.

## Running the MCP Server

Once Qdrant and Ollama are running, you can start the MCP server:

```bash
.venv/bin/python src/run_mcp.py
```

The server runs in stdio mode and communicates via JSON-RPC. You will see startup logs like:
```
2025-06-30 11:35:31 - __main__ - INFO - Starting Codebase RAG MCP Server...
2025-06-30 11:35:31 - __main__ - INFO - Server name: codebase-rag-mcp
2025-06-30 11:35:31 - __main__ - INFO - Listening for JSON-RPC requests on stdin...
```

### Registering with Claude Code

To register this MCP server with Claude Code:

```bash
./register_mcp.sh
```

This will create a configuration file and provide instructions for registering the server.

## MCP Tools

The server provides the following MCP tools that can be used by AI assistants:

### 1. `health_check`

Check the health status of the MCP server.

-   **Parameters**: None
-   **Returns**: Server status information

### 2. `index_directory`

Index files in a directory or Git repository with intelligent code chunking and existing data detection.

-   **Parameters**:
    ```json
    {
      "directory": "string",        // Path to local directory (default: ".")
      "patterns": ["string"],       // File patterns to include (optional)
      "recursive": "boolean",       // Search recursively (default: true)
      "clear_existing": "boolean",  // Clear existing index (default: false)
      "incremental": "boolean",     // Use incremental indexing (default: false)
      "project_name": "string"      // Custom project name for collections (optional)
    }
    ```
-   **Returns**: Indexing results, time estimates, or recommendations for existing data
-   **Smart Behavior**: 
    - **Intelligent Chunking**: Automatically uses syntax-aware chunking for supported languages
    - Automatically detects existing indexed data
    - Provides time estimates and recommendations
    - Suggests manual tool for large operations (>5 minutes)
    - Supports incremental mode for changed files only
    - **Error Tolerance**: Gracefully handles syntax errors with fallback mechanisms

### 3. `search`

Search indexed content using natural language queries with function-level precision.

-   **Parameters**:
    ```json
    {
      "query": "string",           // Natural language query (required)
      "n_results": "integer",      // Number of results (default: 5)
      "cross_project": "boolean",  // Search across all projects (default: false)
      "search_mode": "string",     // Search mode (default: "hybrid")
      "include_context": "boolean", // Include surrounding context (default: true)
      "context_chunks": "integer"  // Number of context chunks (default: 1)
    }
    ```
-   **Returns**: Search results with relevant code snippets and rich metadata
-   **Enhanced Results**: 
    - **Function-Level Precision**: Returns specific functions, classes, or methods instead of entire files
    - **Rich Metadata**: Includes function signatures, docstrings, and breadcrumb navigation
    - **Context Enhancement**: Provides surrounding code context and related imports
    - **Syntax Highlighting**: Results include language detection and proper formatting

## Manual Indexing Tool

For large codebases or operations that might take several minutes, use the standalone manual indexing tool:

```bash
# Full indexing (clear existing data)
python manual_indexing.py -d /path/to/large/repo -m clear_existing

# Incremental indexing (only changed files)
python manual_indexing.py -d /path/to/repo -m incremental

# With verbose output
python manual_indexing.py -d /path/to/repo -m incremental --verbose

# Skip confirmation prompts
python manual_indexing.py -d /path/to/repo -m clear_existing --no-confirm
```

### Manual Tool Features
- **Pre-indexing Analysis**: Shows file count, size, and time estimates
- **Change Detection**: In incremental mode, shows exactly which files changed
- **Progress Reporting**: Real-time progress with ETA and memory usage
- **Error Handling**: Graceful handling of individual file failures
- **Dependency Validation**: Checks Qdrant and embedding service availability

### When to Use Manual Tool
- Large repositories (1000+ files)
- Operations estimated to take >5 minutes
- Heavy indexing that shouldn't block interactive workflows
- Batch processing scenarios

## Incremental Indexing

Incremental indexing dramatically reduces processing time by only handling changed files:

### How It Works
1. **File Metadata Tracking**: Stores modification times and content hashes
2. **Change Detection**: Compares current file state with stored metadata
3. **Selective Processing**: Only re-indexes files that have changed
4. **Smart Updates**: Handles file additions, modifications, and deletions

### Performance Benefits
- **80%+ Time Savings**: For typical development scenarios (5-50 changed files)
- **Memory Efficient**: Only processes changed content
- **Network Friendly**: Fewer embedding API calls

### Usage Examples

#### Via MCP Tool
```python
# Check for existing data and get recommendations
result = await app.call_tool("index_directory", {"directory": "/my/project"})
# If existing data found, you'll get recommendations including incremental update

# Perform incremental update
result = await app.call_tool("index_directory", {
    "directory": "/my/project",
    "incremental": True
})
```

#### Via Manual Tool
```bash
# First time indexing
python manual_indexing.py -d /my/project -m clear_existing

# Subsequent updates (much faster)
python manual_indexing.py -d /my/project -m incremental
```

## Usage Examples

### Using with Python (AsyncIO)

```python
import asyncio
from main import app

async def example():
    # Index current directory with intelligent chunking (default behavior)
    result = await app.call_tool("index_directory", {"directory": "."})
    print(result)
    
    # Perform incremental update (only processes changed files)
    result = await app.call_tool("index_directory", {
        "directory": ".", 
        "incremental": True
    })
    print(result)
    
    # Search for specific functions - now returns precise matches!
    result = await app.call_tool("search", {"query": "validateUser function"})
    print("Function-level results:", result)
    
    # Search for class methods
    result = await app.call_tool("search", {"query": "UserService class methods"})
    print("Class and method results:", result)
    
    # Search with enhanced context
    result = await app.call_tool("search", {
        "query": "authentication logic",
        "include_context": True,
        "context_chunks": 2
    })
    print("Results with surrounding code context:", result)

asyncio.run(example())
```

### Testing the Server

```bash
# Test basic functionality
.venv/bin/python test_full_functionality.py

# Test stdio communication
.venv/bin/python test_mcp_stdio.py

# Run demo
.venv/bin/python demo_mcp_usage.py

# Test manual indexing tool
python manual_indexing.py -d . -m clear_existing --no-confirm

# Test incremental indexing
python manual_indexing.py -d . -m incremental --no-confirm
```

## Performance and Best Practices

### For Large Codebases (1000+ files)
1. **Use Manual Tool**: Avoid blocking interactive workflows
2. **Tune Batch Sizes**: Adjust `INDEXING_BATCH_SIZE` and `EMBEDDING_BATCH_SIZE`
3. **Monitor Memory**: Set appropriate `MEMORY_WARNING_THRESHOLD_MB`
4. **Use Incremental Mode**: After initial indexing, always use incremental updates

### Memory Optimization
- The system automatically manages memory with garbage collection between batches
- Monitor memory warnings in logs and adjust batch sizes if needed
- Large files (>5MB) are automatically skipped

### Network Optimization
- Batch embedding generation reduces API calls
- Incremental indexing minimizes redundant processing
- Failed operations have automatic retry with exponential backoff

### File Filtering
Create a `.ragignore` file in your project root to exclude directories:
```
node_modules/
.git/
dist/
build/
__pycache__/
.venv/
*.pyc
*.log
```

### Intelligent Chunking Error Handling

The system includes sophisticated error handling for syntax errors and malformed code:

- **Syntax Error Tolerance**: Files with syntax errors are processed using smart error recovery
- **Graceful Fallback**: When intelligent chunking fails, the system automatically falls back to whole-file processing
- **Error Reporting**: Detailed error statistics are provided in manual indexing tool output
- **Partial Processing**: Correct code sections are preserved even when parts of the file have errors

Example error handling in action:
```bash
# Manual indexing with verbose error reporting
python manual_indexing.py -d /path/to/project --verbose
# Output includes syntax error statistics and affected files
```

## Running Tests

To run the unit and integration tests, use `pytest`:

```bash
.venv/bin/pytest tests/
```

## Integrating with AI Assistants

This MCP server can be integrated with various AI assistants and development tools.

### Claude Code Integration

The easiest way to register this server with Claude Code:

1. **Automatic Registration** (recommended):
   ```bash
   ./register_mcp.sh
   ```
   This script will create the necessary configuration and provide instructions.

2. **Manual Registration**:
   ```bash
   # Recommended: Use the wrapper script
   claude mcp add codebase-rag-mcp "$(pwd)/mcp_server"
   
   # Alternative: Direct Python execution
   claude mcp add codebase-rag-mcp "$(pwd)/.venv/bin/python" "$(pwd)/src/run_mcp.py"
   ```

3. **Usage in Claude Code**:
   Once registered, you can use the tools in your prompts:
   - `@codebase-rag-mcp:index_directory` - Index a directory (with intelligent recommendations)
   - `@codebase-rag-mcp:search` - Search indexed content
   - `@codebase-rag-mcp:health_check` - Check server status
   
   **Smart Indexing Workflow**:
   - First call to `index_directory` will detect existing data and provide recommendations
   - Choose from: use existing data, incremental update, full reindex, or manual tool
   - For large repos, the system will automatically recommend the manual indexing tool

### Other MCP Clients

For other MCP-compatible clients, configure them to run:
```bash
/path/to/your/project/.venv/bin/python /path/to/your/project/src/run_mcp.py
```

Refer to your specific MCP client documentation for configuration details.

## Project Structure

```
src/
├── main.py                    # FastMCP server entry point
├── mcp_tools.py              # MCP tool implementations
├── run_mcp.py               # MCP server startup script
├── services/                # Core business logic
│   ├── indexing_service.py        # Codebase processing and indexing orchestration
│   ├── code_parser_service.py     # Intelligent code chunking with Tree-sitter
│   ├── qdrant_service.py          # Vector database operations
│   ├── embedding_service.py       # Ollama integration
│   ├── project_analysis_service.py # Project structure analysis
│   ├── file_metadata_service.py   # File change tracking
│   └── change_detector_service.py # Incremental indexing logic
├── models/                  # Data models
│   ├── file_metadata.py          # File metadata tracking
│   ├── code_chunk.py             # Code chunk data structures for intelligent chunking
│   └── __init__.py
├── utils/                   # Utilities
│   ├── performance_monitor.py     # Progress and memory monitoring
│   ├── stage_logger.py           # Detailed logging utilities
│   └── __init__.py
└── api/                     # FastAPI endpoints (legacy)
    └── endpoints.py

manual_indexing.py           # Standalone manual indexing tool
tests/                       # Unit and integration tests
tasks/                       # PRD and task documentation
register_mcp.sh             # MCP registration script
pyproject.toml              # Poetry configuration
poetry.lock                 # Dependency lock file
.env                        # Environment configuration
```

### Key Components

#### Services Layer
- **IndexingService**: Orchestrates the entire indexing process with parallel processing
- **CodeParserService**: Implements intelligent code chunking using Tree-sitter parsers for syntax-aware code analysis
- **QdrantService**: Manages vector database operations with batch optimization
- **EmbeddingService**: Handles Ollama integration with batch embedding generation
- **ProjectAnalysisService**: Analyzes project structure and filters relevant files
- **FileMetadataService**: Tracks file states for incremental indexing
- **ChangeDetectorService**: Detects file changes for selective reprocessing

#### Models
- **FileMetadata**: Tracks file modification times, content hashes, and indexing state
- **CodeChunk**: Represents intelligent code chunks with rich metadata (function signatures, docstrings, syntax tree information)

#### Utilities
- **PerformanceMonitor**: Progress tracking, ETA estimation, and memory monitoring
- **StageLogger**: Detailed timing and performance logging for each processing stage

#### Collection Organization
The system creates the following Qdrant collections:
- `project_{name}_code`: Source code files
- `project_{name}_config`: Configuration files (JSON, YAML, etc.)
- `project_{name}_documentation`: Documentation files (Markdown, etc.)
- `project_{name}_file_metadata`: File change tracking for incremental indexing