# Codebase RAG MCP Server

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) Model-Controller-Provider (MCP) server designed to assist AI agents and junior developers in understanding and navigating codebases. It leverages Qdrant as its vector database for storing codebase embeddings and integrates with Ollama for generating these embeddings. The server provides API endpoints for indexing local directories or GitHub repositories and for querying the indexed codebases using natural language.

## Features

- **Codebase Indexing**: Index local project directories or remote GitHub repositories.
- **Automatic Project Analysis**: Intelligently identifies project types and relevant source code files, respecting `.gitignore` rules.
- **Flexible Embedding Models**: Supports using various embedding models available through Ollama.
- **Metal Performance Shaders (MPS) Acceleration**: Automatically utilizes MPS on macOS for faster embedding generation.
- **Natural Language Querying**: Ask questions about your codebase in natural language and retrieve relevant code snippets.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+**: The project is developed with Python.
- **Poetry**: Used for dependency management. If you don't have it, you can install it via `pipx install poetry` or `pip install poetry` (preferably in a virtual environment).
- **Docker (Recommended for Qdrant)**: Qdrant is best run as a Docker container.
- **Ollama**: For running local language models and embedding models. Download from [ollama.com](https://ollama.com/).

## Configuration

This project uses environment variables for configuration, loaded from a `.env` file in the project root. You can create a `.env` file to customize settings such as the Ollama server URL and the default embedding model.

Example `.env` file:

```
OLLAMA_HOST=http://localhost:11434
OLLAMA_DEFAULT_EMBEDDING_MODEL=nomic-embed-text
```

- `OLLAMA_HOST`: The URL of your Ollama server. Defaults to `http://localhost:11434`.
- `OLLAMA_DEFAULT_EMBEDDING_MODEL`: The default embedding model to use if not specified in API requests. Defaults to `nomic-embed-text`.

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

Index files in a directory or Git repository.

-   **Parameters**:
    ```json
    {
      "directory": "string",        // Path to local directory (default: ".")
      "patterns": ["string"],       // File patterns to include (optional)
      "recursive": "boolean",       // Search recursively (default: true)
      "clear_existing": "boolean"   // Clear existing index (default: false)
    }
    ```
-   **Returns**: Indexing results with file count and collections used

### 3. `search`

Search indexed content using natural language queries.

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
-   **Returns**: Search results with relevant code snippets and metadata

## Usage Examples

### Using with Python (AsyncIO)

```python
import asyncio
from main import app

async def example():
    # Index current directory
    result = await app.call_tool("index_directory", {"directory": "."})
    print(result)
    
    # Search for specific functionality
    result = await app.call_tool("search", {"query": "authentication logic"})
    print(result)

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
   - `@codebase-rag-mcp:index_directory` - Index a directory
   - `@codebase-rag-mcp:search` - Search indexed content
   - `@codebase-rag-mcp:health_check` - Check server status

### Other MCP Clients

For other MCP-compatible clients, configure them to run:
```bash
/path/to/your/project/.venv/bin/python /path/to/your/project/src/run_mcp.py
```

Refer to your specific MCP client documentation for configuration details.

## Project Structure

-   `src/`: Contains the main application source code.
    -   `api/`: Defines FastAPI endpoints.
    -   `services/`: Contains core business logic and service integrations (Qdrant, Ollama, Project Analysis, Indexing).
    -   `main.py`: The main FastAPI application entry point.
-   `tests/`: Contains unit and integration tests for the project.
-   `tasks/`: Stores Product Requirements Documents (PRDs) and task lists generated during development.
-   `pyproject.toml`: Poetry configuration file for project metadata and dependencies.
-   `poetry.lock`: Poetry lock file, ensuring reproducible installations.
-   `.venv/`: Python virtual environment (created during setup).