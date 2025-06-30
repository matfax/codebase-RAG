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

- **Python 3.9+**: The project is developed with Python.
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
    *Note: If `poetry install` fails due to `pyproject.toml` changes, run `.venv/bin/poetry lock` first, then `.venv/bin/poetry install` again.*

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

## Running the Server

Once Qdrant and Ollama are running, you can start the FastAPI server:

```bash
.venv/bin/uvicorn src.main:app --reload
```
The server will typically run on `http://127.0.0.1:8000`.

## API Endpoints

The server exposes the following API endpoints:

### 1. Index Codebase (`POST /index`)

Indexes a local directory or a Git repository into a Qdrant collection.

-   **Request Body**:
    ```json
    {
      "source_path": "string",       // Path to local directory or Git repository URL
      "collection_name": "string",   // Name of the Qdrant collection to store embeddings
      "embedding_model": "string"    // Name of the Ollama embedding model to use (e.g., "nomic-embed-text")
    }
    ```
-   **Example (`curl`)**:
    ```bash
    curl -X POST "http://127.0.0.1:8000/index" \
         -H "Content-Type: application/json" \
         -d '{
           "source_path": "/path/to/your/local/repo",
           "collection_name": "my_project_code",
           "embedding_model": "nomic-embed-text"
         }'
    ```
    Or for a GitHub repository:
    ```bash
    curl -X POST "http://127.0.0.1:8000/index" \
         -H "Content-Type: application/json" \
         -d '{
           "source_path": "https://github.com/owner/repo.git",
           "collection_name": "github_project_code",
           "embedding_model": "nomic-embed-text"
         }'
    ```

### 2. Query Codebase (`POST /query`)

Queries an indexed codebase using a natural language question and retrieves relevant code snippets.

-   **Request Body**:
    ```json
    {
      "query_text": "string",        // The natural language question
      "collection_name": "string",   // Name of the Qdrant collection to query
      "embedding_model": "string",   // Name of the Ollama embedding model used for the query
      "limit": "integer"             // (Optional) Maximum number of results to return (default: 5)
    }
    ```
-   **Example (`curl`)**:
    ```bash
    curl -X POST "http://127.0.0.1:8000/query" \
         -H "Content-Type: application/json" \
         -d '{
           "query_text": "How to handle user authentication?",
           "collection_name": "my_project_code",
           "embedding_model": "nomic-embed-text",
           "limit": 3
         }'
    ```

## Running Tests

To run the unit and integration tests, use `pytest`:

```bash
.venv/bin/pytest tests/
```

## Registering as an MCP Server

This server can be registered as an MCP (Model-Controller-Provider) server with compatible AI agents or platforms. To do so, you typically need to provide the base URL of your running server (e.g., `http://127.0.0.1:8000`). The server exposes a manifest at its root endpoint (`/`) that describes its capabilities.

**General Integration Steps:**
1.  **Start the MCP Server**: Ensure the server is running and accessible via a URL (e.g., `http://127.0.0.1:8000`).
2.  **Configure in your AI Agent or IDE**: Navigate to the settings or configuration section of your AI agent, IDE, or platform where external tools or services can be added.
3.  **Provide the Base URL**: Input the base URL of your running MCP server (e.g., `http://127.0.0.1:8000`). Many tools will automatically discover the available endpoints and their functionalities by fetching the manifest from this URL.
4.  **Follow Specific Tool Documentation**: For detailed, tool-specific instructions, always refer to the official documentation of your AI agent, IDE (e.g., VS Code extensions for AI), or platform (e.g., Gemini, Claude Code). They will provide precise steps on how to integrate custom MCP servers or external APIs.

**Example for VS Code-like IDEs (Conceptual):**

Some IDEs or extensions might allow you to configure external MCP servers by specifying a command to run the server. Here's a conceptual example of how you might configure it, assuming your IDE has a setting for `mcpServers`:

```json
{
  "mcpServers": {
    "codebaseRAG": {
      "command": "${workspaceFolder}/.venv/bin/uvicorn",
      "args": [
        "src.main:app",
        "--reload",
        "--port",
        "8000" // Ensure this matches the port your IDE expects or can configure
      ],
      "baseUrl": "http://127.0.0.1:8000" // The URL where the MCP server will be accessible
    }
  }
}
```

*   **`command`**: The executable to run your server. Here, it points to the `uvicorn` executable within your project's virtual environment.
*   **`args`**: Arguments passed to the `uvicorn` command. `src.main:app` specifies your FastAPI application, `--reload` enables hot-reloading, and `--port` sets the port.
*   **`baseUrl`**: The base URL where the IDE can access your running MCP server. This should match the host and port configured in `args`.

**Specific Integration Examples:**

### Claude Code

Claude Code uses the `claude mcp add` command to register MCP servers. You would typically run this command in your terminal, providing the URL of your running server. For example:

```bash
claude mcp add codebase-rag-mcp http://127.0.0.1:8000
```

This registers your server with the name `codebase-rag-mcp`. You can then use resources exposed by your server in Claude Code prompts (e.g., `@codebase-rag-mcp:index` or `@codebase-rag-mcp:query`). Refer to the [Claude Code MCP documentation](https://docs.anthropic.com/zh-TW/docs/claude-code/mcp) for more details on managing MCP servers and using their resources.

### Gemini CLI (Code Assist)

For Gemini CLI (Code Assist), you would configure MCP servers in your `~/.gemini/settings.json` file. You would add an entry similar to this:

```json
{
  "mcpServers": {
    "codebaseRAG": {
      "url": "http://127.0.0.1:8000"
    }
  }
}
```

After adding this, restart your Gemini CLI or VS Code instance for the changes to take effect. You can then use the `/mcp` command in Gemini CLI to list configured MCP servers. Refer to the [Gemini CLI MCP documentation](https://cloud.google.com/gemini/docs/codeassist/use-agentic-chat-pair-programmer#configure-mcp-servers) for more details.

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
