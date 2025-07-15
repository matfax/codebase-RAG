"# Setup Guide for Codebase RAG MCP Server

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+**: The project is developed with Python.
- **uv**: Modern Python package manager for fast dependency management. Install from [astral.sh/uv](https://docs.astral.sh/uv/getting-started/installation/).
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

### Dynamic Tool Registration

You can control which MCP tools are registered based on the environment using the `MCP_ENV` variable in your `.env` file.

- `MCP_ENV=production`: Registers only core tools (e.g., indexing, search, basic project management) for optimal performance in production environments. This minimizes the number of tools to avoid overloading IDEs or affecting LLM efficiency.
- `MCP_ENV=development`: Registers all tools, including debugging, metrics, and advanced cache management tools, suitable for development and testing.

See `.env.example` for an example configuration. The tool registration happens dynamically in `src/tools/registry.py` based on this variable.

## Setup Steps

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd codebase-rag-mcp # Or your project directory name
    ```

2.  **Install dependencies and create virtual environment**:
    ```bash
    uv sync
    ```
    *This automatically creates a virtual environment and installs all dependencies from the lock file.*

3.  **Copy environment configuration**:
    ```bash
    cp .env.example .env
    # Edit .env file to customize settings as needed
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
    You can use any other embedding model available in Ollama, just ensure you specify its name correctly when making API calls." 