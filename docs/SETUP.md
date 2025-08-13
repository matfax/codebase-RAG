# Detailed Setup and Configuration

This guide provides detailed configuration options for the Codebase RAG MCP Server. For a basic setup and installation guide, please refer to the [Getting Started](../README.md#getting-started) section in the main `README.md` file.

## Configuration

This project uses environment variables for configuration, loaded from a `.env` file in the project root. You can copy the `.env.example` file to `.env` and modify the values as needed.

### Configuration Options

Below is a comprehensive list of all available environment variables.

#### Basic Settings
- `OLLAMA_HOST`: URL of the Ollama server. (Default: `http://localhost:11434`)
- `OLLAMA_DEFAULT_EMBEDDING_MODEL`: The default embedding model to use for indexing and querying. (Default: `nomic-embed-text`)
- `QDRANT_HOST`: Hostname or IP address of the Qdrant database. (Default: `localhost`)
- `QDRANT_PORT`: Port number for the Qdrant gRPC interface. (Default: `6333`)

#### Performance Tuning
- `INDEXING_CONCURRENCY`: Number of parallel worker processes for file processing during indexing. (Default: `4`)
- `INDEXING_BATCH_SIZE`: Number of files to process in a single batch. (Default: `20`)
- `EMBEDDING_BATCH_SIZE`: Number of text chunks to send to the embedding model API in a single call. (Default: `10`)
- `QDRANT_BATCH_SIZE`: Number of points (embeddings) to insert into the Qdrant database in a single batch. (Default: `500`)
- `MEMORY_WARNING_THRESHOLD_MB`: Threshold in megabytes for triggering a memory usage warning. (Default: `1000`)

#### File Processing
- `MAX_FILE_SIZE_MB`: The maximum size in megabytes for a file to be considered for indexing. Files larger than this will be skipped. (Default: `5`)
- `MAX_DIRECTORY_DEPTH`: The maximum depth to traverse into subdirectories when searching for files. (Default: `20`)
- `FOLLOW_SYMLINKS`: Set to `true` to follow symbolic links when discovering files. (Default: `false`)

#### Logging
- `LOG_LEVEL`: The logging level for the application. Can be `DEBUG`, `INFO`, `WARNING`, or `ERROR`. (Default: `INFO`)

### Dynamic Tool Registration

You can control which MCP tools are registered based on the environment using the `MCP_ENV` variable in your `.env` file. This is useful for tailoring the available tools for different environments like development and production.

- `MCP_ENV=production`: Registers only a minimal set of core tools (e.g., indexing, search, basic project management). This is recommended for production environments to optimize performance and reduce clutter in IDEs.
- `MCP_ENV=development`: Registers all available tools, including those for debugging, performance metrics, and advanced cache management. This is ideal for local development and testing.

The dynamic registration logic is handled in `src/tools/registry.py`.
