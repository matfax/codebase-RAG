# Codebase RAG MCP Server

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG) Model-Controller-Provider (MCP) server** designed to assist AI agents and junior developers in understanding and navigating codebases with **function-level precision**. It leverages Qdrant as its vector database for storing codebase embeddings and integrates with Ollama for generating these embeddings. The server provides advanced MCP tools for indexing local directories and querying indexed codebases using natural language with **multi-modal retrieval capabilities**.

🆕 **Multi-Modal Retrieval System**: Enhanced with LightRAG-inspired multi-modal retrieval, intelligent auto-configuration, comprehensive performance monitoring, and robust error handling while maintaining full backward compatibility.

## Features

#### Multi-Modal Retrieval System
- **Local Mode**: Deep entity-focused retrieval using low-level keywords
- **Global Mode**: Broad relationship-focused retrieval using high-level keywords
- **Hybrid Mode**: Combined local+global with balanced context
- **Mix Mode**: Intelligent automatic mode selection based on query analysis
- **Query Analysis**: Automatic query classification and mode recommendation

#### Auto-Configuration & Performance
- **Intelligent Auto-Configuration**: Automatic parameter optimization based on system capabilities
- **15-Second Response Guarantee**: All MCP tools respond within 15 seconds with timeout handling
- **Performance Monitoring**: Real-time performance tracking and optimization
- **Graceful Degradation**: Automatic service degradation and recovery for reliability

### Core Functionality
- **Codebase Indexing**: Index local project directories with intelligent syntax-aware chunking
- **Function-Level Precision**: Search and analyze code at the function, class, and method level
- **Automatic Project Analysis**: Intelligently identifies project types and relevant source code files, respecting `.gitignore` rules
- **Flexible Embedding Models**: Supports various embedding models available through Ollama
- **Metal Performance Shaders (MPS) Acceleration**: Automatically utilizes MPS on macOS for faster embedding generation
- **Natural Language Querying**: Ask questions about your codebase in natural language with multi-modal retrieval
- **Backward Compatibility**: All existing MCP tool interfaces remain unchanged

## Getting Started

This guide will walk you through setting up the Codebase RAG MCP Server.

### 1. Prerequisites

Ensure you have the following installed:

- **Python 3.10+**
- **uv**: A fast Python package manager. Install it via `pip`:
  ```bash
  pip install uv
  ```
- **Docker**: For running the Qdrant vector database.
- **Ollama**: For running local language and embedding models. Download it from [ollama.com](https://ollama.com/).

### 2. Environment Setup

#### Running Dependencies
1.  **Start Qdrant**: Open a terminal and run the following Docker command to start the Qdrant database:
    ```bash
    docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
    ```
    This will store database data in a `qdrant_data` folder in your current directory.

2.  **Start Ollama & Pull Model**:
    - Launch the Ollama application.
    - Pull the default embedding model required for the server:
      ```bash
      ollama pull nomic-embed-text
      ```

### 3. Installation and Configuration

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd codebase-rag-mcp-server # Or your project directory name
    ```

2.  **Create Virtual Environment & Install Packages**:
    Use `uv` to create a virtual environment and install all dependencies from the lock file.
    ```bash
    uv sync
    ```
    This command creates a `.venv` folder in your project directory.

3.  **Configure Environment Variables**:
    Copy the example `.env` file and customize it if needed.
    ```bash
    cp .env.example .env
    ```
    The default settings should work for a local setup. For detailed configuration options, see our [Detailed Setup Guide](docs/SETUP.md).

## MCP Integration

**IMPORTANT**: You must complete MCP integration before you can index or query your codebase through AI tools. The MCP server provides the interface for AI agents to interact with your code.

### Benefits of MCP Integration
- **Seamless AI Assistance**: Your AI assistant can understand your codebase at function-level precision
- **Natural Language Queries**: Ask questions about your code in plain English
- **Multi-Modal Retrieval**: Automatically optimized search strategies for different query types
- **Real-Time Analysis**: Live codebase exploration and analysis during development

### Claude Code Integration

Claude Code is Anthropic's official CLI that provides excellent MCP support. Here's how to register this server:

1. **Complete Environment Setup**: Ensure you've completed all setup steps above (dependencies, virtual environment, configuration).

2. **Register the MCP Server**: Use Claude Code's built-in MCP registration:
   ```bash
   claude mcp add codebase-rag-mcp -- /path/to/your/project/.venv/bin/python /path/to/your/project/src/run_mcp.py --cwd /path/to/your/project/
   ```

   Replace `/path/to/your/project/` with your actual project path.

3. **Verify Registration**: Check that the server is registered:
   ```bash
   claude mcp list
   ```

### Cursor IDE Integration

Cursor IDE supports MCP servers through its extensions system:

1. **Install MCP Extension**: Search for "MCP" in the Cursor extensions marketplace and install the official MCP extension.

2. **Configure the Server**: Open Cursor settings (⌘/Ctrl + ,) and navigate to Extensions > MCP.

3. **Add Server Configuration**: Use the direct Python path format:
   ```json
   {
     "name": "codebase-rag",
     "command": "/path/to/your/project/.venv/bin/python",
     "args": ["/path/to/your/project/src/run_mcp.py"],
     "cwd": "/path/to/your/project/",
     "env": {}
   }
   ```

   Replace paths with your actual project location.

4. **Restart Cursor**: Restart the IDE to load the new MCP server.

5. **Verify Integration**: Check the MCP panel in Cursor to see available tools from the codebase-rag server.

### Gemini CLI Integration

For Google's Gemini CLI with MCP support:

1. **Install Gemini CLI**: Follow Google's official installation guide for the Gemini CLI.

2. **Configure MCP Server**: Add the server using the direct Python path:
   ```bash
   gemini mcp add codebase-rag -- /path/to/your/project/.venv/bin/python /path/to/your/project/src/run_mcp.py --cwd /path/to/your/project/
   ```

3. **Test Connection**:
   ```bash
   gemini mcp list  # Should show codebase-rag server
   ```

### Generic MCP Integration

For other MCP-compatible tools:

1. **Server Command**: Direct Python execution:
   ```bash
   /path/to/your/project/.venv/bin/python /path/to/your/project/src/run_mcp.py
   ```

2. **Standard MCP Configuration**: Most MCP clients accept this format:
   ```json
   {
     "name": "codebase-rag",
     "command": "/path/to/your/project/.venv/bin/python",
     "args": ["/path/to/your/project/src/run_mcp.py"],
     "cwd": "/path/to/your/project/"
   }
   ```

### Troubleshooting Integration Issues

**Connection Problems**:
- Ensure all dependencies (Qdrant, Ollama) are running
- Check that the virtual environment is properly set up with `uv sync`
- Verify the server starts manually with `uv run python src/run_mcp.py`

**Tool Not Available**:
- Restart your IDE/CLI after adding the MCP server
- Check the MCP server logs for errors
- Ensure the `cwd` path is correct and accessible

**Performance Issues**:
- Increase timeout settings in your IDE if indexing large codebases
- Monitor system resources (CPU, memory) during indexing
- Consider adjusting batch sizes in the `.env` file


## Quick Start: Using Your MCP-Enabled Environment

Now that you've registered the MCP server with your AI development environment, you can start indexing and querying your codebase.

### 1. Start Your AI Environment

Start your registered AI tool:
```bash
# For Claude Code
claude

# For Gemini CLI
gemini

# For Cursor IDE
# Just open the application
```

### 2. Index Your Codebase

Before querying, you need to index your codebase using the MCP tools. In your AI environment, you can now use natural language:

**For Claude Code or Gemini CLI:**
```
Please use the codebase-rag MCP tools to index the current directory with clear_existing mode.
```

**Alternative direct method** (activate virtual environment first):
```bash
source .venv/bin/activate
python manual_indexing.py -d "." -m clear_existing
```

### 3. Query Your Codebase

After indexing, you can explore your project using natural language through your AI assistant:

```
Please search the codebase for functions related to "embedding generation" and explain how they work.
```

```
What are the main components of this project and how do they interact?
```

```
Find all the MCP tools available and explain what each one does.
```

The AI assistant will use the MCP tools automatically to understand your codebase and provide detailed, accurate answers.

### Documentation
- **[MCP Tools Reference](docs/MCP_TOOLS.md)**: Complete guide to all MCP tools and Wave 7.0 enhancements
- **[Features](docs/FEATURES.md)**: Detailed feature documentation including intelligent code chunking
- **[Advanced Topics](docs/ADVANCED.md)**: Project structure, testing, and real-world examples

## Advanced Usage

### Manual Indexing Options

The manual indexing script requires specific parameters and provides the following options:

#### Required Parameters
All manual indexing operations require both directory and mode parameters:

```bash
# Basic syntax
python manual_indexing.py -d DIRECTORY -m MODE

# Clear existing data and reindex (recommended for fresh starts)
python manual_indexing.py -d "." -m clear_existing

# Incremental indexing (only index changed files)
python manual_indexing.py -d "." -m incremental

# Index specific directory
python manual_indexing.py -d "/path/to/your/project" -m clear_existing
```

#### Optional Parameters
```bash
# Verbose output for debugging
python manual_indexing.py -d "." -m clear_existing -v

# Skip confirmation prompts
python manual_indexing.py -d "." -m clear_existing --no-confirm

# Specify custom error report directory
python manual_indexing.py -d "." -m clear_existing --error-report-dir "./error_reports"

# Combined example with all options
python manual_indexing.py -d "." -m clear_existing -v --no-confirm --error-report-dir "./logs"
```

#### Performance Tuning for Large Codebases
```bash
# For very large codebases, you can adjust environment variables:
export INDEXING_CONCURRENCY=4  # Reduce concurrent workers
export EMBEDDING_BATCH_SIZE=50  # Reduce batch size
export QDRANT_BATCH_SIZE=100   # Adjust vector database batch size

# Then run indexing
python manual_indexing.py -d "." -m clear_existing
```

### Server Management

#### Running the MCP Server Manually
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the server directly
python src/run_mcp.py

# Run with debug logging
MCP_DEBUG_LEVEL=DEBUG python src/run_mcp.py

# Run in production mode (optimized for performance)
MCP_ENV=production python src/run_mcp.py
```

### Performance Optimization

#### System Requirements for Large Codebases
- **Memory**: Minimum 8GB RAM, 16GB+ recommended for large projects
- **Storage**: SSD recommended for vector database performance
- **CPU**: Multi-core processor for parallel indexing

#### Configuration Tuning
Edit your `.env` file for optimal performance:

```bash
# Vector database settings
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_BATCH_SIZE=200

# Embedding settings
OLLAMA_HOST=http://localhost:11434
EMBEDDING_BATCH_SIZE=100

# Performance settings
INDEXING_CONCURRENCY=8          # Adjust based on CPU cores
MEMORY_WARNING_THRESHOLD_MB=4096 # Memory usage alert threshold
CACHE_DEBUG_MODE=false          # Disable for production

# MCP settings
MCP_ENV=production              # Use production optimizations
MCP_DEBUG_LEVEL=INFO           # Reduce logging verbosity
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Support

If you encounter any issues or have questions, please open an issue on GitHub.
