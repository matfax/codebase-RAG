# Codebase RAG MCP Server

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG) Model-Controller-Provider (MCP) server** designed to assist AI agents and junior developers in understanding and navigating codebases with **function-level precision**. It leverages Qdrant as its vector database for storing codebase embeddings and integrates with Ollama for generating these embeddings. The server provides advanced MCP tools for indexing local directories and querying indexed codebases using natural language with **multi-modal retrieval capabilities**.

🆕 **Wave 7.0**: Enhanced with LightRAG-inspired multi-modal retrieval, intelligent auto-configuration, comprehensive performance monitoring, and robust error handling while maintaining full backward compatibility.

## Features

### 🆕 Wave 7.0 Enhanced Features

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

#### Enhanced Graph RAG Capabilities
- **Increased Capacity**: Up to 400% improvement in analysis limits
- **Structure Analysis**: Advanced code component relationship analysis
- **Pattern Recognition**: Architectural pattern identification with enhanced capacity
- **Cross-Project Analysis**: Find similar implementations across multiple projects
- **Function Chain Tracing**: Comprehensive execution flow analysis

### Core Functionality
- **Codebase Indexing**: Index local project directories with intelligent syntax-aware chunking
- **Function-Level Precision**: Search and analyze code at the function, class, and method level
- **Automatic Project Analysis**: Intelligently identifies project types and relevant source code files, respecting `.gitignore` rules
- **Flexible Embedding Models**: Supports various embedding models available through Ollama
- **Metal Performance Shaders (MPS) Acceleration**: Automatically utilizes MPS on macOS for faster embedding generation
- **Natural Language Querying**: Ask questions about your codebase in natural language with multi-modal retrieval
- **Graph RAG Architecture**: Advanced code relationship analysis with on-demand graph building
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

### 4. Quick Start: Indexing and Querying

Once the setup is complete, you can start using the MCP server.

1.  **Activate the Virtual Environment**:
    ```bash
    source .venv/bin/activate
    ```

2.  **Run Manual Indexing**:
    Before querying, you need to index your codebase. The following command indexes the current directory (`.`) and clears any previously indexed data to ensure freshness.
    ```bash
    python manual_indexing.py -d "." -m clear_existing
    ```

3.  **Query Your Codebase**:
    After indexing, you can use registered MCP tools to explore your project. For example, if you are using an environment like Claude Code that supports custom prompts, you can run a command like this:
    ```
    /codebase-rag-mcp:explore_project (MCP) ., Graph RAG, detailed
    ```
    This command explores the codebase (`.`) for information related to "Graph RAG" with a "detailed" level of output, allowing you or an AI agent to quickly understand the project's structure and features.

### Documentation
- **[MCP Tools Reference](docs/MCP_TOOLS.md)**: Complete guide to all MCP tools and Wave 7.0 enhancements
- **[Wave 7.0 Enhancements](docs/WAVE_7_0_ENHANCEMENTS.md)**: Comprehensive Wave 7.0 feature guide
- **[Features](docs/FEATURES.md)**: Detailed feature documentation including intelligent code chunking
- **[Advanced Topics](docs/ADVANCED.md)**: Project structure, testing, and real-world examples
- **[Graph RAG Architecture](docs/GRAPH_RAG_ARCHITECTURE.md)**: Advanced code relationship analysis guide

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Support

If you encounter any issues or have questions, please open an issue on GitHub.
