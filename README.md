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
