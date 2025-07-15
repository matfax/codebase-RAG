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

For full features details, including intelligent code chunking and advanced features, see [docs/FEATURES.md](docs/FEATURES.md).

For advanced topics like project structure, running tests, and real-world examples, see [docs/ADVANCED.md](docs/ADVANCED.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Support

If you encounter any issues or have questions, please open an issue on GitHub.
