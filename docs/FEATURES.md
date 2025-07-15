# Codebase RAG MCP Server Features

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) Model-Controller-Provider (MCP) server designed to assist AI agents and junior developers in understanding and navigating codebases. It leverages Qdrant as its vector database for storing codebase embeddings and integrates with Ollama for generating these embeddings. The server provides API endpoints for indexing local directories or GitHub repositories and for querying the indexed codebases using natural language.

## Core Functionality
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

**Fully Implemented Languages:**
- **Python (.py, .pyw, .pyi)**: Functions, classes, methods, constants, docstrings, decorators
- **JavaScript (.js, .jsx, .mjs, .cjs)**: Functions, classes, modules, arrow functions
- **TypeScript (.ts)**: Interfaces, types, classes, functions, generics, annotations
- **TypeScript JSX (.tsx)**: React components, interfaces, types, functions
- **Go (.go)**: Functions, structs, interfaces, methods, packages
- **Rust (.rs)**: Functions, structs, impl blocks, traits, modules, macros
- **Java (.java)**: Classes, methods, interfaces, annotations, generics
- **C++ (.cpp, .cxx, .cc, .c, .hpp, .hxx, .hh, .h)**: Functions, classes, structs, namespaces, templates

**Structured Files:**
- **JSON/YAML**: Object-level chunking (e.g., separate chunks for `scripts`, `dependencies`)
- **Markdown**: Header-based hierarchical chunking
- **Configuration files**: Section-based parsing

### How It Works

1. **AST Parsing**: Uses Tree-sitter to generate syntax trees for each source file
2. **Semantic Extraction**: Identifies functions, classes, methods, and other semantic units
3. **Context Enhancement**: Preserves surrounding context (imports, docstrings, inline comments)
4. **Metadata Enrichment**: Extracts signatures, parameter types, access modifiers, and relationships
5. **Error Handling**: Gracefully handles syntax errors with smart fallback to whole-file processing

For more details on intelligent chunking, see [INTELLIGENT_CHUNKING_GUIDE.md](INTELLIGENT_CHUNKING_GUIDE.md).

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