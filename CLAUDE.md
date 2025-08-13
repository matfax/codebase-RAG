# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Codebase RAG (Retrieval-Augmented Generation) MCP Server** - a sophisticated Python application that provides AI agents with function-level precision understanding of codebases through intelligent indexing and multi-modal search capabilities. The project is built around MCP (Model Context Protocol) tools and leverages Qdrant (vector database), Ollama (local embeddings), and Redis (caching) for high-performance code analysis.

## Common Development Commands

### Setup and Installation
```bash
# Install dependencies and create virtual environment
uv sync

# Copy environment configuration template
cp .env.example .env
# Edit .env file to customize settings
```

### MCP Server Operations
```bash
# Run the MCP server (primary interface)
./mcp_server

# Alternatively, run directly with uv
uv run python src/run_mcp.py
```

### Testing
**IMPORTANT**: All testing commands require the virtual environment to be activated first:

```bash
# Activate virtual environment (required for all testing)
source .venv/bin/activate    # macOS/Linux
# or
.venv\Scripts\activate      # Windows

# Run all tests with comprehensive coverage
python tests/run_tests.py

# Run specific test suites
python tests/run_tests.py --unit        # Unit tests only
python tests/run_tests.py --integration # Integration tests only
python tests/run_tests.py --coverage    # Coverage analysis
python tests/run_tests.py --benchmark   # Performance benchmarks

# Run tests with pytest directly
pytest tests/                           # All tests
pytest tests/ -m unit                  # Unit tests only
pytest tests/ -m integration           # Integration tests only
pytest tests/ -m "not slow"           # Exclude slow tests

# Alternative: Use uv run to auto-activate environment
uv run python tests/run_tests.py
uv run pytest tests/
```

### Code Quality and Linting
```bash
# Format code with Black
black src/ tests/

# Lint with Ruff
ruff check src/ tests/
ruff check --fix src/ tests/           # Auto-fix issues

# Combined format and lint
black src/ tests/ && ruff check --fix src/ tests/
```

### Manual Indexing and Development Testing
**IMPORTANT**: Development scripts require virtual environment activation:

```bash
# Activate virtual environment first
source .venv/bin/activate    # macOS/Linux

# Index a codebase manually (for testing)
python manual_indexing.py /path/to/project

# Validate runtime fixes
python validate_runtime_fixes.py

# Alternative: Use uv run to auto-activate environment
uv run python manual_indexing.py /path/to/project
uv run python validate_runtime_fixes.py
```

## Architecture Overview

### High-Level Architecture
The system implements a **layered MCP architecture** with intelligent caching, multi-modal retrieval, and Graph RAG capabilities:

1. **MCP Layer** (`src/main.py`, `src/tools/registry.py`)
   - FastMCP application with environment-based tool registration
   - Production vs Development tool sets for optimal performance
   - Comprehensive tool ecosystem with 50+ specialized tools

2. **Service Layer** (`src/services/`)
   - **Core Services**: Embedding, Indexing, Search, Caching
   - **Graph RAG Services**: Function chain analysis, pattern recognition, structure analysis
   - **Multi-Modal Services**: LightRAG-inspired retrieval strategies (Local, Global, Hybrid, Mix)
   - **Performance Services**: Monitoring, optimization, health checks

3. **Storage Layer**
   - **Qdrant**: Vector database for embeddings and semantic search
   - **Redis**: Multi-tier caching system with intelligent eviction
   - **File System**: Intelligent chunking with Tree-sitter parsing

4. **Intelligence Layer**
   - **Multi-Modal Retrieval**: Automatic mode selection based on query analysis
   - **Graph RAG**: Function-level relationship analysis and chain tracing
   - **Smart Chunking**: Syntax-aware code parsing for multiple languages

### Key Components

#### MCP Tools Architecture (`src/tools/`)
- **Registry System**: Environment-aware tool registration (production/development)
- **Core Tools**: Health, auto-configuration, performance monitoring
- **Search Tools**: Multi-modal search, repository analysis, index management
- **Graph RAG Tools**: Function chain tracing, pattern identification, structural analysis
- **Cache Tools**: Comprehensive cache management and optimization

#### Service Architecture (`src/services/`)
- **Modular Design**: Each service is independently testable and scalable
- **Performance-First**: All services designed for <15 second response times
- **Error Handling**: Graceful degradation and automatic recovery
- **Caching Integration**: Multi-tier caching throughout the stack

#### Models and Data Structures (`src/models/`)
- **Code Models**: `code_chunk.py`, `file_metadata.py`, `function_call.py`
- **Cache Models**: `cache_models.py`, `breadcrumb_cache_models.py`
- **Query Models**: `query_features.py`, `routing_decision.py`

## Development Environment

### Environment Variables
The project uses extensive environment configuration for optimal performance tuning:

- **Core Settings**: `OLLAMA_HOST`, `QDRANT_HOST`, `QDRANT_PORT`
- **Performance Tuning**: `INDEXING_CONCURRENCY`, `EMBEDDING_BATCH_SIZE`, `QDRANT_BATCH_SIZE`
- **MCP Configuration**: `MCP_ENV` (production/development), `MCP_DEBUG_LEVEL`
- **Cache Settings**: `MEMORY_WARNING_THRESHOLD_MB`, `CACHE_DEBUG_MODE`

### Code Standards
- **Python 3.10+** with modern type hints and async/await patterns
- **Line Length**: 140 characters (configured in pyproject.toml)
- **Code Quality**: Black formatting + Ruff linting with project-specific rules
- **Test Coverage**: 90%+ coverage requirement with comprehensive test suite

### Directory Structure Insights
- **`src/prompts/`**: MCP Prompts system for guided AI workflows
- **`src/utils/`**: Shared utilities for performance, caching, and diagnostics
- **`docs/`**: Comprehensive documentation including architecture guides
- **`tests/`**: Extensive test suite with fixtures and performance benchmarks
- **`scripts/`**: Deployment and migration utilities

## Testing Strategy

### Test Organization
- **Unit Tests**: Individual component testing with mocking
- **Integration Tests**: Service integration and MCP tool testing
- **Performance Tests**: Benchmarking and load testing
- **Runtime Reliability**: Comprehensive validation of production scenarios

### Test Configuration
- **Pytest Configuration**: `tests/pytest.ini` with coverage requirements
- **Test Markers**: `unit`, `integration`, `performance`, `slow`, `redis`
- **Coverage**: 90% minimum with HTML and XML reporting
- **Async Testing**: Full asyncio support with `pytest-asyncio`

## Key Development Patterns

### MCP Tool Development
- All tools follow the async pattern and return structured dictionaries
- Environment-aware output (minimal for agents, detailed for development)
- Comprehensive error handling with graceful degradation
- Performance monitoring with 15-second timeout guarantees

### Service Development
- **Dependency Injection**: Services are loosely coupled with clear interfaces
- **Caching Integration**: All services use multi-tier caching strategies
- **Performance Monitoring**: Built-in telemetry and health monitoring
- **Error Recovery**: Automatic retry logic and fallback mechanisms

### Cache Strategy
- **Multi-Tier**: L1 (memory) + L2 (Redis) with intelligent promotion/demotion
- **Invalidation**: Sophisticated cascade invalidation with partial updates
- **Performance**: Sub-millisecond L1 cache hits, optimized serialization
- **Monitoring**: Real-time performance metrics and alerting

## Performance Considerations

### Indexing Performance
- **Batch Processing**: Configurable batch sizes for optimal throughput
- **Concurrency**: Parallel file processing with resource management
- **Memory Management**: Automatic memory monitoring and cleanup
- **Incremental Updates**: Smart change detection and partial reindexing

### Search Performance
- **Multi-Modal Optimization**: Automatic mode selection for optimal results
- **Caching Strategy**: Aggressive caching of embeddings and search results
- **Timeout Management**: Guaranteed <15 second responses with graceful degradation
- **Resource Management**: Automatic resource cleanup and memory management

This architecture provides a robust foundation for AI-powered code understanding with production-ready performance and comprehensive tooling for development and maintenance.
