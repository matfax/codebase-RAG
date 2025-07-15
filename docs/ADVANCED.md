"# Advanced Guide for Codebase RAG MCP Server

## Running Tests

To run the unit and integration tests, use `pytest`:

```bash
uv run pytest tests/
```

## Project Structure

```
./
├── README.md                  # Project overview and quick start
├── manual_indexing.py         # Standalone indexing tool
├── mcp_server                 # MCP server entry script
├── run_performance_tests.py   # Performance testing script
├── CLAUDE.md                  # Claude-specific documentation
├── FINAL_PROJECT_COMPLETION_REPORT.md # Project completion report
├── .ragignore                 # File exclusion patterns for indexing
├── pyproject.toml             # Project configuration and dependencies
├── uv.lock                    # Dependency lock file
├── docker-compose.cache.yml   # Docker Compose for cache services
├── redis.conf                 # Redis configuration file
├── LICENSE                    # Project license
├── .gitignore                 # Git ignore patterns
├── .pre-commit-config.yaml    # Pre-commit hooks configuration
├── src/                       # Source code directory
│   ├── main.py                # Main entry point
│   ├── run_mcp.py             # MCP server runner
│   ├── mcp_prompts.py         # MCP prompt definitions
│   ├── services/              # Core services (e.g., indexing, embedding)
│   ├── models/                # Data models
│   ├── utils/                 # Utility functions
│   ├── config/                # Configuration modules
│   ├── tests/                 # Unit tests
│   ├── prompts/               # Prompt templates
│   └── tools/                 # MCP tools implementations
├── docs/                      # Documentation files
│   ├── ADVANCED.md            # Advanced guide (this file)
│   ├── FEATURES.md            # Features overview
│   ├── SETUP.md               # Setup instructions
│   ├── BEST_PRACTICES.md      # Best practices
│   ├── INTEGRATION.md         # Integration guide
│   ├── MCP_TOOLS.md           # Tools reference
│   ├── ARCHITECTURE_DEEP_DIVE.md # Architecture deep dive
│   └── ... (other docs like cache-*.md)
├── tests/                     # Integration and unit tests
├── reports/                   # Generated reports
├── scripts/                   # Utility scripts
├── resources/                 # Additional resources
├── logs/                      # Log files
├── tasks/                     # Task-related files
├── ai_docs/                   # AI-generated documentation
├── progress/                  # Progress tracking
└── trees/                     # Tree structures or additional modules
```

### Key Components

#### Services Layer
- **IndexingService**: Orchestrates the indexing process.
- **CodeParserService**: Handles code chunking with Tree-sitter.
- **QdrantService**: Manages vector database operations.
- **EmbeddingService**: Integrates with Ollama for embeddings.
- **ProjectAnalysisService**: Analyzes project structure.
- **FileMetadataService**: Tracks file metadata for incremental indexing.
- **ChangeDetectorService**: Detects file changes.
- **CacheService**: Manages caching (Redis and memory).
- **ResilientRedisManager**: Handles robust Redis connections.

#### Models
- **FileMetadata**: Tracks file modification times, content hashes, and indexing state
- **CodeChunk**: Represents intelligent code chunks with rich metadata (function signatures, docstrings, syntax tree information)

#### Utilities
- **PerformanceMonitor**: Progress tracking, ETA estimation, and memory monitoring
- **StageLogger**: Detailed timing and performance logging for each processing stage

#### Collection Organization
The system creates the following Qdrant collections:
- `project_{name}_code`: Source code files
- `project_{name}_config`: Configuration files (JSON, YAML, etc.)
- `project_{name}_documentation`: Documentation files (Markdown, etc.)
- `project_{name}_file_metadata`: File change tracking for incremental indexing

## Real-World Usage Examples

This section demonstrates real-world usage of the Agentic-RAG system with actual exploration and analysis examples.

### 📋 Project Architecture Exploration Example

The [ARCHITECTURE_DEEP_DIVE.md](ARCHITECTURE_DEEP_DIVE.md) file was generated using this RAG system as a live demonstration. It showcases:

- **Comprehensive codebase analysis** using function-level search precision
- **Component relationship mapping** through intelligent chunking
- **Architecture documentation** generated from actual code exploration
- **Performance insights** derived from real system metrics

This document serves as both:
1. **Usage Example**: Shows how the system explores and understands complex codebases
2. **Architecture Reference**: Complete technical documentation of system components

### 🔍 How the Example Was Generated

The architecture documentation was created by:

1. **Initial Exploration**: Using `codebaseRAG:search` tools to understand entry points
2. **Component Discovery**: Function-level searches to map service relationships
3. **Data Flow Analysis**: Tracing execution paths through intelligent chunking
4. **Performance Analysis**: Real metrics from the current 11,363 indexed chunks
5. **Best Practices**: Derived from actual system behavior and optimization

### 📊 Key Insights from Real Usage

**Search Precision Results**:
- 🎯 **Function-level accuracy**: Returns specific functions like `src/services/code_parser_service.py:30`
- 🌳 **AST parsing insights**: Tree-sitter integration details from live code
- ⚡ **Performance data**: < 100ms parsing times from actual benchmarks
- 📈 **Scalability metrics**: 11,363 chunks indexed in ~1.1 minutes

**Real System Stats** (Current Project):
```
📊 Indexed Collections:
├── Code: 8,524 intelligent chunks (functions, classes, methods)
├── Config: 280 configuration chunks (JSON/YAML objects)
├── Documentation: 2,559 documentation chunks (Markdown sections)
└── Total: 11,363 semantic chunks ready for search
```

## Documentation

For comprehensive guides and references:
- [MCP_TOOLS.md](MCP_TOOLS.md): Complete MCP tools reference with parameters and examples
- [BEST_PRACTICES.md](BEST_PRACTICES.md): Best practices for search optimization and cross-project workflows" 