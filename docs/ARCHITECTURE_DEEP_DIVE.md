# 🏗️ Agentic-RAG Project Architecture Deep Dive

## Overview

**Agentic-RAG** is a sophisticated **Codebase RAG (Retrieval-Augmented Generation) MCP Server** that enables AI agents to understand and query codebases with **function-level precision** through intelligent syntax-aware code chunking and advanced query caching optimization.

### Core Innovation: Intelligent Code Chunking with Query Caching Layer
Unlike traditional RAG systems that process entire files, this system uses **Tree-sitter AST parsing** to break code into semantically meaningful chunks (functions, classes, methods) with rich metadata. The **query-caching-layer-wave** branch introduces a comprehensive multi-layer caching architecture with Redis integration for enhanced performance optimization.

## System Architecture & Data Flow

### Entry Points
- **`src/main.py:7`**: FastMCP server initialization and tool registration
- **`manual_indexing.py`**: Standalone heavy indexing operations
- **`demo_mcp_usage.py`**: Usage demonstrations

### Complete Data Flow with Query Caching

```
📁 Source Code
    ↓
🔍 File Discovery (project_analysis_service.py)
    ↓
🌳 AST Parsing (code_parser_service.py + Tree-sitter)
    ↓
⚡ Intelligent Chunking (Function/Class-level granularity)
    ↓
🧠 Batch Embedding Generation (embedding_service.py + Ollama)
    ↓
💾 Vector Storage (qdrant_service.py + Qdrant)
    ↓
📊 Metadata Tracking (file_metadata_service.py)
    ↓
🚀 Multi-Layer Cache (Redis + TTL optimization)
    ↓
🔎 Natural Language Search → Function-level Results
    ↓
🔗 Graph RAG Analysis (On-Demand) → Code Relationship Insights
```

## Key Architectural Components

### 🎯 Intelligent Chunking System (`src/services/code_parser_service.py:30`)
- **Tree-sitter Integration**: Supports 8+ programming languages (Python, JS/TS, Go, Rust, Java, C/C++)
- **AST-based Parsing**: Function-level precision instead of whole-file processing
- **Rich Metadata**: Extracts signatures, docstrings, breadcrumbs, access modifiers
- **Error Recovery**: Graceful handling of syntax errors with partial content preservation

**Supported Languages & Chunk Types:**

| Language | Supported Chunk Types |
|----------|----------------------|
| Python | Functions, Classes, Constants, Variables, Imports |
| JavaScript/TypeScript | Functions, Classes, Interfaces, Types, Imports/Exports |
| Go | Functions, Structs, Constants, Variables |
| Rust | Functions, Structs, Enums, Traits |
| Java | Classes, Methods, Interfaces |
| C/C++ | Functions, Structs, Classes |

### 🛠️ MCP Server & Tools (`src/tools/registry.py:12`)

**Core Tools Available:**
- `index_directory()`: Smart indexing with time estimation and incremental updates
- `search()`: Natural language semantic search with function-level results
- `health_check()`: System health monitoring
- `analyze_repository_tool()`: Repository structure analysis
- `check_index_status()`: Indexing status and recommendations
- `get_chunking_metrics_tool()`: Performance monitoring

### ⚙️ Services Layer (`src/services/`)

| Service | Purpose | Key Features |
|---------|---------|--------------|
| `indexing_service.py` | Orchestrates processing | Parallel processing, batch optimization |
| `code_parser_service.py` | AST parsing | Tree-sitter integration, intelligent chunking |
| `qdrant_service.py` | Vector database | Streaming operations, retry logic, cache integration |
| `embedding_service.py` | Embeddings | Ollama integration, batch processing, embedding cache |
| `project_analysis_service.py` | Repository analysis | File filtering, structure analysis |
| `file_metadata_service.py` | Change tracking | Incremental indexing, metadata storage |
| `search_cache_service.py` | Query caching | Redis integration, TTL management, cache optimization |

### 🚀 Query Caching Layer (New in query-caching-layer-wave)

**Multi-Tier Cache Architecture:**
- **Embedding Cache**: 3600s TTL (1 hour) - Caches generated embeddings for repeated queries
- **Search Cache**: 900s TTL (15 minutes) - Stores search results for identical queries
- **Project Cache**: 7200s TTL (2 hours) - Maintains project metadata and configuration
- **File Cache**: 1800s TTL (30 minutes) - Caches file metadata and parsing results

**Performance Optimizations:**
- **Async Cache Operations**: Non-blocking cache reads/writes using async/await patterns
- **Redis Integration**: High-performance in-memory caching with persistence options
- **Smart Cache Keys**: Hierarchical cache key design for efficient invalidation
- **Cache Health Monitoring**: Real-time cache performance metrics and alerts

## Navigation Strategy & Development Workflow

### 🚀 Most Efficient Exploration Path

1. **Start Here**: `src/main.py:7` - FastMCP app initialization
2. **Core Tools**: `src/tools/registry.py:12` - Understand all available MCP tools
3. **Intelligent Chunking**: `src/services/code_parser_service.py:30` - Tree-sitter AST parsing
4. **Data Flow**: `src/services/indexing_service.py` - Parallel processing orchestration
5. **Vector Operations**: `src/services/qdrant_service.py` - Database interactions

### 🔄 Request/Data Flow with Caching
```
MCP Client → FastMCP Tools → Cache Check → Services Layer → Tree-sitter Parser →
Embedding Service → Cache Store → Qdrant Storage → Cache Results → Search Results
```

**Cache-Optimized Search Flow:**
1. **Query Reception**: MCP client submits natural language query
2. **Cache Lookup**: Check Redis for cached embeddings and search results
3. **Cache Miss Handling**: Generate embeddings and perform vector search if not cached
4. **Result Caching**: Store results with appropriate TTL for future queries
5. **Response Delivery**: Return function-level results with context

### 🔗 Graph RAG Extension (Wave 4)

The Graph RAG enhancement adds advanced code relationship analysis through on-demand graph building:

**Graph RAG Flow:**
```
Graph RAG Tool Request → build_structure_graph() → Retrieve Chunks from Qdrant →
Structure Relationship Builder → Dependency Graph → Cache Results → Analysis Response
```

**Key Features:**
- **On-Demand Building**: Graphs are built only when Graph RAG tools are used
- **Breadcrumb System**: Unique identifiers for code components (e.g., "cache.service.RedisCacheService")
- **Dependency Tracking**: Uses `imports_used` metadata to build relationships
- **Caching Layer**: Built graphs are cached for performance
- **Multi-Analysis**: Structure analysis, similarity search, pattern recognition

**Graph RAG Tools:**
- `graph_analyze_structure_tool`: Hierarchical code structure analysis
- `graph_find_similar_implementations_tool`: Cross-project similarity search
- `graph_identify_patterns_tool`: Architectural pattern recognition

For detailed Graph RAG documentation, see [Graph RAG Architecture Guide](GRAPH_RAG_ARCHITECTURE.md).

### Development Commands Quick Reference

```bash
# Setup
.venv/bin/poetry install
.venv/bin/python src/run_mcp.py

# Testing
.venv/bin/pytest tests/
python test_full_functionality.py

# Manual Indexing
python manual_indexing.py -d /path/to/repo -m clear_existing
python manual_indexing.py -d /path/to/repo -m incremental --verbose
```

## Key Design Insights

### Why This Architecture?

1. **Function-Level Precision**: Instead of searching entire files, you get specific functions/classes
2. **Language-Agnostic**: Tree-sitter supports multiple programming languages consistently
3. **Scalable**: Parallel processing, batch operations, and incremental indexing
4. **Error-Tolerant**: Graceful fallback for syntax errors
5. **Memory-Efficient**: Streaming operations with garbage collection

### Performance Characteristics

- **AST Parsing**: < 100ms per file (99th percentile)
- **Memory Usage**: < 50MB additional per 1000 files
- **Total Processing Impact**: < 20% increase over whole-file approach
- **Indexing Speed**: ~1.1 minutes for 129 files (typical small project)

### Collection Architecture

**Content Collections** (store intelligent chunks with embeddings):
- `project_{name}_code`: Intelligent code chunks - functions, classes, methods
- `project_{name}_config`: Structured config chunks - JSON/YAML objects
- `project_{name}_documentation`: Document chunks - Markdown headers, docs

**Metadata Collection** (tracks file states):
- `project_{name}_file_metadata`: File change tracking for incremental indexing

**Example Collection Stats** (Current Project - query-caching-layer-wave):
- Code: 104,779 intelligent chunks
- Config: 204 configuration chunks
- Documentation: 13,253 documentation chunks
- File Metadata: 324 file tracking records
- **Total**: 118,560 indexed chunks

## Advanced Features

### Intelligent Error Handling

**Syntax Error Tolerance:**
- **ERROR Node Detection**: Identifies Tree-sitter ERROR nodes in AST
- **Partial Processing**: Preserves correct code sections even with syntax errors
- **Smart Recovery**: Includes surrounding context for better understanding
- **Graceful Fallback**: Falls back to whole-file processing when AST parsing fails

**Error Classification:**
- **Minor Errors**: Small syntax issues that don't affect overall structure
- **Major Errors**: Significant parsing failures requiring fallback
- **Recoverable Errors**: Errors where partial content can still be extracted

### Performance Optimization Features

**Core Optimizations:**
- **Parallel Processing**: Multi-threaded file reading and AST parsing
- **Parser Caching**: Cached Tree-sitter parsers for improved performance
- **Batch Operations**: Grouped embedding generation and database insertions
- **Streaming Architecture**: Memory-efficient processing for large codebases
- **Progress Monitoring**: Real-time progress tracking with ETA calculation
- **Memory Management**: Automatic cleanup and garbage collection
- **Adaptive Batching**: Dynamic batch size adjustment based on memory usage
- **Retry Logic**: Exponential backoff for failed operations

**Query Caching Optimizations:**
- **Redis Cache Layer**: High-performance in-memory caching with configurable TTL
- **Async Cache Operations**: Non-blocking cache reads/writes for improved throughput
- **Smart Cache Invalidation**: Hierarchical cache key design for efficient updates
- **Cache Warm-up**: Preemptive cache population for frequently accessed data
- **Cache Health Monitoring**: Real-time metrics for cache hit rates and performance
- **Memory-Efficient Caching**: Optimized serialization for large result sets
- **Cache Compression**: Optional compression for large cached objects
- **Distributed Cache Support**: Redis cluster support for horizontal scaling

### Configuration & Environment

**Environment Variables** (`.env` file):

```bash
# Basic Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_DEFAULT_EMBEDDING_MODEL=nomic-embed-text
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Redis Cache Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_MAX_CONNECTIONS=20

# Cache TTL Settings
CACHE_DEFAULT_TTL=1800
EMBEDDING_CACHE_TTL=3600
SEARCH_CACHE_TTL=900
PROJECT_CACHE_TTL=7200
FILE_CACHE_TTL=1800

# Performance Tuning
INDEXING_CONCURRENCY=4
INDEXING_BATCH_SIZE=20
EMBEDDING_BATCH_SIZE=10
QDRANT_BATCH_SIZE=500
MEMORY_WARNING_THRESHOLD_MB=1000
MAX_FILE_SIZE_MB=5
MAX_DIRECTORY_DEPTH=20

# Cache Performance Settings
CACHE_COMPRESSION_ENABLED=true
CACHE_WARM_UP_ENABLED=true
CACHE_MONITORING_ENABLED=true

# Development Settings
LOG_LEVEL=INFO
FOLLOW_SYMLINKS=false
```

## Development Workflow

### How Components Work Together

**Indexing Workflow:**
1. **Project Discovery**: `ProjectAnalysisService` scans directories respecting `.ragignore`
2. **AST Processing**: `CodeParserService` uses Tree-sitter to extract semantic chunks
3. **Parallel Processing**: `IndexingService` coordinates batch processing across multiple threads
4. **Vector Generation**: `EmbeddingService` creates embeddings via Ollama with caching
5. **Storage**: `QdrantService` streams data to vector database with retry logic
6. **Cache Population**: Initial cache warm-up for frequently accessed data

**Search Workflow with Caching:**
1. **Query Reception**: Natural language query received via MCP
2. **Cache Check**: `SearchCacheService` checks Redis for cached results
3. **Cache Hit**: Return cached results immediately (sub-millisecond response)
4. **Cache Miss**: Generate embeddings (with embedding cache check)
5. **Vector Search**: Query Qdrant for semantic matches
6. **Result Processing**: Add context and metadata to function-level results
7. **Cache Store**: Store results in Redis with appropriate TTL
8. **Response**: Return enriched function-level matches with context

### Incremental Indexing Workflow

1. **Initial Indexing**: Full codebase processing with metadata storage
2. **Change Detection**: Compare file modification times and content hashes
3. **Selective Processing**: Only reprocess files with detected changes
4. **Metadata Updates**: Update file metadata after successful processing
5. **Collection Management**: Automatic cleanup of stale entries

## Next Steps for Deep Understanding

### Immediate Actions
1. **Explore AST Parsing**: Read `src/services/code_parser_service.py` to understand Tree-sitter integration
2. **Test Intelligent Search**: Use the RAG search tools to see function-level precision in action
3. **Review Language Support**: Check `src/utils/tree_sitter_manager.py` for supported languages
4. **Study Performance**: Examine `src/utils/performance_monitor.py` for optimization strategies

### Advanced Exploration
1. **Custom Language Support**: Extend Tree-sitter parsers for new languages
2. **Chunk Optimization**: Tune chunk types and metadata extraction
3. **Search Enhancement**: Improve semantic search algorithms
4. **Scaling**: Optimize for larger codebases

### Testing & Validation

```bash
# Core functionality tests
.venv/bin/pytest tests/test_code_parser_service.py
.venv/bin/pytest tests/test_intelligent_chunking.py

# Performance benchmarks
.venv/bin/pytest tests/test_performance_benchmarks.py

# Full integration test
python test_full_functionality.py
```

## Key Takeaways

This is a **production-ready, intelligent codebase RAG system** with advanced query caching optimization:

- ✅ **Function-level search precision**
- ✅ **Multi-language support** via Tree-sitter
- ✅ **Scalable parallel processing**
- ✅ **Error-tolerant AST parsing**
- ✅ **Incremental indexing capabilities**
- ✅ **Rich MCP tool ecosystem**
- ✅ **Comprehensive performance monitoring**
- ✅ **Memory-efficient streaming operations**
- 🚀 **Multi-layer query caching** with Redis integration
- 🚀 **Sub-millisecond cache hit responses**
- 🚀 **Async cache operations** for non-blocking performance
- 🚀 **Intelligent cache warming** and invalidation strategies
- 🚀 **Real-time cache health monitoring** and optimization

The system is designed for immediate productivity with advanced semantic search operations and enterprise-grade caching performance. You can start using the RAG search tools to explore code relationships and understand implementation patterns with unprecedented precision and speed.

## External Dependencies

- **Qdrant**: Vector database for embeddings storage
- **Ollama**: Local embedding model service
- **Tree-sitter**: Syntax-aware parsing for multiple languages
- **FastMCP**: Model Context Protocol server framework
- **Redis**: High-performance in-memory caching layer
- **Python async/await**: Asynchronous processing for optimal performance

## File Organization Patterns

- **Smart Filtering**: Uses `.ragignore` files for excluding directories/files
- **Automatic Categorization**: Files categorized by type (code/config/documentation)
- **Collection Naming**: Deterministic collection names: `project_{name}_{type}`
- **Binary Detection**: Automatically skips binary files and large files
- **Language Detection**: Identifies programming languages for better categorization
- **Gitignore Integration**: Respects `.gitignore` patterns for relevant file detection
