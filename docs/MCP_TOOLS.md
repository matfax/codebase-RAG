# MCP Tools Reference

Comprehensive documentation for all MCP tools provided by the Codebase RAG MCP Server.

## Core Search Tool

### `search` - Semantic Code Search

Search indexed codebases using natural language queries with function-level precision.

**Parameters:**
- `query` (required): Natural language search query
- `n_results` (optional, default: 5): Number of results to return (1-100)
- `cross_project` (optional, default: false): Search across all indexed projects
- `search_mode` (optional, default: "hybrid"): Search strategy ("semantic", "keyword", or "hybrid")
- `include_context` (optional, default: true): Include surrounding code context
- `context_chunks` (optional, default: 1): Number of context chunks before/after (0-5)
- `target_projects` (optional): List of specific project names to search in

**Example Queries:**
- "Find functions that handle file uploads"
- "Show me React components that use useState hook"
- "Find error handling patterns in Python"
- "Locate database connection initialization code"

## Indexing Tools

### `index_directory` - Index a Codebase

Index a directory for intelligent searching with syntax-aware code chunking.

**Parameters:**
- `directory` (optional, default: "."): Directory to index
- `patterns` (optional): File patterns to include (e.g., ["*.py", "*.js"])
- `recursive` (optional, default: true): Index subdirectories
- `clear_existing` (optional, default: false): Clear existing indexed data
- `incremental` (optional, default: false): Only process changed files
- `project_name` (optional): Custom project name for collections

**Features:**
- Intelligent syntax-aware chunking for 8+ programming languages
- Automatic change detection for incremental updates
- Progress tracking with time estimation
- Comprehensive error handling and recovery

### `check_index_status` - Check Indexing Status

Get information about the current indexing state of a directory.

**Parameters:**
- `directory` (optional, default: "."): Directory to check

**Returns:** Status information and recommendations for indexed data.

## Analysis Tools

### `health_check` - Server Health Check

Verify MCP server health and dependency connectivity.

**Features:**
- Qdrant database connectivity test
- Ollama service availability check
- System resource monitoring
- Performance metrics collection

### `analyze_repository_tool` - Repository Analysis

Get detailed statistics and analysis of a repository structure.

**Parameters:**
- `directory` (optional, default: "."): Directory to analyze

**Returns:** File counts, language distribution, complexity assessment, and indexing recommendations.

### `get_file_filtering_stats_tool` - File Filtering Analysis

Analyze how files are filtered during indexing to optimize .ragignore patterns.

**Parameters:**
- `directory` (optional, default: "."): Directory to analyze

**Returns:** Detailed breakdown of file filtering with exclusion reasons and recommendations.

## Progress Monitoring Tools

### `get_indexing_progress_tool` - Real-time Progress

Get current progress of any ongoing indexing operations.

**Returns:** Real-time progress updates including ETA, processing rate, and memory usage.

### `get_chunking_metrics_tool` - Chunking Performance Metrics

Get detailed metrics about intelligent code chunking performance.

**Parameters:**
- `language` (optional): Specific language to get metrics for
- `export_path` (optional): Path to export detailed metrics to JSON file

**Returns:** Chunking success rates, processing speeds, and quality metrics by language.

## Utility Tools

### `diagnose_parser_health_tool` - Parser Diagnostics

Diagnose Tree-sitter parser health and functionality.

**Parameters:**
- `comprehensive` (optional, default: false): Run comprehensive diagnostics
- `language` (optional): Test specific language only

**Returns:** Parser installation status, functionality tests, and health recommendations.

### `reset_chunking_metrics_tool` - Reset Metrics

Reset session-specific chunking performance metrics.

**Returns:** Confirmation of metrics reset.

## Project Management Tools

### `get_project_info_tool` - Project Information

Get information about the current project context.

**Parameters:**
- `directory` (optional, default: "."): Directory to analyze

**Returns:** Project metadata, configuration, and indexing status.

### `list_indexed_projects_tool` - List All Projects

List all projects that have indexed data in the system.

**Returns:** Information about all indexed projects with statistics.

### `clear_project_data_tool` - Clear Project Data

Clear all indexed data for a specific project.

**Parameters:**
- `project_name` (optional): Specific project to clear
- `directory` (optional, default: "."): Directory context if project_name not provided

**Returns:** Clearing operation results and confirmation.

## File-Level Tools

### `get_file_metadata_tool` - File Metadata

Get detailed metadata for a specific file from the vector database.

**Parameters:**
- `file_path` (required): Path to the file

**Returns:** File metadata, indexing information, and chunk statistics.

### `clear_file_metadata_tool` - Clear File Data

Clear all chunks and metadata for a specific file.

**Parameters:**
- `file_path` (required): Path to the file to clear
- `collection_name` (optional): Specific collection to clear from

**Returns:** File clearing operation results.

### `reindex_file_tool` - Reindex Single File

Reindex a specific file by clearing existing chunks and reprocessing.

**Parameters:**
- `file_path` (required): Path to the file to reindex

**Returns:** File reindexing operation results and statistics.

## Best Practices

### Query Optimization
- Use specific, descriptive language in search queries
- Include relevant programming language or framework names
- Combine functional and technical terms for better results

### Cross-Project Search
- Enable `cross_project=true` for searching across all indexed projects
- Use `target_projects=["project1", "project2"]` to search specific projects only
- Use project-specific terms when searching within a single project
- Consider context chunks for understanding code relationships

**Search Scope Options:**
- **Current Project Only** (default): `cross_project=false`
- **All Projects**: `cross_project=true`
- **Specific Projects**: `target_projects=["PocketFlow", "MyApp"]`

### Performance Optimization
- Use incremental indexing for large codebases after initial indexing
- Monitor memory usage during large indexing operations
- Configure batch sizes based on system resources

## Error Handling

All tools include comprehensive error handling with:
- Detailed error messages and recovery suggestions
- Graceful degradation for partial failures
- Automatic retry logic for transient failures
- Progress preservation during interruptions

## Tool Integration

Tools are designed to work together:
1. Use `analyze_repository_tool` before indexing to understand scope
2. Run `health_check` to verify system readiness
3. Index with `index_directory` using recommended settings
4. Search with `search` using various query strategies
5. Monitor with progress tools during long operations
