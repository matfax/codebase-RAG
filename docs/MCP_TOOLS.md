# MCP Tools Reference

Comprehensive documentation for all MCP tools provided by the Codebase RAG MCP Server.

## 🔧 Output Control & Environment Configuration

The MCP server automatically adjusts output detail levels based on environment variables to optimize performance for different use cases:

### Environment Variables
- **`MCP_ENV`**: Controls overall tool behavior and output detail
  - `production`: Minimal output optimized for AI agents (faster, concise)
  - `development`: Full technical details and metadata (debugging-friendly)

- **`MCP_DEBUG_LEVEL`**: Fine-grained control over diagnostic information
  - `DEBUG`: Includes performance metrics, internal metadata, detailed diagnostics
  - `INFO`: Standard technical details (default)
  - `WARNING`/`ERROR`: Minimal output for production use

- **`CACHE_DEBUG_MODE`**: When `true`, includes cache performance data in outputs

### Manual Override with `minimal_output`

Many tools support a `minimal_output` parameter to explicitly request simplified output regardless of environment settings:

**When `minimal_output=true`:**
- Returns only essential fields: `file_path`, `content`, `breadcrumb`, `chunk_type`, `language`, `line_start`, `line_end`
- Removes performance metrics, technical scoring details, and internal metadata
- Optimized for AI agents that need focused, actionable results
- Reduces response size by ~60-80% for faster processing

**Example:**
```python
# Standard output (full details)
await search("error handling functions")

# Minimal output (agent-optimized)
await search("error handling functions", minimal_output=true)
```

## Core Search Tools

### `search` - Enhanced Semantic Code Search

Search indexed codebases using natural language queries with function-level precision and advanced multi-modal retrieval capabilities.

**Standard Parameters:**
- `query` (required): Natural language search query
- `n_results` (optional, default: 5): Number of results to return (1-100)
- `cross_project` (optional, default: false): Search across all indexed projects
- `search_mode` (optional, default: "hybrid"): Search strategy ("semantic", "keyword", or "hybrid")
- `include_context` (optional, default: true): Include surrounding code context
- `context_chunks` (optional, default: 1): Number of context chunks before/after (0-5)
- `target_projects` (optional): List of specific project names to search in
- 🆕 `collection_types` (optional): List of collection types to search in (["code"], ["config"], ["documentation"], or ["code", "config"])

**🆕 Wave 7.0 Enhanced Parameters:**
- `multi_modal_mode` (optional): Manual mode selection ("local", "global", "hybrid", "mix")
- `enable_multi_modal` (optional, default: false): Enable LightRAG-inspired multi-modal retrieval
- `enable_manual_mode_selection` (optional, default: false): Allow manual override of automatic mode selection
- `include_query_analysis` (optional, default: false): Include detailed query analysis in response
- `performance_timeout_seconds` (optional, default: 15): Maximum execution time in seconds
- `minimal_output` (optional, default: false): Return simplified output optimized for AI agents

**Multi-Modal Retrieval Modes:**
- **Local Mode**: Deep entity-focused retrieval using low-level keywords
- **Global Mode**: Broad relationship-focused retrieval using high-level keywords
- **Hybrid Mode**: Combined local+global with balanced context
- **Mix Mode**: Intelligent automatic mode selection based on query analysis

**Example Queries:**
- "Find functions that handle file uploads"
- "Show me React components that use useState hook"
- "Find error handling patterns in Python"
- "Locate database connection initialization code"

**Collection Filtering Examples:**
- Search only code files: `collection_types=["code"]`
- Search only config files: `collection_types=["config"]` for cache configurations
- Search only documentation: `collection_types=["documentation"]` for architecture docs
- Search multiple types: `collection_types=["code", "config"]` for implementation + configuration

### `multi_modal_search` - Advanced Multi-Modal Retrieval

🆕 **NEW in Wave 7.0** - Dedicated multi-modal search tool with LightRAG-inspired retrieval strategies.

**Parameters:**
- `query` (required): Natural language search query
- `n_results` (optional, default: 10): Number of results to return (1-50)
- `mode` (optional): Manual mode selection ("local", "global", "hybrid", "mix")
- `target_projects` (optional): List of specific project names to search in
- `cross_project` (optional, default: false): Search across all projects
- `enable_manual_mode_selection` (optional, default: false): Allow manual mode override
- `include_analysis` (optional, default: true): Include query analysis in response
- `include_performance_metrics` (optional, default: false): Include performance metrics

**Mode Selection Strategy:**
- **Automatic**: Uses query analysis to select optimal mode
- **Manual**: Allows explicit mode specification
- **Adaptive**: Learns from query patterns and performance

**Performance Features:**
- <20 second timeout with graceful degradation
- Performance monitoring and metrics collection
- Automatic fallback to simpler modes on timeout
- Cache-aware execution for repeated queries

### `analyze_query_features` - Query Analysis Tool

🆕 **NEW in Wave 7.0** - Analyze query characteristics and recommend optimal retrieval strategies.

**Parameters:**
- `query` (required): The search query to analyze

**Returns:**
- Query type classification (entity_focused, relationship_focused, conceptual)
- Complexity analysis (simple, complex, multi_faceted)
- Keyword extraction (entity names, concept terms, technical terms)
- Context hints (language, framework, domain)
- Mode recommendation with confidence score
- Detailed reasoning for recommendations

### `get_retrieval_mode_performance` - Performance Analytics

🆕 **NEW in Wave 7.0** - Get comprehensive performance metrics for multi-modal retrieval modes.

**Parameters:**
- `mode` (optional): Specific mode to analyze ("local", "global", "hybrid", "mix")
- `include_comparison` (optional, default: true): Include cross-mode comparison
- `include_alerts` (optional, default: true): Include performance alerts
- `include_history` (optional, default: false): Include query history
- `history_limit` (optional, default: 50): Limit for query history

**Returns:**
- Mode-specific performance statistics
- Cross-mode comparison analysis
- Active performance alerts
- Query history and trends
- Optimization recommendations

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

## System Management & Configuration Tools

### `generate_auto_configuration` - Auto-Configuration Service

🆕 **NEW in Wave 7.0** - Automatically generate optimal configuration based on system capabilities and project characteristics.

**Parameters:**
- `directory` (optional, default: "."): Directory to analyze for configuration
- `usage_pattern` (optional, default: "balanced"): Expected usage pattern ("development", "production", "research", "balanced")

**Returns:**
- System capability analysis (CPU, memory, storage)
- Project characteristics assessment (size, complexity, languages)
- Optimized parameter recommendations for all MCP tools
- Performance optimization suggestions
- Resource allocation recommendations

**Configuration Areas:**
- **Search Parameters**: Optimal n_results, timeout values, mode preferences
- **Indexing Settings**: Batch sizes, chunking strategies, incremental settings
- **Cache Configuration**: Memory allocation, TTL settings, eviction policies
- **Performance Limits**: Timeout thresholds, max depth values, result limits

### `run_compatibility_check` - Backward Compatibility Verification

🆕 **NEW in Wave 7.0** - Verify that all existing MCP tool interfaces remain functional.

**Returns:**
- Compatibility test results for all core tools
- Interface validation status
- Breaking change detection
- Migration recommendations if needed

### `get_performance_dashboard` - Performance Dashboard

🆕 **NEW in Wave 7.0** - Comprehensive performance monitoring dashboard for all MCP tools.

**Returns:**
- Overall system performance summary
- Per-tool performance statistics
- Active operations monitoring
- System resource utilization
- Performance target compliance (15-second response time)
- Performance history and trends

### `get_service_health_status` - Service Health Monitoring

🆕 **NEW in Wave 7.0** - Get comprehensive health status across all services with error tracking.

**Returns:**
- Service degradation levels for all components
- Error history and patterns
- Recovery suggestions
- Service health trends
- Automatic degradation thresholds

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

## Graph RAG Tools

The Graph RAG tools provide advanced code relationship analysis by building and analyzing code dependency graphs. These tools work on-demand and require the project to be indexed first.

### `graph_analyze_structure_tool` - Enhanced Code Structure Analysis

🔧 **ENHANCED in Wave 7.0** - Analyze the hierarchical structure and relationships of code components using graph traversal algorithms with performance optimizations.

**Parameters:**
- `breadcrumb` (required): The code component to analyze (e.g., "cache.service.RedisCacheService")
- `project_name` (required): Name of the project to analyze
- `analysis_type` (optional, default: "comprehensive"): Type of analysis ("comprehensive", "hierarchy", "connectivity", "overview")
- `max_depth` (optional, default: auto-configured): Maximum depth for relationship traversal (1-15, auto-configured based on system capabilities)
- `include_siblings` (optional, default: false): Whether to include sibling components
- `include_connectivity` (optional, default: true): Whether to analyze connectivity patterns
- `force_rebuild_graph` (optional, default: false): Force rebuild the structure graph
- 🆕 `generate_report` (optional, default: false): Whether to generate comprehensive analysis report
- 🆕 `include_recommendations` (optional, default: true): Whether to include optimization recommendations
- 🆕 `enable_performance_optimization` (optional, default: true): Whether to enable performance optimizations

**🆕 Wave 7.0 Enhancements:**
- **Auto-Configuration**: max_depth automatically configured based on system capabilities
- **Performance Monitoring**: <15 second response time guarantee
- **Enhanced Reports**: Comprehensive analysis reports with statistics and recommendations
- **Optimization Engine**: Intelligent performance optimization for large projects

**Example Usage:**
```python
result = await graph_analyze_structure_tool(
    breadcrumb="cache.service.RedisCacheService",
    project_name="my-project",
    analysis_type="comprehensive"
)
```

### `graph_find_similar_implementations_tool` - Enhanced Cross-Project Similarity Search

🔧 **ENHANCED in Wave 7.0** - Find similar code implementations across projects using semantic and structural analysis with improved performance limits.

**Parameters:**
- `query` (required): Natural language description of what to search for
- `source_breadcrumb` (optional): Specific breadcrumb to find similar implementations for
- `source_project` (optional): Source project name (used with source_breadcrumb)
- `target_projects` (optional): List of specific projects to search in
- `exclude_projects` (optional): List of projects to exclude from search
- `chunk_types` (optional): List of chunk types to include ("function", "class", "method", etc.)
- `languages` (optional): List of programming languages to include
- `similarity_threshold` (optional, default: 0.7): Minimum similarity score (0.0-1.0)
- `structural_weight` (optional, default: 0.5): Weight for structural vs semantic similarity
- 🔧 `max_results` (optional, default: 25): Maximum number of results (1-100, increased from 50)
- `include_implementation_chains` (optional, default: false): Include implementation chain analysis
- `include_architectural_context` (optional, default: true): Include architectural context

**🆕 Wave 7.0 Enhancements:**
- **Increased Limits**: max_results increased to 100 for comprehensive analysis
- **Auto-Configuration**: Parameters automatically optimized based on system capabilities
- **Performance Monitoring**: <15 second response time with timeout handling

**Example Usage:**
```python
result = await graph_find_similar_implementations_tool(
    query="cache service implementation patterns",
    target_projects=["project1", "project2"],
    similarity_threshold=0.7
)
```

### `graph_identify_patterns_tool` - Enhanced Architectural Pattern Recognition

🔧 **ENHANCED in Wave 7.0** - Identify architectural patterns and design patterns in codebases using pattern recognition algorithms with increased capacity.

**Parameters:**
- `project_name` (required): Name of the project to analyze
- `pattern_types` (optional): List of pattern types to look for ("structural", "behavioral", "creational", "naming", "architectural")
- `scope_breadcrumb` (optional): Limit analysis to specific breadcrumb scope
- `min_confidence` (optional, default: 0.6): Minimum confidence threshold (0.0-1.0)
- `include_comparisons` (optional, default: true): Include pattern comparison analysis
- `include_improvements` (optional, default: false): Suggest pattern improvements
- 🔧 `max_patterns` (optional, default: 50): Maximum number of patterns to return (1-100, increased from 20)
- `analysis_depth` (optional, default: "comprehensive"): Depth of analysis ("basic", "comprehensive", "detailed")

**🆕 Wave 7.0 Enhancements:**
- **Increased Capacity**: max_patterns increased to 100 for comprehensive pattern analysis
- **Performance Monitoring**: <15 second response time guarantee
- **Auto-Configuration**: Parameters optimized based on project size and complexity

**Example Usage:**
```python
result = await graph_identify_patterns_tool(
    project_name="my-project",
    pattern_types=["architectural", "behavioral"],
    min_confidence=0.6
)
```

### Graph RAG Best Practices

**Prerequisites:**
1. **Index First**: Always index your project before using Graph RAG tools
2. **Verify Data**: Use `check_index_status` to ensure adequate indexed data

**Performance Tips:**
- Graph building is done on-demand and cached for performance
- Use `force_rebuild_graph=false` (default) to leverage caching
- Monitor memory usage during graph analysis of large projects

**Analysis Strategy:**
1. Start with `graph_analyze_structure_tool` for component overview
2. Use `graph_find_similar_implementations_tool` for cross-project insights
3. Apply `graph_identify_patterns_tool` for architectural analysis

## Function Chain Tools

The Function Chain tools provide specialized analysis of function call chains, execution flows, and path finding within codebases. These tools complement Graph RAG capabilities with focused function-level analysis.

### `trace_function_chain_tool` - Enhanced Function Chain Tracing

🔧 **ENHANCED in Wave 7.0** - Trace complete function chains from an entry point with comprehensive analysis options and improved capacity.

**Parameters:**
- `entry_point` (required): Function/class identifier (breadcrumb or natural language)
- `project_name` (required): Name of the project to analyze
- `direction` (optional, default: "forward"): Tracing direction ("forward", "backward", "bidirectional")
- 🔧 `max_depth` (optional, default: 20): Maximum depth for chain traversal (increased from 10)
- `output_format` (optional, default: "arrow"): Output format ("arrow", "mermaid", "both")
- `include_mermaid` (optional, default: false): Whether to include Mermaid diagram output
- `chain_type` (optional, default: "execution_flow"): Type of chain to trace ("execution_flow", "data_flow", "dependency_chain")
- `min_link_strength` (optional, default: 0.3): Minimum link strength threshold (0.0-1.0)
- `identify_branch_points` (optional, default: true): Whether to identify branch points
- `identify_terminal_points` (optional, default: true): Whether to identify terminal points
- `performance_monitoring` (optional, default: true): Whether to include performance monitoring

**🆕 Wave 7.0 Enhancements:**
- **Increased Depth**: max_depth increased to 20 for deeper chain analysis
- **Performance Monitoring**: <15 second response time guarantee
- **Auto-Configuration**: Parameters optimized based on system capabilities

**Example Usage:**
```python
result = await trace_function_chain_tool(
    entry_point="process_user_data",
    project_name="my-app",
    direction="bidirectional",
    output_format="both"
)
```

### `find_function_path_tool` - Enhanced Function Path Finding

🔧 **ENHANCED in Wave 7.0** - Find the most efficient path between two functions in a codebase with quality metrics and improved capacity.

**Parameters:**
- `start_function` (required): Starting function identifier (breadcrumb or natural language)
- `end_function` (required): Target function identifier (breadcrumb or natural language)
- `project_name` (required): Name of the project to search within
- `strategy` (optional, default: "optimal"): Path finding strategy ("shortest", "optimal", "all")
- 🔧 `max_paths` (optional, default: 10): Maximum number of paths to return (1-20, increased from 3)
- 🔧 `max_depth` (optional, default: 25): Maximum search depth for path finding (increased from 15)
- `include_quality_metrics` (optional, default: true): Whether to calculate path quality metrics
- `output_format` (optional, default: "arrow"): Output format ("arrow", "mermaid", "both")
- `include_mermaid` (optional, default: false): Whether to include Mermaid diagram output
- `min_link_strength` (optional, default: 0.3): Minimum link strength for path inclusion
- `optimize_for` (optional, default: "reliability"): Optimization criteria ("reliability", "directness", "simplicity")

**🆕 Wave 7.0 Enhancements:**
- **Increased Capacity**: max_paths increased to 20, max_depth increased to 25
- **Performance Monitoring**: <15 second response time guarantee
- **Auto-Configuration**: Parameters optimized based on project complexity

**Example Usage:**
```python
result = await find_function_path_tool(
    start_function="authenticate_user",
    end_function="save_to_database",
    project_name="my-app",
    strategy="optimal",
    max_paths=3
)
```

### `analyze_project_chains_tool` - Enhanced Project-Wide Chain Analysis

🔧 **ENHANCED in Wave 7.0** - Analyze function chains across an entire project with comprehensive insights, patterns, and increased capacity.

**Parameters:**
- `project_name` (required): Name of the project to analyze
- `analysis_scope` (optional, default: "full_project"): Scope of analysis ("full_project", "scoped_breadcrumbs", "specific_modules", "function_patterns")
- `breadcrumb_patterns` (optional): List of breadcrumb patterns to focus analysis on
- `analysis_types` (optional): Types of analysis to perform ("complexity_analysis", "hotspot_identification", "pattern_detection", "architectural_analysis")
- 🔧 `max_functions_per_chain` (optional, default: 100): Maximum functions to include per chain (increased from 50)
- `complexity_threshold` (optional, default: 0.7): Complexity threshold for highlighting (0.0-1.0)
- `output_format` (optional, default: "comprehensive"): Output format ("comprehensive", "summary", "detailed")
- `include_mermaid` (optional, default: true): Whether to include Mermaid diagram outputs
- `include_hotspot_analysis` (optional, default: true): Whether to identify complexity hotspots
- `include_refactoring_suggestions` (optional, default: false): Whether to provide refactoring recommendations
- `enable_complexity_weighting` (optional, default: true): Whether to use weighted complexity calculations
- `complexity_weights` (optional): Custom complexity weights (branching_factor, cyclomatic_complexity, etc.)
- 🆕 `max_functions_to_analyze` (optional, default: 5000): Maximum total functions to analyze across project

**🆕 Wave 7.0 Enhancements:**
- **Increased Capacity**: max_functions_per_chain increased to 100, added max_functions_to_analyze (5000)
- **Performance Monitoring**: <15 second response time with progress tracking
- **Auto-Configuration**: Parameters optimized based on project size and system capabilities

**Example Usage:**
```python
result = await analyze_project_chains_tool(
    project_name="my-app",
    analysis_scope="function_patterns",
    breadcrumb_patterns=["*service*", "*controller*"],
    include_hotspot_analysis=True
)
```

### Function Chain Best Practices

**Prerequisites:**
1. **Index First**: Ensure your project is indexed before using function chain tools
2. **Verify Coverage**: Use `analyze_repository_tool` to understand function coverage

**Performance Optimization:**
- All function chain tools meet <2 second response time requirements
- Use appropriate `max_depth` values to balance detail vs performance
- Leverage caching by avoiding `force_rebuild_graph=true` unless necessary

**Analysis Workflow:**
1. **Start with Project Analysis**: Use `analyze_project_chains_tool` for overview
2. **Identify Entry Points**: Find key functions or hotspots to investigate
3. **Trace Specific Chains**: Use `trace_function_chain_tool` for detailed analysis
4. **Find Connections**: Use `find_function_path_tool` to understand relationships

**Output Formats:**
- **Arrow Format**: Simple text-based representation ideal for logs and reports
- **Mermaid Format**: Rich visual diagrams for documentation and presentations
- **Both**: Complete output with text and visual representations

For detailed information about Graph RAG architecture, see [Graph RAG Architecture Guide](GRAPH_RAG_ARCHITECTURE.md).

## 🆕 Wave 7.0 Performance & Error Handling Features

### Performance Monitoring & Timeout Management

All MCP tools in Wave 7.0 include enhanced performance features:

**15-Second Response Time Guarantee:**
- All tools automatically timeout at 15 seconds (configurable)
- Performance monitoring tracks execution time and resource usage
- Automatic timeout handling with graceful error responses
- Progress preservation for long-running operations

**Performance Decorators:**
```python
@with_performance_monitoring(timeout_seconds=15, tool_name="search")
@with_graceful_degradation(service_name="search", fallback_function=simple_search_fallback)
```

**Performance Metrics Included in Responses:**
```json
{
  "_performance": {
    "execution_time_ms": 8432,
    "within_timeout": true,
    "memory_usage_mb": 245.6,
    "cpu_usage_percent": 12.4
  }
}
```

### Graceful Degradation System

**Service Health Tracking:**
- Monitors all services (search, indexing, graph_rag, multi_modal, cache)
- Automatic degradation levels: FULL_SERVICE → PARTIAL_DEGRADATION → MINIMAL_SERVICE → EMERGENCY_MODE
- Error pattern recognition for proactive degradation

**Fallback Mechanisms:**
- Automatic fallback to simpler operations on failure
- Preserved functionality during service degradation
- User-friendly error messages with recovery suggestions

**Error Recovery Features:**
- Automatic service recovery when errors decrease
- Comprehensive error logging and analysis
- Performance alerts and recommendations

### Auto-Configuration System

**Intelligent Parameter Optimization:**
- Analyzes system capabilities (CPU, memory, storage)
- Assesses project characteristics (size, complexity, languages)
- Automatically configures optimal parameters for all tools
- Reduces user configuration burden

**Configuration Areas:**
- Search timeouts and result limits
- Graph analysis depth and complexity thresholds
- Cache settings and memory allocation
- Performance monitoring thresholds

### Backward Compatibility

**Compatibility Guarantee:**
- All existing MCP tool interfaces remain unchanged
- New parameters are optional with sensible defaults
- Comprehensive compatibility testing included
- Zero breaking changes for existing implementations

**Migration Support:**
- Compatibility check tools for validation
- Detailed migration guides for new features
- Gradual adoption path for enhanced capabilities

## Error Handling

All tools include comprehensive error handling with:
- Detailed error messages and recovery suggestions
- Graceful degradation for partial failures
- Automatic retry logic for transient failures
- Progress preservation during interruptions
- 🆕 **Service health monitoring and automatic recovery**
- 🆕 **Performance-based fallback mechanisms**
- 🆕 **Intelligent timeout handling with partial results**

## 🆕 Wave 7.0 Usage Examples

### Multi-Modal Search Examples

**Basic Multi-Modal Search:**
```python
# Enable multi-modal search with automatic mode selection
result = await search(
    query="Find authentication middleware implementations",
    enable_multi_modal=True,
    include_query_analysis=True
)
```

**Manual Mode Selection:**
```python
# Use specific mode for targeted search
result = await multi_modal_search(
    query="Show me all database connection patterns",
    mode="global",  # Focus on relationships
    include_analysis=True,
    include_performance_metrics=True
)
```

**Cross-Project Analysis:**
```python
# Search across multiple projects with performance monitoring
result = await multi_modal_search(
    query="Find similar API endpoint implementations",
    cross_project=True,
    target_projects=["api-service", "user-service"],
    mode="hybrid",
    performance_timeout_seconds=20
)
```

### Auto-Configuration Workflow

**System Setup:**
```python
# Generate optimal configuration for your system
config = await generate_auto_configuration(
    directory="/path/to/large/project",
    usage_pattern="production"
)

# Apply auto-generated settings to search
result = await search(
    query="Find error handling patterns",
    n_results=config["search"]["optimal_n_results"],
    performance_timeout_seconds=config["performance"]["timeout_seconds"]
)
```

### Performance Monitoring Examples

**Dashboard Monitoring:**
```python
# Get comprehensive performance overview
dashboard = await get_performance_dashboard()
print(f"Overall success rate: {dashboard['overall_summary']['success_rate']}%")
print(f"Average response time: {dashboard['overall_summary']['performance']['average_time_ms']}ms")

# Check if performance targets are met
compliance = dashboard['overall_summary']['compliance']['compliance_rate']
if compliance < 95:
    print("⚠️ Performance targets not met - consider optimization")
```

**Service Health Monitoring:**
```python
# Monitor service health and degradation
health = await get_service_health_status()
for service, level in health['service_health'].items():
    if level != 'full_service':
        print(f"⚠️ {service} is degraded: {level}")

# Get recovery suggestions
for service in health['degraded_services']:
    print(f"Recovery suggestions for {service}:")
    # Implement recovery actions
```

### Enhanced Graph RAG Examples

**Comprehensive Structure Analysis:**
```python
# Analyze with auto-configured parameters
result = await graph_analyze_structure_tool(
    breadcrumb="auth.service.AuthenticationService",
    project_name="my-app",
    generate_report=True,
    include_recommendations=True,
    enable_performance_optimization=True
)
```

**Large-Scale Pattern Analysis:**
```python
# Analyze patterns across entire project
patterns = await graph_identify_patterns_tool(
    project_name="large-codebase",
    max_patterns=100,  # Enhanced capacity
    analysis_depth="detailed",
    include_improvements=True
)
```

**Cross-Project Similarity with Enhanced Limits:**
```python
# Find similar implementations with increased capacity
similar = await graph_find_similar_implementations_tool(
    query="cache implementation patterns",
    max_results=100,  # Enhanced from 50
    target_projects=["service-a", "service-b", "service-c"],
    include_implementation_chains=True
)
```

## Tool Integration

**🆕 Wave 7.0 Recommended Workflow:**

1. **System Configuration:**
   ```python
   # Auto-configure optimal settings
   config = await generate_auto_configuration()

   # Verify compatibility
   compatibility = await run_compatibility_check()
   ```

2. **Project Analysis:**
   ```python
   # Analyze repository structure
   analysis = await analyze_repository_tool()

   # Check system health
   health = await health_check()
   ```

3. **Indexing with Performance Monitoring:**
   ```python
   # Index with auto-configured settings
   result = await index_directory(
       incremental=config["indexing"]["use_incremental"],
       batch_size=config["indexing"]["optimal_batch_size"]
   )
   ```

4. **Multi-Modal Search:**
   ```python
   # Search with enhanced capabilities
   results = await search(
       query="Your search query",
       enable_multi_modal=True,
       include_query_analysis=True,
       performance_timeout_seconds=config["performance"]["timeout"]
   )
   ```

5. **Performance Monitoring:**
   ```python
   # Monitor and optimize
   dashboard = await get_performance_dashboard()

   # Get service health
   health_status = await get_service_health_status()
   ```

**Traditional Workflow (Still Supported):**
1. Use `analyze_repository_tool` before indexing to understand scope
2. Run `health_check` to verify system readiness
3. Index with `index_directory` using recommended settings
4. Search with `search` using various query strategies
5. Monitor with progress tools during long operations

## Advanced Cache Management Tools

### Cache Inspection & Statistics Tools

### `inspect_cache_state_tool` - Cache State Inspection

Inspect current cache state with comprehensive debugging information.

**Returns:** Detailed cache state analysis including memory usage, hit rates, and performance metrics.

### `debug_cache_key_tool` - Cache Key Debugging

Debug specific cache keys across all services with detailed analysis.

**Parameters:**
- `cache_key` (required): The cache key to debug
- `include_content` (optional, default: false): Whether to include cached content in response

**Returns:** Detailed debugging information for the specified cache key.

### `get_cache_invalidation_stats_tool` - Invalidation Statistics

Get comprehensive cache invalidation statistics and metrics.

**Returns:** Invalidation statistics including frequencies, patterns, and performance impact.

### `get_comprehensive_cache_stats_tool` - Comprehensive Statistics

Get comprehensive cache statistics across all services and cache types.

**Returns:** Complete cache statistics including hit rates, memory usage, and performance metrics.

### `generate_cache_report_tool` - Cache Performance Reports

Generate detailed cache performance reports for analysis.

**Parameters:**
- `report_type` (optional, default: "comprehensive"): Type of report to generate
- `time_range_hours` (optional, default: 24): Time range for report data
- `include_recommendations` (optional, default: true): Include optimization recommendations

**Returns:** Detailed cache performance report with analysis and recommendations.

### Cache Control & Invalidation Tools

### `manual_invalidate_file_cache_tool` - File Cache Invalidation

Manually invalidate cache entries for specific files.

**Parameters:**
- `file_path` (required): Path to the file to invalidate
- `reason` (optional, default: "manual_invalidation"): Reason for invalidation
- `cascade` (optional, default: true): Whether to cascade invalidation to dependent caches
- `project_name` (optional): Project name for scoped invalidation

**Returns:** Invalidation results and statistics.

### `manual_invalidate_project_cache_tool` - Project Cache Invalidation

Invalidate all cache entries for a specific project.

**Parameters:**
- `project_name` (required): Name of the project to invalidate
- `reason` (optional, default: "manual_invalidation"): Reason for invalidation
- `cascade` (optional, default: true): Whether to cascade invalidation

**Returns:** Project invalidation results and statistics.

### `manual_invalidate_cache_keys_tool` - Specific Key Invalidation

Invalidate specific cache keys across services.

**Parameters:**
- `cache_keys` (required): List of cache keys to invalidate
- `reason` (optional, default: "manual_invalidation"): Reason for invalidation
- `services` (optional): List of specific services to target

**Returns:** Key-by-key invalidation results.

### `manual_invalidate_cache_pattern_tool` - Pattern-based Invalidation

Invalidate cache keys matching specific patterns.

**Parameters:**
- `pattern` (required): Pattern to match cache keys (supports wildcards)
- `reason` (optional, default: "manual_invalidation"): Reason for invalidation
- `max_keys` (optional, default: 1000): Maximum number of keys to invalidate

**Returns:** Pattern invalidation results and affected keys count.

### `invalidate_chunks_tool` - Chunk-specific Invalidation

Invalidate specific chunks within files for granular cache control.

**Parameters:**
- `file_path` (required): Path to the file containing chunks
- `chunk_ids` (optional): Specific chunk IDs to invalidate
- `chunk_types` (optional): Types of chunks to invalidate

**Returns:** Chunk invalidation results and statistics.

### Cache Configuration Tools

### `get_cache_configuration_tool` - Configuration Retrieval

Get current cache configuration across all services.

**Returns:** Complete cache configuration including TTL settings, memory limits, and policies.

### `update_cache_configuration_tool` - Configuration Updates

Update cache settings and policies.

**Parameters:**
- `configuration` (required): Dictionary of configuration updates
- `services` (optional): List of specific services to update
- `validate_only` (optional, default: false): Only validate configuration without applying

**Returns:** Configuration update results and validation status.

### `export_cache_configuration_tool` - Configuration Export

Export current cache configuration to file for backup or sharing.

**Parameters:**
- `export_path` (required): Path to export configuration file
- `include_runtime_stats` (optional, default: false): Include runtime statistics
- `format` (optional, default: "json"): Export format ("json", "yaml")

**Returns:** Export operation results and file location.

### `import_cache_configuration_tool` - Configuration Import

Import cache configuration from external file.

**Parameters:**
- `import_path` (required): Path to configuration file to import
- `validate_only` (optional, default: false): Only validate without applying
- `merge_strategy` (optional, default: "replace"): How to merge with existing config

**Returns:** Import operation results and applied changes.

### Cache Monitoring & Alerts

### `configure_cache_alerts_tool` - Alert Configuration

Configure monitoring alerts for cache performance and health.

**Parameters:**
- `alert_rules` (required): Dictionary of alert rules and thresholds
- `notification_channels` (optional): List of notification channels
- `enable_alerts` (optional, default: true): Whether to enable alerts

**Returns:** Alert configuration results and validation status.

### `get_cache_alerts_tool` - Recent Alerts

Get recent cache alerts and notifications.

**Parameters:**
- `time_range_hours` (optional, default: 24): Time range for alert history
- `severity_filter` (optional): Filter by alert severity
- `service_filter` (optional): Filter by specific services

**Returns:** List of recent cache alerts with details and recommendations.

### Cache Optimization Tools

### `preload_embedding_cache_tool` - Embedding Cache Preloading

Preload embedding cache with commonly used vectors.

**Parameters:**
- `project_name` (optional): Specific project to preload
- `preload_strategy` (optional, default: "smart"): Preloading strategy
- `max_embeddings` (optional, default: 1000): Maximum embeddings to preload

**Returns:** Preloading results and cache performance improvement.

### `preload_search_cache_tool` - Search Cache Preloading

Preload search result cache with common queries.

**Parameters:**
- `common_queries` (optional): List of queries to preload
- `project_name` (optional): Specific project context
- `preload_depth` (optional, default: 3): Depth of related searches to preload

**Returns:** Search cache preloading results and performance metrics.

### `backup_cache_data_tool` - Cache Backup

Create comprehensive backup of cache data.

**Parameters:**
- `backup_path` (required): Path for backup storage
- `include_content` (optional, default: false): Whether to backup cached content
- `compress` (optional, default: true): Whether to compress backup

**Returns:** Backup operation results and backup file information.

### `restore_cache_data_tool` - Cache Restoration

Restore cache data from backup.

**Parameters:**
- `backup_path` (required): Path to backup file
- `selective_restore` (optional): List of specific caches to restore
- `merge_strategy` (optional, default: "replace"): How to merge with existing data

**Returns:** Restoration results and recovered cache statistics.

### `migrate_cache_data_tool` - Cache Migration

Migrate cache data between different configurations or versions.

**Parameters:**
- `source_config` (required): Source cache configuration
- `target_config` (required): Target cache configuration
- `migration_strategy` (optional, default: "safe"): Migration strategy

**Returns:** Migration results and data transfer statistics.

### `get_migration_status_tool` - Migration Status

Get status of ongoing or recent cache migrations.

**Returns:** Current migration status, progress, and estimated completion time.

## File Monitoring Tools

Real-time file monitoring tools for automatic cache invalidation and project synchronization.

### `setup_project_monitoring` - Project Monitoring Setup

Set up real-time file monitoring for a project with automatic cache invalidation.

**Parameters:**
- `project_name` (required): Name of the project to monitor
- `root_directory` (required): Root directory of the project
- `auto_detect` (optional, default: true): Auto-detect project characteristics
- `file_patterns` (optional): File patterns to monitor (e.g., ["*.py", "*.js"])
- `exclude_patterns` (optional): Patterns to exclude (e.g., ["*.pyc", "node_modules/*"])
- `polling_interval` (optional, default: 5.0): Polling interval in seconds
- `enable_real_time` (optional, default: true): Enable real-time monitoring
- `enable_polling` (optional, default: true): Enable polling-based monitoring

**Returns:** Monitoring setup results and configuration details.

### `remove_project_monitoring` - Remove Monitoring

Remove file monitoring for a project.

**Parameters:**
- `project_name` (required): Name of the project to stop monitoring
- `cleanup_resources` (optional, default: true): Whether to cleanup monitoring resources

**Returns:** Monitoring removal results and cleanup status.

### `get_monitoring_status` - Monitoring Status

Get current monitoring status for all projects.

**Parameters:**
- `project_name` (optional): Specific project to check
- `include_statistics` (optional, default: true): Include monitoring statistics

**Returns:** Comprehensive monitoring status and performance metrics.

### `trigger_manual_scan` - Manual File Scan

Trigger manual file system scan for changes.

**Parameters:**
- `project_name` (required): Project to scan
- `scan_type` (optional, default: "incremental"): Type of scan ("full", "incremental")
- `force_invalidation` (optional, default: false): Force cache invalidation for all files

**Returns:** Scan results and detected changes.

### `configure_monitoring_mode` - Monitoring Configuration

Configure global monitoring mode and settings.

**Parameters:**
- `monitoring_mode` (required): Global monitoring mode ("real_time", "polling", "hybrid", "disabled")
- `global_settings` (optional): Global monitoring settings
- `apply_to_existing` (optional, default: true): Apply to existing project monitors

**Returns:** Configuration results and affected projects.

### `update_project_monitoring_config` - Update Project Config

Update monitoring configuration for a specific project.

**Parameters:**
- `project_name` (required): Project to update
- `config_updates` (required): Configuration updates to apply
- `restart_monitoring` (optional, default: true): Whether to restart monitoring

**Returns:** Configuration update results and new settings.

### `trigger_file_invalidation` - Manual File Invalidation

Manually trigger file cache invalidation through monitoring system.

**Parameters:**
- `file_path` (required): Path to file to invalidate
- `invalidation_reason` (optional, default: "manual_trigger"): Reason for invalidation
- `cascade` (optional, default: true): Whether to cascade invalidation

**Returns:** File invalidation results and affected caches.

## Cascade Invalidation Tools

Advanced cascade invalidation management for handling complex cache dependencies.

### `add_cascade_dependency_rule` - Dependency Rule Creation

Add cascade invalidation dependency rules for automatic cache management.

**Parameters:**
- `source_pattern` (required): Pattern matching source cache keys (supports wildcards)
- `target_pattern` (required): Pattern matching dependent cache keys
- `dependency_type` (optional, default: "file_content"): Type of dependency
- `cascade_strategy` (optional, default: "immediate"): Cascade strategy
- `condition` (optional): Optional condition for dependency
- `metadata` (optional): Additional metadata for the rule

**Returns:** Rule creation results and validation status.

### `add_explicit_cascade_dependency` - Explicit Dependencies

Add explicit cascade dependencies between specific cache keys.

**Parameters:**
- `source_key` (required): Source cache key
- `target_keys` (required): List of dependent cache keys
- `dependency_strength` (optional, default: 1.0): Strength of dependency (0.0-1.0)
- `bidirectional` (optional, default: false): Whether dependency is bidirectional

**Returns:** Explicit dependency creation results.

### `get_cascade_stats` - Cascade Statistics

Get comprehensive cascade invalidation statistics and performance metrics.

**Parameters:**
- `time_range_hours` (optional, default: 24): Time range for statistics
- `include_patterns` (optional, default: true): Include pattern-based stats
- `include_performance` (optional, default: true): Include performance metrics

**Returns:** Detailed cascade invalidation statistics and analysis.

### `get_dependency_graph` - Dependency Graph Analysis

Get visual representation of cache dependency relationships.

**Parameters:**
- `graph_format` (optional, default: "json"): Output format ("json", "dot", "mermaid")
- `max_depth` (optional, default: 5): Maximum depth for graph traversal
- `include_weights` (optional, default: true): Include dependency weights

**Returns:** Cache dependency graph in requested format.

### `detect_circular_dependencies` - Circular Dependency Detection

Detect and analyze circular dependencies in cache invalidation rules.

**Parameters:**
- `include_resolution` (optional, default: true): Include resolution suggestions
- `max_cycles` (optional, default: 10): Maximum number of cycles to detect

**Returns:** Circular dependency analysis and resolution recommendations.

### `test_cascade_invalidation` - Cascade Testing

Test cascade invalidation behavior for specific cache keys.

**Parameters:**
- `test_key` (required): Cache key to test invalidation for
- `dry_run` (optional, default: true): Whether to perform dry run only
- `max_cascade_depth` (optional, default: 10): Maximum cascade depth to test

**Returns:** Cascade test results and affected keys analysis.

### `configure_cascade_settings` - Cascade Configuration

Configure global cascade invalidation settings and policies.

**Parameters:**
- `cascade_settings` (required): Dictionary of cascade configuration updates
- `apply_immediately` (optional, default: true): Whether to apply settings immediately
- `validate_rules` (optional, default: true): Validate existing rules with new settings

**Returns:** Configuration results and rule validation status.

## Progress Monitoring Tools

### `get_indexing_progress_tool` - Real-time Indexing Progress

Get current progress of any ongoing indexing operations with detailed metrics.

**Returns:**
- Real-time progress updates including percentage completion
- Estimated time to completion (ETA)
- Processing rate (files/second, chunks/second)
- Memory usage during indexing
- Error counts and recovery status

### `reset_chunking_metrics_tool` - Reset Performance Metrics

Reset session-specific chunking performance metrics for fresh measurement.

**Returns:**
- Confirmation of metrics reset
- Previous metrics summary before reset
- Timestamp of reset operation
