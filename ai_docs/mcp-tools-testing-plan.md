# Comprehensive Testing Plan for Codebase RAG MCP Tools

## Overview
This document provides a systematic testing plan for all 75+ MCP tools in the codebase RAG system. This plan is designed to be executed by LLMs for automated regression testing and tool validation.

## Testing Categories & Tool Inventory

### 1. Core Health & System Tools (2 tools)
- `health_check_tool` - Verify Qdrant and Ollama connectivity, system resources
- `get_cache_health_status_tool` - Check comprehensive cache health across all services

### 2. Indexing & Repository Analysis Tools (9 tools)
- `index_directory` - Index files with smart existing data detection
- `check_index_status_tool` - Verify current index state and recommendations
- `analyze_repository_tool` - Get repository statistics and complexity analysis
- `get_file_filtering_stats_tool` - Check file filtering statistics and exclusions
- `get_chunking_metrics_tool` - Review intelligent chunking performance metrics
- `diagnose_parser_health_tool` - Test Tree-sitter parser functionality
- `get_indexing_progress_tool` - Check real-time indexing progress
- `reset_chunking_metrics_tool` - Reset performance metrics
- `check_index_status` - Duplicate check for index status

### 3. Search & Query Tools (5 tools)
- `search` - Natural language search with multiple modes (semantic, keyword, hybrid, mix/multi-modal)
- `multi_modal_search` - Advanced multi-modal retrieval with LightRAG-inspired strategies
- `analyze_query_features` - Query analysis and mode recommendation tool
- `get_retrieval_mode_performance` - Performance metrics for multi-modal retrieval modes

### 4. Project Management Tools (6 tools)
- `get_project_info_tool` - Get current project information and collections
- `list_indexed_projects_tool` - List all projects with indexed data
- `clear_project_data_tool` - Clear project data (use with caution)
- `get_file_metadata_tool` - Get specific file metadata and indexing status
- `clear_file_metadata_tool` - Clear file-specific metadata (use with caution)
- `reindex_file_tool` - Reindex specific files

### 5. Cache Management Tools (24 tools)

#### Cache Inspection & Statistics (10 tools)
- `inspect_cache_state_tool` - Inspect current cache state with debugging info
- `debug_cache_key_tool` - Debug specific cache keys across services
- `get_cache_invalidation_stats_tool` - Get invalidation statistics and metrics
- `get_comprehensive_cache_stats_tool` - Comprehensive cache statistics
- `generate_cache_report_tool` - Generate performance reports
- `manual_invalidate_file_cache_tool` - Manually invalidate file cache entries
- `manual_invalidate_project_cache_tool` - Invalidate project cache entries
- `manual_invalidate_cache_keys_tool` - Invalidate specific cache keys
- `manual_invalidate_cache_pattern_tool` - Invalidate keys matching patterns
- `invalidate_chunks_tool` - Invalidate specific chunks within files

#### Cache Configuration & Control (7 tools)
- `get_cache_configuration_tool` - Get current cache configuration
- `update_cache_configuration_tool` - Update cache settings
- `export_cache_configuration_tool` - Export configuration to file
- `import_cache_configuration_tool` - Import configuration from file
- `configure_cache_alerts_tool` - Configure monitoring alerts
- `get_cache_alerts_tool` - Get recent cache alerts
- `get_project_invalidation_policy_tool` - Get project invalidation policies
- `set_project_invalidation_policy_tool` - Set invalidation policies

#### Cache Optimization & Migration (7 tools)
- `clear_all_caches_tool` - Clear all caches (DESTRUCTIVE - require confirmation)
- `warm_cache_for_project_tool` - Warm up caches for projects
- `preload_embedding_cache_tool` - Preload embedding cache
- `preload_search_cache_tool` - Preload search result cache
- `optimize_cache_performance_tool` - Analyze and optimize performance
- `backup_cache_data_tool` - Create cache backups
- `restore_cache_data_tool` - Restore from cache backups
- `migrate_cache_data_tool` - Migrate cache data between configurations
- `get_migration_status_tool` - Get migration status

### 6. File Monitoring Tools (7 tools)
- `setup_project_monitoring` - Set up real-time file monitoring
- `remove_project_monitoring` - Remove file monitoring
- `get_monitoring_status` - Get monitoring status for projects
- `trigger_manual_scan` - Trigger manual file system scan
- `configure_monitoring_mode` - Configure global monitoring mode
- `update_project_monitoring_config` - Update monitoring configuration
- `trigger_file_invalidation` - Manually trigger file cache invalidation

### 7. Cascade Invalidation Tools (10 tools)
- `add_cascade_dependency_rule` - Add dependency rules for cascade invalidation
- `add_explicit_cascade_dependency` - Add explicit dependencies
- `get_cascade_stats` - Get cascade invalidation statistics
- `get_dependency_graph` - Get cache dependency graph
- `detect_circular_dependencies` - Detect circular dependencies
- `test_cascade_invalidation` - Test cascade invalidation for keys
- `configure_cascade_settings` - Configure cascade settings

### 8. Graph RAG Tools (3 tools)
- `graph_analyze_structure_tool` - Enhanced code structure analysis with performance optimizations
- `graph_find_similar_implementations_tool` - Cross-project similarity search with semantic analysis
- `graph_identify_patterns_tool` - Architectural pattern recognition with increased capacity

### 9. Function Chain Analysis Tools (3 tools)
- `trace_function_chain_tool` - Trace complete function chains with multiple directions and output formats
- `find_function_path_tool` - Find optimal paths between functions with quality metrics
- `analyze_project_chains_tool` - Project-wide chain analysis with complexity insights

### 10. Wave 7.0 Performance & Configuration Tools (5 tools)
- `get_auto_configuration` - Generate optimal configuration based on system capabilities
- `check_tool_compatibility` - Backward compatibility verification for all MCP tools
- `get_performance_dashboard_tool` - Comprehensive performance monitoring dashboard
- `get_service_health_status_tool` - Service health monitoring with error tracking

## Testing Strategy

### Phase 1: Read-Only System Verification
**Priority**: High
**Tools to test**: Health checks, status queries, information retrieval

```python
# Example test sequence
test_tools = [
    "health_check_tool",
    "get_cache_health_status_tool",
    "check_index_status_tool",
    "get_project_info_tool",
    "list_indexed_projects_tool",
    "get_monitoring_status",
    "get_cascade_stats",
    "get_dependency_graph"
]
```

### Phase 2: Search & Query Functionality
**Priority**: High
**Focus**: Test core search capabilities

```python
# Test different search modes including multi-modal
search_tests = [
    {"query": "cache invalidation", "search_mode": "hybrid", "n_results": 3},
    {"query": "function definition", "search_mode": "semantic", "n_results": 5},
    {"query": "class CodeParser", "search_mode": "keyword", "n_results": 2},
    {"query": "error handling", "search_mode": "hybrid", "include_context": True},
    {"query": "graph analysis", "enable_multi_modal": True, "multi_modal_mode": "mix", "n_results": 3}
]

# Test multi-modal search tools
multi_modal_tests = [
    {"query": "function chain analysis", "mode": "local", "n_results": 3},
    {"query": "error patterns", "mode": "global", "n_results": 5},
    {"query": "cache implementation", "mode": "hybrid", "include_analysis": True}
]

# Test query analysis
query_analysis_tests = [
    "error handling patterns in python",
    "find database connection code",
    "React components with hooks"
]
```

### Phase 3: Cache System Inspection
**Priority**: Medium
**Focus**: Cache health and statistics without modifications

```python
# Safe cache inspection tests
cache_inspection_tests = [
    "get_comprehensive_cache_stats_tool",
    "generate_cache_report_tool",
    "get_cache_configuration_tool",
    "inspect_cache_state_tool"
]
```

### Phase 4: Repository Analysis
**Priority**: Medium
**Focus**: Analysis tools that don't modify data

```python
# Repository analysis tests
analysis_tests = [
    "analyze_repository_tool",
    "get_file_filtering_stats_tool",
    "get_chunking_metrics_tool",
    "diagnose_parser_health_tool"
]
```


### Phase 5: Graph RAG & Function Chain Analysis
**Priority**: Medium
**Focus**: Advanced code analysis capabilities

```python
# Graph RAG tests
graph_rag_tests = [
    ("graph_analyze_structure_tool", {"breadcrumb": "cache.service", "project_name": "test_project", "analysis_type": "overview"}),
    ("graph_find_similar_implementations_tool", {"query": "cache implementation", "max_results": 3}),
    ("graph_identify_patterns_tool", {"project_name": "test_project", "max_patterns": 5})
]

# Function Chain tests
function_chain_tests = [
    ("trace_function_chain_tool", {"entry_point": "search", "project_name": "test_project", "max_depth": 3}),
    ("find_function_path_tool", {"start_function": "search", "end_function": "cache", "project_name": "test_project", "max_paths": 2}),
    ("analyze_project_chains_tool", {"project_name": "test_project", "max_functions_per_chain": 5})
]
```

### Phase 6: Wave 7.0 Performance & Configuration Tools
**Priority**: Medium
**Focus**: New performance monitoring and auto-configuration

```python
# Performance and configuration tests
performance_tests = [
    ("get_auto_configuration", {"directory": ".", "usage_pattern": "balanced"}),
    ("check_tool_compatibility", {}),
    ("get_performance_dashboard_tool", {}),
    ("get_service_health_status_tool", {})
]
```

### Phase 7: Destructive Operations Testing
**Priority**: Low, Caution Required
**Focus**: Test with dry-run or test data only

**⚠️ DESTRUCTIVE TOOLS (use with extreme caution):**
- `clear_all_caches_tool` (requires confirm=True)
- `clear_project_data_tool`
- `clear_file_metadata_tool`

## Automated Testing Script Template

```python
async def run_mcp_tools_regression_test():
    """
    Comprehensive MCP tools regression test.
    Execute this with codebase RAG MCP tools available.
    """

    test_results = {
        "total_tools": 75,
        "tested_tools": 0,
        "successful_tools": 0,
        "failed_tools": 0,
        "errors": []
    }

    # Phase 1: System Health
    health_tools = [
        ("health_check_tool", {}),
        ("get_cache_health_status_tool", {})
    ]

    for tool_name, params in health_tools:
        try:
            result = await globals()[f"mcp__codebase-rag-mcp__{tool_name}"](**params)
            test_results["tested_tools"] += 1
            if "error" not in result:
                test_results["successful_tools"] += 1
            else:
                test_results["failed_tools"] += 1
                test_results["errors"].append(f"{tool_name}: {result.get('error', 'Unknown error')}")
        except Exception as e:
            test_results["tested_tools"] += 1
            test_results["failed_tools"] += 1
            test_results["errors"].append(f"{tool_name}: {str(e)}")

    # Phase 2: Project Information
    project_tools = [
        ("get_project_info_tool", {"directory": "."}),
        ("list_indexed_projects_tool", {}),
        ("check_index_status_tool", {"directory": "."})
    ]

    for tool_name, params in project_tools:
        try:
            result = await globals()[f"mcp__codebase-rag-mcp__{tool_name}"](**params)
            test_results["tested_tools"] += 1
            if "error" not in result:
                test_results["successful_tools"] += 1
            else:
                test_results["failed_tools"] += 1
                test_results["errors"].append(f"{tool_name}: {result.get('error', 'Unknown error')}")
        except Exception as e:
            test_results["tested_tools"] += 1
            test_results["failed_tools"] += 1
            test_results["errors"].append(f"{tool_name}: {str(e)}")

    # Phase 3: Search Tests
    search_tests = [
        {"query": "cache invalidation", "n_results": 3, "search_mode": "hybrid"},
        {"query": "function", "n_results": 2, "search_mode": "semantic"},
        {"query": "class", "n_results": 1, "search_mode": "keyword"}
    ]

    for search_params in search_tests:
        try:
            result = await mcp__codebase_rag_mcp__search(**search_params)
            test_results["tested_tools"] += 1
            if "error" not in result:
                test_results["successful_tools"] += 1
            else:
                test_results["failed_tools"] += 1
                test_results["errors"].append(f"search: {result.get('error', 'Unknown error')}")
        except Exception as e:
            test_results["tested_tools"] += 1
            test_results["failed_tools"] += 1
            test_results["errors"].append(f"search: {str(e)}")

    # Calculate success rate
    test_results["success_rate"] = (
        test_results["successful_tools"] / test_results["tested_tools"] * 100
        if test_results["tested_tools"] > 0 else 0
    )

    return test_results

# Usage
# test_results = await run_mcp_tools_regression_test()
# print(f"Success Rate: {test_results['success_rate']:.1f}%")
# print(f"Successful: {test_results['successful_tools']}/{test_results['tested_tools']}")
```

## Expected Baseline Results

### Pre-Fix Baseline (Reference)
- **Total Tools**: 75+
- **Success Rate**: 30-35%
- **Common Errors**:
  - `attempted relative import beyond top-level package`
  - `'ImportError' object has no attribute '__name__'`
  - Module loading failures

### Post-Fix Target
- **Target Success Rate**: 85-90%
- **Expected Working Categories**:
  - Core Health & System Tools: 100%
  - Project Management Tools: 100%
  - Search & Query Tools: 100%
  - File Monitoring Tools: 90%+
  - Cache Management Tools: 80%+

## Test Execution Guidelines

### For LLM Execution
1. **Start with read-only operations** - Never begin with destructive tools
2. **Test in phases** - Complete each phase before moving to the next
3. **Record all results** - Document both successes and failures
4. **Safe failure handling** - Continue testing even if individual tools fail
5. **Report format** - Use structured output for easy analysis

### Safety Protocols
1. **Avoid destructive operations** unless explicitly testing with test data
2. **Use dry-run mode** when available
3. **Backup important data** before testing cache clearing operations
4. **Test on non-production systems** when possible

### Regression Testing Schedule
- **After major fixes**: Run full test suite
- **Before releases**: Run Phase 1-4 tests
- **Weekly maintenance**: Run Phase 1-2 tests
- **After dependency updates**: Run full test suite

## Test Result Analysis

### Success Metrics
- **Tool Availability**: Tools can be called without import errors
- **Error Handling**: Graceful error responses instead of exceptions
- **Data Integrity**: Tools return expected data structures
- **Performance**: Reasonable response times

### Failure Classification
1. **Import Errors**: Module loading failures
2. **Runtime Errors**: Exceptions during execution
3. **Data Errors**: Malformed responses or missing data
4. **Performance Issues**: Timeouts or slow responses

## Integration with CI/CD

This testing plan can be integrated into automated workflows:

```yaml
# Example GitHub Actions workflow
name: MCP Tools Regression Test
on: [push, pull_request]
jobs:
  test-mcp-tools:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: uv sync
      - name: Run MCP Tools Test
        run: python -c "import ai_docs.mcp_tools_test; ai_docs.mcp_tools_test.run_regression_test()"
```

## Future Enhancements

1. **Performance Benchmarking**: Add timing and resource usage metrics
2. **Load Testing**: Test tools under concurrent usage
3. **Data Validation**: Verify response schema compliance
4. **Integration Testing**: Test tool interactions and dependencies
5. **Mock Testing**: Test with simulated failure conditions

---

*This document should be updated whenever new tools are added or existing tools are modified. The baseline results should be updated after major fixes or improvements.*
