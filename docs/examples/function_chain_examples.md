# Function Chain MCP Tools - Examples and Tutorial

This document provides practical examples and tutorials for using the Function Chain MCP tools to analyze code execution flows, find optimal paths between functions, and understand project-wide patterns.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Examples](#basic-examples)
3. [Advanced Use Cases](#advanced-use-cases)
4. [Real-World Scenarios](#real-world-scenarios)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

Before using Function Chain tools, ensure your project is indexed:

```python
# Index your project first
await index_directory(
    directory=".",
    recursive=True,
    incremental=False
)

# Verify indexing status
status = await check_index_status(".")
print(f"Project indexed: {status['has_indexed_data']}")
```

### Basic Workflow

1. **Project Analysis** → **Function Tracing** → **Path Finding**
2. Start broad, then drill down to specific functions
3. Use appropriate output formats for your needs

## Basic Examples

### Example 1: Trace a Function Chain

**Scenario**: Understanding how user authentication flows through your application

```python
# Trace forward from main authentication function
result = await trace_function_chain_tool(
    entry_point="authenticate_user",
    project_name="my-app",
    direction="forward",
    max_depth=8,
    output_format="both"
)

print("Chain found:")
print(result["arrow_format"])
print("\nMermaid diagram:")
print(result["mermaid_format"])
```

**Sample Output**:
```
authenticate_user => validate_credentials --[calls]--> check_database =>
  update_session --[calls]--> log_login_event => send_notification
```

### Example 2: Find Path Between Functions

**Scenario**: Understanding how data flows from input validation to database storage

```python
# Find optimal path between two functions
result = await find_function_path_tool(
    start_function="validate_input_data",
    end_function="save_to_database",
    project_name="my-app",
    strategy="optimal",
    max_paths=3,
    include_quality_metrics=True
)

# Review path quality
for i, path in enumerate(result["paths"]):
    print(f"Path {i+1}: {path['arrow_format']}")
    print(f"Quality Score: {path['quality']['overall_score']:.2f}")
    print(f"Reliability: {path['quality']['reliability_score']:.2f}")
    print("---")
```

**Sample Output**:
```
Path 1: validate_input_data => sanitize_data => transform_data => save_to_database
Quality Score: 0.85
Reliability: 0.92

Path 2: validate_input_data => cache_check => save_to_database
Quality Score: 0.78
Reliability: 0.88
```

### Example 3: Analyze Project Complexity

**Scenario**: Getting an overview of your project's function complexity patterns

```python
# Analyze entire project for complexity patterns
result = await analyze_project_chains_tool(
    project_name="my-app",
    analysis_scope="full_project",
    include_hotspot_analysis=True,
    complexity_threshold=0.7,
    output_format="comprehensive"
)

print("Complexity Hotspots:")
for hotspot in result["hotspots"]:
    print(f"- {hotspot['breadcrumb']}: {hotspot['complexity_score']:.2f}")

print("\nArchitectural Patterns:")
for pattern in result["patterns"]:
    print(f"- {pattern['pattern_type']}: {pattern['description']}")
```

## Advanced Use Cases

### Use Case 1: API Endpoint Analysis

**Scenario**: Analyzing all paths from API endpoints to understand system complexity

```python
async def analyze_api_endpoints():
    # Step 1: Find all API endpoints
    project_analysis = await analyze_project_chains_tool(
        project_name="api-service",
        analysis_scope="function_patterns",
        breadcrumb_patterns=["*api*", "*endpoint*", "*handler*"],
        include_hotspot_analysis=True
    )

    # Step 2: Trace each endpoint's execution flow
    endpoint_chains = {}
    for endpoint in project_analysis["matching_functions"][:5]:  # Top 5
        chain = await trace_function_chain_tool(
            entry_point=endpoint["breadcrumb"],
            project_name="api-service",
            direction="forward",
            max_depth=10,
            identify_branch_points=True
        )
        endpoint_chains[endpoint["breadcrumb"]] = chain

    # Step 3: Find connections between endpoints
    endpoint_paths = {}
    endpoints = list(endpoint_chains.keys())
    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            path = await find_function_path_tool(
                start_function=endpoints[i],
                end_function=endpoints[j],
                project_name="api-service",
                strategy="shortest",
                max_paths=1
            )
            if path["paths"]:
                endpoint_paths[f"{endpoints[i]} -> {endpoints[j]}"] = path

    return {
        "analysis": project_analysis,
        "chains": endpoint_chains,
        "connections": endpoint_paths
    }

# Run the analysis
api_analysis = await analyze_api_endpoints()
```

### Use Case 2: Error Handling Analysis

**Scenario**: Understanding error propagation patterns through your codebase

```python
async def analyze_error_handling():
    # Find all error handling functions
    error_analysis = await analyze_project_chains_tool(
        project_name="my-app",
        analysis_scope="function_patterns",
        breadcrumb_patterns=["*error*", "*exception*", "*handle*", "*catch*"],
        analysis_types=["pattern_detection", "architectural_analysis"]
    )

    # Trace error handling chains
    error_chains = []
    for error_func in error_analysis["matching_functions"][:10]:
        # Trace backward to see what triggers errors
        backward_chain = await trace_function_chain_tool(
            entry_point=error_func["breadcrumb"],
            project_name="my-app",
            direction="backward",
            max_depth=6,
            chain_type="execution_flow"
        )

        # Trace forward to see error handling flow
        forward_chain = await trace_function_chain_tool(
            entry_point=error_func["breadcrumb"],
            project_name="my-app",
            direction="forward",
            max_depth=6,
            chain_type="execution_flow"
        )

        error_chains.append({
            "function": error_func["breadcrumb"],
            "triggers": backward_chain,
            "handling": forward_chain
        })

    return error_chains

# Analyze error patterns
error_patterns = await analyze_error_handling()
for pattern in error_patterns:
    print(f"Error Function: {pattern['function']}")
    print(f"Triggered by: {pattern['triggers'].get('arrow_format', 'No triggers found')}")
    print(f"Handles via: {pattern['handling'].get('arrow_format', 'No handling found')}")
    print("---")
```

### Use Case 3: Performance Critical Path Analysis

**Scenario**: Identifying performance-critical execution paths

```python
async def analyze_performance_paths():
    # Find performance-critical functions
    perf_analysis = await analyze_project_chains_tool(
        project_name="my-app",
        analysis_scope="function_patterns",
        breadcrumb_patterns=["*db*", "*cache*", "*api*", "*process*"],
        complexity_threshold=0.6,
        include_hotspot_analysis=True
    )

    # Analyze paths between critical functions with quality focus
    critical_functions = [f["breadcrumb"] for f in perf_analysis["hotspots"][:5]]

    performance_paths = []
    for i in range(len(critical_functions)):
        for j in range(i+1, len(critical_functions)):
            path_analysis = await find_function_path_tool(
                start_function=critical_functions[i],
                end_function=critical_functions[j],
                project_name="my-app",
                strategy="all",
                max_paths=5,
                optimize_for="directness",  # Focus on performance
                include_quality_metrics=True
            )

            # Filter paths by performance criteria
            fast_paths = [
                p for p in path_analysis["paths"]
                if p["quality"]["directness_score"] > 0.7
            ]

            if fast_paths:
                performance_paths.append({
                    "start": critical_functions[i],
                    "end": critical_functions[j],
                    "paths": fast_paths
                })

    return performance_paths

# Analyze performance-critical paths
perf_paths = await analyze_performance_paths()
```

## Real-World Scenarios

### Scenario 1: Debugging a Complex Bug

**Problem**: A user reports data corruption, but you're not sure where in the pipeline it occurs.

**Solution**: Use function chain tools to trace the data flow and identify potential corruption points.

```python
async def debug_data_corruption():
    # Start from the user-reported entry point
    data_flow = await trace_function_chain_tool(
        entry_point="process_user_data",
        project_name="data-pipeline",
        direction="forward",
        max_depth=15,
        identify_branch_points=True,
        identify_terminal_points=True
    )

    # Look for branch points where data could be corrupted
    branch_points = data_flow.get("branch_points", [])

    # Trace each branch to understand parallel processing
    branch_analyses = []
    for branch in branch_points:
        branch_trace = await trace_function_chain_tool(
            entry_point=branch["breadcrumb"],
            project_name="data-pipeline",
            direction="forward",
            max_depth=8
        )
        branch_analyses.append(branch_trace)

    return {
        "main_flow": data_flow,
        "branch_points": branch_points,
        "branch_analyses": branch_analyses
    }

debug_result = await debug_data_corruption()
print("Main data flow:")
print(debug_result["main_flow"]["arrow_format"])
print(f"\nFound {len(debug_result['branch_points'])} potential corruption points")
```

### Scenario 2: Refactoring Legacy Code

**Problem**: You need to refactor a legacy module but want to understand its dependencies first.

**Solution**: Use comprehensive analysis to understand the module's role in the system.

```python
async def plan_legacy_refactoring(legacy_module_pattern):
    # Analyze the legacy module comprehensively
    legacy_analysis = await analyze_project_chains_tool(
        project_name="legacy-system",
        analysis_scope="scoped_breadcrumbs",
        breadcrumb_patterns=[legacy_module_pattern],
        include_refactoring_suggestions=True,
        analysis_types=["complexity_analysis", "architectural_analysis"]
    )

    # Find all functions that depend on the legacy module
    dependent_functions = []
    for func in legacy_analysis["matching_functions"]:
        dependencies = await trace_function_chain_tool(
            entry_point=func["breadcrumb"],
            project_name="legacy-system",
            direction="backward",
            max_depth=10
        )
        dependent_functions.append({
            "function": func["breadcrumb"],
            "dependencies": dependencies
        })

    # Find all functions the legacy module depends on
    dependency_analysis = []
    for func in legacy_analysis["matching_functions"]:
        deps = await trace_function_chain_tool(
            entry_point=func["breadcrumb"],
            project_name="legacy-system",
            direction="forward",
            max_depth=8
        )
        dependency_analysis.append({
            "function": func["breadcrumb"],
            "calls": deps
        })

    return {
        "module_analysis": legacy_analysis,
        "dependent_functions": dependent_functions,
        "dependency_analysis": dependency_analysis,
        "refactoring_plan": legacy_analysis.get("refactoring_suggestions", [])
    }

# Plan refactoring for old authentication module
refactor_plan = await plan_legacy_refactoring("*auth*legacy*")
```

### Scenario 3: Performance Optimization

**Problem**: Your application is slow, and you need to identify bottlenecks.

**Solution**: Analyze function chains to find the longest execution paths and complexity hotspots.

```python
async def identify_performance_bottlenecks():
    # Get overall project complexity analysis
    complexity_analysis = await analyze_project_chains_tool(
        project_name="slow-app",
        analysis_scope="full_project",
        complexity_threshold=0.8,  # High complexity threshold
        include_hotspot_analysis=True,
        analysis_types=["complexity_analysis", "hotspot_identification"]
    )

    # Analyze the most complex functions in detail
    bottlenecks = []
    for hotspot in complexity_analysis["hotspots"][:10]:
        # Find the longest paths through this function
        paths_through = await find_function_path_tool(
            start_function="main",  # Common entry point
            end_function=hotspot["breadcrumb"],
            project_name="slow-app",
            strategy="all",
            max_paths=3,
            optimize_for="simplicity"  # Find simpler alternatives
        )

        # Trace the function's internal execution
        internal_trace = await trace_function_chain_tool(
            entry_point=hotspot["breadcrumb"],
            project_name="slow-app",
            direction="forward",
            max_depth=5,
            performance_monitoring=True
        )

        bottlenecks.append({
            "function": hotspot["breadcrumb"],
            "complexity": hotspot["complexity_score"],
            "paths_to": paths_through,
            "internal_flow": internal_trace
        })

    return {
        "overall_analysis": complexity_analysis,
        "bottlenecks": bottlenecks
    }

# Identify performance issues
perf_issues = await identify_performance_bottlenecks()
```

## Best Practices

### 1. Start with Project-Wide Analysis

Always begin with `analyze_project_chains_tool` to get an overview:

```python
# Good: Start broad
overview = await analyze_project_chains_tool(
    project_name="my-app",
    analysis_scope="full_project",
    include_hotspot_analysis=True
)

# Then drill down based on findings
for hotspot in overview["hotspots"][:5]:
    detailed_trace = await trace_function_chain_tool(
        entry_point=hotspot["breadcrumb"],
        project_name="my-app"
    )
```

### 2. Use Appropriate Depth Limits

Balance detail with performance:

```python
# For quick exploration
shallow_trace = await trace_function_chain_tool(
    entry_point="main",
    project_name="my-app",
    max_depth=5  # Quick overview
)

# For detailed analysis
deep_trace = await trace_function_chain_tool(
    entry_point="critical_function",
    project_name="my-app",
    max_depth=15  # Detailed analysis
)
```

### 3. Leverage Output Formats Effectively

Choose the right format for your use case:

```python
# For documentation - use Mermaid
doc_trace = await trace_function_chain_tool(
    entry_point="api_handler",
    project_name="my-app",
    output_format="mermaid"
)

# For analysis/logging - use arrow format
analysis_trace = await trace_function_chain_tool(
    entry_point="api_handler",
    project_name="my-app",
    output_format="arrow"
)

# For comprehensive reports - use both
complete_trace = await trace_function_chain_tool(
    entry_point="api_handler",
    project_name="my-app",
    output_format="both"
)
```

### 4. Use Quality Metrics for Decision Making

Let quality scores guide your analysis:

```python
path_analysis = await find_function_path_tool(
    start_function="input",
    end_function="output",
    project_name="my-app",
    strategy="all",
    include_quality_metrics=True
)

# Sort by quality for best recommendations
best_paths = sorted(
    path_analysis["paths"],
    key=lambda p: p["quality"]["overall_score"],
    reverse=True
)

print(f"Recommended path: {best_paths[0]['arrow_format']}")
```

### 5. Combine Tools for Complete Analysis

Use tools together for comprehensive insights:

```python
async def comprehensive_function_analysis(function_name, project_name):
    # 1. Analyze the function's role in the project
    project_context = await analyze_project_chains_tool(
        project_name=project_name,
        analysis_scope="scoped_breadcrumbs",
        breadcrumb_patterns=[f"*{function_name}*"]
    )

    # 2. Trace its execution flow
    execution_flow = await trace_function_chain_tool(
        entry_point=function_name,
        project_name=project_name,
        direction="bidirectional",
        output_format="both"
    )

    # 3. Find alternative paths to/from this function
    main_to_func = await find_function_path_tool(
        start_function="main",
        end_function=function_name,
        project_name=project_name,
        strategy="optimal"
    )

    return {
        "context": project_context,
        "execution": execution_flow,
        "access_paths": main_to_func
    }
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "No paths found" when you expect connections

**Causes**:
- Functions are in different modules with no direct connections
- Max depth is too shallow
- Link strength threshold is too high

**Solutions**:
```python
# Try increasing max_depth
result = await find_function_path_tool(
    start_function="func_a",
    end_function="func_b",
    project_name="my-app",
    max_depth=20  # Increased from default 15
)

# Try lowering link strength threshold
result = await find_function_path_tool(
    start_function="func_a",
    end_function="func_b",
    project_name="my-app",
    min_link_strength=0.1  # Lowered from default 0.3
)

# Try "all" strategy to find any possible path
result = await find_function_path_tool(
    start_function="func_a",
    end_function="func_b",
    project_name="my-app",
    strategy="all"
)
```

#### Issue 2: Performance is slower than expected

**Causes**:
- Large codebase with deep function chains
- High complexity threshold causing extensive analysis
- Cache is not being utilized

**Solutions**:
```python
# Reduce scope for better performance
result = await analyze_project_chains_tool(
    project_name="large-app",
    analysis_scope="scoped_breadcrumbs",  # Instead of "full_project"
    breadcrumb_patterns=["*service*"],    # Focus on specific patterns
    max_functions_per_chain=20           # Limit chain size
)

# Use lower complexity threshold
result = await analyze_project_chains_tool(
    project_name="large-app",
    complexity_threshold=0.5,  # Lower threshold
    include_refactoring_suggestions=False  # Skip expensive analysis
)
```

#### Issue 3: Results are too broad or not specific enough

**Causes**:
- Breadcrumb patterns are too general
- Analysis scope is too wide
- Need more specific function targeting

**Solutions**:
```python
# Use more specific patterns
result = await analyze_project_chains_tool(
    project_name="my-app",
    analysis_scope="function_patterns",
    breadcrumb_patterns=["api.handlers.*", "core.auth.*"]  # More specific
)

# Use natural language for precise targeting
result = await trace_function_chain_tool(
    entry_point="user authentication handler",  # Natural language
    project_name="my-app"
)
```

### Performance Tips

1. **Use caching effectively**: Let the system cache graph structures
2. **Start small**: Begin with specific modules before analyzing entire projects
3. **Monitor depth**: Use appropriate max_depth values (5-10 for exploration, 15+ for detailed analysis)
4. **Batch operations**: When analyzing multiple functions, consider the order to leverage caching

### Error Handling

All Function Chain tools include comprehensive error handling:

```python
try:
    result = await trace_function_chain_tool(
        entry_point="nonexistent_function",
        project_name="my-app"
    )

    if "error" in result:
        print(f"Error: {result['error']}")
        if "suggestions" in result:
            print(f"Suggestions: {result['suggestions']}")

except Exception as e:
    print(f"Unexpected error: {e}")
```

## Conclusion

Function Chain MCP tools provide powerful capabilities for understanding code execution flows, finding optimal paths, and analyzing project-wide patterns. By following the examples and best practices in this guide, you can effectively use these tools to:

- Debug complex issues by tracing execution flows
- Plan refactoring efforts with comprehensive dependency analysis
- Optimize performance by identifying bottlenecks and alternative paths
- Understand system architecture through pattern analysis

For more information, see:
- [MCP Tools Reference](../MCP_TOOLS.md)
- [Graph RAG Architecture Guide](../GRAPH_RAG_ARCHITECTURE.md)
- [Best Practices](../BEST_PRACTICES.md)
