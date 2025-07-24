# Graph RAG MCP Tools

This document describes the Graph RAG MCP tools implemented in Wave 4 of the graph-rag-enhancement project.

## Overview

The Graph RAG tools provide advanced code structure analysis, cross-project search, and pattern recognition capabilities built on top of the Graph RAG infrastructure developed in previous waves.

## Available Tools

### 1. graph_analyze_structure_tool

Analyzes the structural relationships of a specific breadcrumb in the codebase.

**Usage:**
```python
result = await graph_analyze_structure_tool(
    breadcrumb="MyClass.method_name",
    project_name="my_project",
    analysis_type="comprehensive",  # "comprehensive", "hierarchy", "connectivity", "overview"
    max_depth=3,
    include_siblings=False,
    include_connectivity=True,
    force_rebuild_graph=False
)
```

**Returns:**
- Hierarchical relationships
- Connectivity patterns
- Related components
- Navigation paths
- Graph statistics

### 2. graph_find_similar_implementations_tool

Finds similar implementations across projects using Graph RAG capabilities.

**Usage:**
```python
result = await graph_find_similar_implementations_tool(
    query="authentication middleware implementation",
    source_breadcrumb="AuthMiddleware.authenticate",  # Optional
    source_project="project_a",  # Optional
    target_projects=["project_b", "project_c"],  # Optional
    similarity_threshold=0.7,
    max_results=10,
    include_implementation_chains=True,
    include_architectural_context=True
)
```

**Returns:**
- Similar implementations with scores
- Architectural context
- Implementation chains (optional)
- Cross-project statistics

### 3. graph_identify_patterns_tool

Identifies architectural patterns in a codebase using Graph RAG pattern recognition.

**Usage:**
```python
result = await graph_identify_patterns_tool(
    project_name="my_project",
    pattern_types=["structural", "behavioral", "architectural"],  # Optional
    scope_breadcrumb="MyModule",  # Optional
    min_confidence=0.6,
    include_comparisons=True,
    include_improvements=False,
    max_patterns=20,
    analysis_depth="comprehensive"  # "basic", "comprehensive", "detailed"
)
```

**Returns:**
- Identified patterns with confidence scores
- Pattern comparisons
- Quality insights
- Improvement suggestions (optional)

## Function Chain Analysis Tools

### 4. trace_function_chain_tool

Traces complete function execution chains from an entry point with comprehensive analysis.

**Usage:**
```python
result = await trace_function_chain_tool(
    entry_point="MyClass.method_name",  # Function/class identifier
    project_name="my_project",
    direction="forward",  # "forward", "backward", "bidirectional"
    max_depth=10,
    output_format="arrow",  # "arrow", "mermaid", "both"
    include_mermaid=False,
    chain_type="execution_flow",  # "execution_flow", "data_flow", "dependency_chain"
    min_link_strength=0.3,
    identify_branch_points=True,
    identify_terminal_points=True,
    performance_monitoring=True
)
```

**Returns:**
- Function execution chains with formatted output
- Branch points and terminal points identification
- Performance metrics and timing
- Chain complexity analysis
- Optional Mermaid diagrams

### 5. find_function_path_tool

Finds the most efficient path between two functions in a codebase with quality metrics.

**Usage:**
```python
result = await find_function_path_tool(
    start_function="authenticate_user",  # Starting function identifier
    end_function="validate_credentials",  # Target function identifier
    project_name="my_project",
    strategy="optimal",  # "shortest", "optimal", "all"
    max_paths=3,
    max_depth=15,
    include_quality_metrics=True,
    output_format="arrow",  # "arrow", "mermaid", "both"
    include_mermaid=False,
    min_link_strength=0.3,
    optimize_for="reliability"  # "reliability", "directness", "simplicity"
)
```

**Returns:**
- Optimal paths between functions with quality metrics
- Path reliability and complexity scores
- Multiple path options with rankings
- Visual path representations
- Performance diagnostics

### 6. analyze_project_chains_tool

Analyzes function chains across an entire project with comprehensive insights.

**Usage:**
```python
result = await analyze_project_chains_tool(
    project_name="my_project",
    analysis_scope="full_project",  # "full_project", "scoped_breadcrumbs", "specific_modules"
    breadcrumb_patterns=["*.service.*", "*.controller.*"],  # Optional
    analysis_types=["complexity_analysis", "hotspot_identification", "pattern_detection"],
    max_functions_per_chain=50,
    complexity_threshold=0.7,
    output_format="comprehensive",  # "comprehensive", "summary", "detailed"
    include_mermaid=True,
    include_hotspot_analysis=True,
    include_refactoring_suggestions=False,
    enable_complexity_weighting=True
)
```

**Returns:**
- Project-wide function chain analysis
- Complexity hotspot identification
- Architectural pattern detection
- Refactoring recommendations (optional)
- Comprehensive reporting with visualizations

## Integration with Existing Tools

### Compatibility with Search Tools

The Graph RAG tools are designed to complement the existing search functionality:

- **Semantic Search**: Use `search` tool for general content search
- **Graph Analysis**: Use `graph_analyze_structure_tool` for structural relationships
- **Pattern Search**: Use `graph_find_similar_implementations_tool` for cross-project patterns
- **Architecture Analysis**: Use `graph_identify_patterns_tool` for design patterns
- **Function Chain Analysis**: Use `trace_function_chain_tool` for execution flow tracing
- **Function Path Finding**: Use `find_function_path_tool` for optimal path discovery
- **Project-wide Analysis**: Use `analyze_project_chains_tool` for comprehensive insights

### Service Dependencies

Graph RAG tools rely on services from previous waves:

**Wave 1 Dependencies:**
- `StructureAnalyzerService` - Enhanced CodeChunk analysis
- `BreadcrumbExtractor` - Hierarchical relationship extraction

**Wave 2 Dependencies:**
- `GraphRAGService` - Core Graph RAG controller
- `StructureRelationshipBuilder` - Graph construction
- `GraphTraversalAlgorithms` - Navigation algorithms

**Wave 3 Dependencies:**
- `CrossProjectSearchService` - Cross-project capabilities
- `PatternRecognitionService` - Pattern identification
- `ImplementationChainService` - Implementation tracing
- `PatternComparisonService` - Pattern analysis

**Enhanced Function Call Detection Dependencies:**
- `FunctionCallExtractor` - Function call detection from AST
- `CallConfidenceScorer` - Function call confidence analysis
- `CallWeightCalculator` - Function call weight calculation
- `BreadcrumbResolver` - Natural language to breadcrumb conversion
- `IntegratedFunctionCallResolver` - Integrated call resolution pipeline

### Error Handling

All Graph RAG tools follow consistent error handling patterns:

```python
{
    "success": false,
    "error": "Detailed error message",
    "query/breadcrumb/project_name": "original_input"
}
```

### Performance Considerations

- Tools use caching mechanisms from previous waves
- Graph construction is cached and reused across requests
- `force_rebuild_graph=True` can be used to refresh stale data
- Similarity thresholds can be adjusted for performance vs. accuracy

## Tool Registration

The tools are registered in `src/tools/registry.py` and follow the standard MCP tool registration pattern:

```python
@mcp_app.tool()
async def graph_analyze_structure_tool(...):
    return await graph_analyze_structure(...)
```

## Testing and Validation

### Syntax Validation
All tools pass Python syntax validation:
```bash
python -m py_compile tools/graph_rag/*.py
```

### Import Validation
Tools use consistent import patterns matching existing codebase:
```python
from services.embedding_service import EmbeddingService
from services.graph_rag_service import GraphRAGService
```

### Consistency Checks
- Parameter validation follows existing patterns
- Return formats include comprehensive metadata
- Error handling provides detailed diagnostic information
- Async patterns match existing tool implementations

## Usage Examples

### Finding Similar Authentication Implementations
```python
auth_implementations = await graph_find_similar_implementations_tool(
    query="JWT token validation middleware",
    target_projects=["backend_api", "auth_service"],
    chunk_types=["function", "class"],
    languages=["python", "typescript"],
    similarity_threshold=0.8
)
```

### Analyzing Service Architecture Patterns
```python
patterns = await graph_identify_patterns_tool(
    project_name="microservices_app",
    pattern_types=["architectural", "structural"],
    min_confidence=0.7,
    include_comparisons=True,
    analysis_depth="detailed"
)
```

### Exploring Code Relationships
```python
structure = await graph_analyze_structure_tool(
    breadcrumb="UserService.createUser",
    project_name="user_management",
    analysis_type="comprehensive",
    include_connectivity=True,
    max_depth=4
)
```
