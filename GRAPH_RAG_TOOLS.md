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

## Integration with Existing Tools

### Compatibility with Search Tools

The Graph RAG tools are designed to complement the existing search functionality:

- **Semantic Search**: Use `search` tool for general content search
- **Graph Analysis**: Use `graph_analyze_structure_tool` for structural relationships
- **Pattern Search**: Use `graph_find_similar_implementations_tool` for cross-project patterns
- **Architecture Analysis**: Use `graph_identify_patterns_tool` for design patterns

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
