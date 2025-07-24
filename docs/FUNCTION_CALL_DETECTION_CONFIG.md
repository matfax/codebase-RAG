# Function Call Detection Configuration Guide

This guide explains how to configure the function call detection feature in the Graph RAG system.

## Overview

Function call detection is a feature that automatically identifies and maps function calls within your codebase, creating edges in the Graph RAG structure that represent calling relationships between functions and methods.

## Default Configuration

By default, function call detection is **enabled** with the following settings:

```python
{
    "enabled": True,
    "confidence_threshold": 0.5  # Minimum confidence for including function call edges
}
```

## Configuration Methods

### 1. Using GraphRAGService

The primary way to configure function call detection is through the `GraphRAGService`:

```python
from src.services.graph_rag_service import get_graph_rag_service

# Get the Graph RAG service instance
graph_service = get_graph_rag_service()

# Enable function call detection with default settings
graph_service.configure_function_call_detection(enable=True)

# Enable with custom confidence threshold
graph_service.configure_function_call_detection(
    enable=True,
    confidence_threshold=0.7  # Higher threshold = fewer but more confident edges
)

# Disable function call detection
graph_service.configure_function_call_detection(enable=False)

# Check current configuration
config = graph_service.get_function_call_detection_config()
print(f"Function call detection: {config}")
```

### 2. Direct StructureRelationshipBuilder Configuration

For more advanced use cases, you can configure the relationship builder directly:

```python
from src.services.structure_relationship_builder import StructureRelationshipBuilder

# Assuming you have a relationship builder instance
builder.configure_function_call_detection(
    enable=True,
    confidence_threshold=0.8
)
```

## Configuration Parameters

### `enable` (bool)
- **Default**: `True`
- **Description**: Whether to enable function call detection and edge creation
- **Impact**: When disabled, no function call relationships will be detected or added to the graph

### `confidence_threshold` (float)
- **Default**: `0.5`
- **Range**: `0.0` to `1.0`
- **Description**: Minimum confidence score required for a function call edge to be included in the graph
- **Impact**:
  - Higher values (e.g., 0.8): Fewer edges, higher precision
  - Lower values (e.g., 0.3): More edges, lower precision

### `invalidate_cache` (bool)
- **Default**: `True` (for GraphRAGService methods)
- **Description**: Whether to clear existing graph caches when configuration changes
- **Recommendation**: Keep as `True` to ensure configuration changes take effect immediately

## Performance Considerations

### When to Disable Function Call Detection

Consider disabling function call detection in the following scenarios:

1. **Large Codebases**: Function call detection adds processing overhead
2. **Memory Constraints**: Additional edges increase memory usage
3. **Focus on Structure Only**: When you only need hierarchical relationships
4. **Legacy Code Analysis**: When function calls are not the primary concern

### Optimizing Performance

```python
# For large codebases, use higher confidence threshold
graph_service.configure_function_call_detection(
    enable=True,
    confidence_threshold=0.8  # Reduces number of edges
)

# For memory-sensitive environments, disable if not needed
graph_service.configure_function_call_detection(enable=False)
```

## Usage in Graph RAG Tools

### Filtering Function Call Relationships

When using Graph RAG tools, you can filter relationships:

```python
# Include only function call relationships
from src.services.graph_traversal_algorithms import RelationshipFilter, TraversalOptions

options = TraversalOptions(
    relationship_filter=RelationshipFilter.FUNCTION_CALLS_ONLY
)

# Exclude function call relationships
options = TraversalOptions(
    relationship_filter=RelationshipFilter.NO_FUNCTION_CALLS
)
```

### Analyzing Function Call Patterns

Use the enhanced traversal algorithms to analyze function call patterns:

```python
from src.services.graph_traversal_algorithms import GraphTraversalAlgorithms

traversal = GraphTraversalAlgorithms()
graph = await graph_service.build_structure_graph("my_project")

# Analyze function call patterns
analysis = await traversal.analyze_function_call_patterns(graph)
print(f"Total function calls: {analysis['total_function_calls']}")
print(f"Most called functions: {analysis['most_called_functions']}")
```

## Environment Variables

For production deployments, you can set default configuration via environment variables:

```bash
# .env file
GRAPH_RAG_FUNCTION_CALL_DETECTION_ENABLED=true
GRAPH_RAG_FUNCTION_CALL_CONFIDENCE_THRESHOLD=0.6
```

## Backward Compatibility

Function call detection is designed to be fully backward compatible:

- **Existing Code**: All existing Graph RAG functionality continues to work unchanged
- **Default Behavior**: Function call detection is enabled by default but doesn't break existing workflows
- **Gradual Adoption**: You can enable/disable the feature per project or analysis
- **Legacy Support**: Existing relationship types and analysis methods are preserved

## Troubleshooting

### Common Issues

1. **No Function Call Edges Detected**
   - Check that `enable=True` in configuration
   - Verify confidence threshold isn't too high
   - Ensure code contains detectable function calls

2. **Too Many Function Call Edges**
   - Increase `confidence_threshold` (e.g., from 0.5 to 0.7)
   - Use relationship filters in traversal algorithms

3. **Performance Issues**
   - Consider disabling function call detection for large codebases
   - Increase confidence threshold to reduce edge count
   - Use selective relationship filtering

### Debug Information

```python
# Check current configuration
config = graph_service.get_function_call_detection_config()
print(f"Configuration: {config}")

# Get statistics after graph building
builder_stats = graph_service.relationship_builder.get_build_statistics()
print(f"Function call relationships: {builder_stats.function_call_relationships}")
```

## Migration Guide

### Upgrading from Previous Versions

If you're upgrading from a version without function call detection:

1. **No Code Changes Required**: Existing code continues to work
2. **Optional Enhancement**: Use new function call analysis features
3. **Performance Testing**: Monitor performance impact on large codebases
4. **Gradual Rollout**: Test with function call detection disabled initially

### Disabling for Specific Projects

```python
# Disable for a specific analysis
graph_service.configure_function_call_detection(enable=False)
graph = await graph_service.build_structure_graph("legacy_project")

# Re-enable for other projects
graph_service.configure_function_call_detection(enable=True)
```

## Best Practices

1. **Start with Defaults**: Begin with default settings and adjust based on your needs
2. **Monitor Performance**: Track graph building time and memory usage
3. **Use Appropriate Thresholds**: Balance between precision and recall
4. **Leverage Filtering**: Use relationship filters to focus on relevant connections
5. **Regular Review**: Periodically review configuration as codebase evolves

## Support

For issues or questions about function call detection configuration:

1. Check this documentation
2. Review the backward compatibility test results
3. Examine the Graph RAG architecture documentation
4. Use debug logging to understand detection behavior
