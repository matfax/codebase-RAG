# Graph Trace Performance Optimization Plan

## Background
The current graph trace tools in our Codebase RAG project are experiencing severe performance issues, making it impractical to trace function call routes in real projects. This document outlines optimization strategies inspired by LightRAG's architecture.

## LightRAG Key Insights

### Performance Achievements
- Reduces token usage by 99%
- Requires only a single API call
- Shows up to 86.4% better performance in complex domains

### Core Optimization Strategies
1. **Deduplication**: Merges identical entities and relations to reduce graph size
2. **Dual-Level Retrieval**: Separates low-level (specific) and high-level (broad) queries
3. **Incremental Updates**: Updates only changed portions without full reprocessing
4. **Optimized Key-Value Indexing**: Fast and precise retrieval with text summaries
5. **Graph Segmentation**: Divides large graphs into manageable sub-graphs

## Proposed Implementation for Our Project

### 1. Function Call Deduplication
**Goal**: Reduce graph size by eliminating redundant edges

**Implementation**:
```python
# Example: Multiple calls to the same function from one source
# Before: A -> B (3 times)
# After: A -> B (weight: 3)

class FunctionCallDeduplicator:
    def deduplicate_edges(self, edges):
        # Group by (source, target)
        # Aggregate call counts as edge weights
        # Merge metadata (line numbers, contexts)
```

**Benefits**:
- Significantly smaller graphs
- Faster traversal
- Reduced memory usage

### 2. Breadcrumb-Based Fast Indexing
**Goal**: Enable O(1) lookup for function relationships

**Implementation**:
```python
# Hierarchical index structure
breadcrumb_index = {
    "module.class.method": {
        "direct_calls": ["func1", "func2"],
        "called_by": ["func3", "func4"],
        "transitive_calls": ["func5", "func6"],  # 2-hop
        "summary": "Core authentication method",
        "complexity": 5,
        "last_modified": "2024-01-01"
    }
}

# Quick lookups
def get_function_calls(breadcrumb: str) -> List[str]:
    return breadcrumb_index.get(breadcrumb, {}).get("direct_calls", [])
```

**Benefits**:
- Instant function relationship queries
- No graph traversal needed for common queries
- Cached summaries for context

### 3. Hierarchical Graph Architecture
**Goal**: Provide different levels of detail based on query needs

**Levels**:
```python
class GraphHierarchy:
    # Level 1: Module Dependencies (fastest)
    module_graph = {
        "src.services": ["src.models", "src.utils"],
        "src.tools": ["src.services"]
    }

    # Level 2: Class/Function Relationships (balanced)
    class_function_graph = {
        "ServiceA": ["ServiceB", "ModelC"],
        "function_x": ["function_y", "function_z"]
    }

    # Level 3: Detailed Method Calls (comprehensive)
    method_call_graph = {
        "ClassA.method1": {
            "calls": [("ClassB.method2", 15), ("func3", 42)],
            "context": "authentication flow"
        }
    }
```

**Query Router**:
```python
def route_query(query_type, depth_required):
    if query_type == "module_dependencies":
        return query_level_1()
    elif depth_required <= 2:
        return query_level_2()
    else:
        return query_level_3()
```

### 4. Incremental Graph Updates
**Goal**: Update only changed portions of the codebase

**Implementation**:
```python
class IncrementalGraphUpdater:
    def update_graph(self, changed_files):
        # 1. Identify affected breadcrumbs
        affected = self.get_affected_breadcrumbs(changed_files)

        # 2. Remove old relationships
        self.remove_stale_edges(affected)

        # 3. Parse only changed files
        new_chunks = self.parse_files(changed_files)

        # 4. Update only affected portions
        self.update_partial_graph(new_chunks)

        # 5. Invalidate only affected caches
        self.invalidate_caches(affected)
```

**Benefits**:
- 10-100x faster updates
- Preserves valid cached data
- Minimal disruption to service

### 5. Vector-Graph Hybrid Search
**Goal**: Combine semantic search with graph traversal

**Implementation**:
```python
class HybridSearcher:
    def search(self, query, mode="hybrid"):
        if mode == "semantic":
            # Use existing vector search
            return self.vector_search(query)

        elif mode == "graph":
            # Pure graph traversal
            return self.graph_traverse(query)

        elif mode == "hybrid":
            # 1. Vector search for starting points
            candidates = self.vector_search(query, top_k=10)

            # 2. Expand via graph relationships
            expanded = self.expand_via_graph(candidates)

            # 3. Re-rank by combined score
            return self.rerank(expanded)
```

### 6. Smart Caching Strategy
**Goal**: Aggressive caching with intelligent invalidation

**Cache Layers**:
```python
class SmartCache:
    # L1: Breadcrumb lookups (Redis)
    breadcrumb_cache = {}  # TTL: 1 hour

    # L2: Common traversal patterns
    pattern_cache = {
        "auth_flow": ["login->validate->create_session"],
        "data_pipeline": ["fetch->transform->store"]
    }  # TTL: 6 hours

    # L3: Full subgraph results
    subgraph_cache = {}  # TTL: 30 minutes

    # L4: Pre-computed embeddings
    embedding_cache = {}  # TTL: 24 hours
```

## Implementation Timeline

### Phase 1: Quick Wins (1-2 weeks)
- [ ] Implement function call deduplication
- [ ] Build breadcrumb fast index
- [ ] Add aggressive L1/L2 caching
- [ ] Create simple query router

### Phase 2: Core Improvements (3-4 weeks)
- [ ] Implement hierarchical graph architecture
- [ ] Build incremental update system
- [ ] Integrate vector-graph hybrid search
- [ ] Optimize cache invalidation logic

### Phase 3: Advanced Features (1-2 months)
- [ ] Integrate Node2Vec embeddings
- [ ] Implement smart graph segmentation
- [ ] Build full dual-level retrieval system
- [ ] Add pattern recognition and caching

## Performance Targets

### Current State
- Graph build time: >60 seconds for medium projects
- Query time: 5-30 seconds for complex traces
- Memory usage: Several GB for large codebases

### Target State
- Graph build time: <5 seconds (incremental)
- Query time: <500ms for most queries
- Memory usage: <500MB with smart caching

## Success Metrics

1. **Query Performance**
   - P50 latency < 100ms
   - P95 latency < 500ms
   - P99 latency < 2s

2. **Resource Usage**
   - Memory reduction > 80%
   - CPU usage reduction > 70%
   - Storage optimization > 60%

3. **Functionality**
   - Support for real-time queries
   - Ability to handle 100k+ function projects
   - Cross-project search capabilities

## Risk Mitigation

1. **Backward Compatibility**
   - Maintain existing API interfaces
   - Gradual migration path
   - Feature flags for new functionality

2. **Data Consistency**
   - Validation of deduplicated graphs
   - Consistency checks for incremental updates
   - Rollback mechanisms

3. **Performance Regression**
   - Comprehensive benchmarking suite
   - A/B testing for optimizations
   - Monitoring and alerting

## Next Steps

1. Review and prioritize optimization strategies
2. Set up performance benchmarking infrastructure
3. Create proof-of-concept for highest priority items
4. Establish success criteria for each phase
5. Begin implementation of Phase 1

## References

- [LightRAG GitHub Repository](https://github.com/HKUDS/LightRAG)
- [LightRAG Paper](https://arxiv.org/abs/2410.05779)
- [Graph RAG Best Practices](https://lightrag.github.io/)
