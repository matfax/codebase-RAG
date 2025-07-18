# Tasks: Enhanced Function Call Detection for Graph RAG Tools

Based on PRD: `prd-enhanced-function-call-detection.md`

## Relevant Files

- `src/services/chunking_strategies.py` - Extend PythonChunkingStrategy and other language strategies with call detection
- `src/services/ast_extraction_service.py` - Add function call extraction methods to AST processing
- `src/utils/tree_sitter_manager.py` - Ensure Tree-sitter queries support call detection patterns
- `src/services/function_call_extractor_service.py` - NEW: Service for extracting function calls from AST
- `src/services/function_call_extractor_service.test.py` - Unit tests for function call extraction
- `src/services/call_weight_calculator_service.py` - NEW: Service for calculating call weights and confidence
- `src/services/call_weight_calculator_service.test.py` - Unit tests for weight calculation
- `src/services/breadcrumb_resolver_service.py` - Extend to resolve function call targets to breadcrumbs
- `src/services/structure_relationship_builder.py` - Extend to include function call edges in graphs
- `src/models/function_call.py` - NEW: Data model for function call relationships
- `src/utils/call_pattern_registry.py` - NEW: Registry for language-specific call patterns
- `src/services/graph_performance_optimizer.py` - Extend with call detection performance optimizations
- `src/tests/test_function_call_detection.py` - NEW: Integration tests for call detection feature
- `src/tests/test_call_detection_performance.py` - NEW: Performance tests for call detection

### Notes

- Focus on Python implementation first, then extend to JavaScript/TypeScript
- Use existing Tree-sitter infrastructure and extend chunking strategies
- Integrate with existing Graph RAG services rather than creating parallel systems
- Use pytest for testing: `pytest src/tests/test_function_call_detection.py`

## Tasks

- [x] 1.0 Enhance Tree-sitter AST Queries for Function Call Detection
  - [x] 1.1 Research Tree-sitter query syntax for Python function calls (direct calls, method calls, attribute calls)
  - [x] 1.2 Create new Tree-sitter query patterns for detecting `function_name()`, `object.method()`, `self.method()`
  - [x] 1.3 Add async call detection patterns for `await function()`, `asyncio.gather()`, `asyncio.create_task()`
  - [x] 1.4 Extend `PythonChunkingStrategy.get_node_mappings()` to include call detection node types
  - [x] 1.5 Test query patterns against real Python codebases to verify detection accuracy

- [x] 2.0 Implement Function Call Weight and Confidence System
  - [x] 2.1 Create `FunctionCall` data model with source/target breadcrumbs, call type, confidence, weight
  - [x] 2.2 Implement `CallWeightCalculator` service with configurable weights (direct: 1.0, method: 0.9, attribute: 0.7)
  - [x] 2.3 Add frequency factor calculation for repeated calls in same file
  - [x] 2.4 Implement confidence scoring based on call context and AST node completeness
  - [x] 2.5 Create configurable weight thresholds and filtering system

- [ ] 3.0 Build Function Call Resolver and Breadcrumb Integration
  - [ ] 3.1 Create `FunctionCallExtractor` service to extract calls from AST nodes
  - [ ] 3.2 Extend `BreadcrumbResolver` to resolve function call targets to breadcrumbs
  - [ ] 3.3 Handle cross-file function resolution using existing project indexing
  - [ ] 3.4 Implement attribute call chain resolution (e.g., `self.progress_tracker.set_total_items`)
  - [ ] 3.5 Add error handling for unresolvable calls with confidence degradation

- [ ] 4.0 Integrate Function Call Detection with Graph Builder
  - [ ] 4.1 Extend `StructureRelationshipBuilder` to include function call edges alongside import edges
  - [ ] 4.2 Modify `GraphEdge` model to support call relationship types and metadata
  - [ ] 4.3 Update graph traversal algorithms to handle function call relationships
  - [ ] 4.4 Ensure backward compatibility with existing Graph RAG tools
  - [ ] 4.5 Add configuration toggle to enable/disable call detection feature

- [ ] 5.0 Add Performance Optimization and Caching Layer
  - [ ] 5.1 Implement breadcrumb resolution caching with TTL based on file modification times
  - [ ] 5.2 Add concurrent processing for function call extraction across multiple files
  - [ ] 5.3 Optimize Tree-sitter query patterns for performance on large codebases
  - [ ] 5.4 Implement incremental call detection for modified files only
  - [ ] 5.5 Add performance monitoring and metrics collection for call detection pipeline
