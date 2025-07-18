# PRD: Enhanced Function Call Detection for Graph RAG Tools

## Introduction/Overview

The current Graph RAG system primarily relies on import dependencies to build code structure graphs, which limits its ability to trace actual function execution flows. This feature enhances the system with comprehensive function call detection capabilities, enabling users to trace complete execution paths and understand data flow within codebases.

**Problem Statement**: Developers and AI systems need to understand how functions interact at runtime, not just their static dependencies. Current tools show "what imports what" but fail to show "what calls what", making it difficult to trace execution flows like `process_specific_files` → `_reset_counters` → `progress_tracker.set_total_items`.

**Goal**: Transform Graph RAG from a "dependency analysis tool" into an "execution flow analysis tool" by adding multi-layered function call detection.

## Goals

1. **Increase Detection Coverage**: Detect 70%+ of function call relationships in Python codebases
2. **Support Multiple Call Types**: Direct calls, method calls, async calls, attribute calls
3. **Provide Weighted Relationships**: Assign confidence scores based on call frequency and context
4. **Maintain Performance**: Process large codebases within acceptable time limits (prefer accuracy 70%, performance 30%)
5. **Enable Flow Tracing**: Allow users to trace complete execution paths from entry points

## User Stories

### Primary User Stories
1. **As a developer**, I want to trace the complete call chain of `process_specific_files` function so that I can understand the data flow and identify optimization opportunities.

2. **As a system analyst**, I want to identify async call patterns in the codebase so that I can optimize performance bottlenecks and understand concurrency flows.

3. **As an AI system**, I want to automatically detect function call relationships so that I can provide accurate code analysis and recommendations.

4. **As a code reviewer**, I want to visualize function call graphs so that I can understand the impact of proposed changes.

### Secondary User Stories
5. **As a maintenance developer**, I want to identify frequently called functions so that I can prioritize optimization efforts.

6. **As an architect**, I want to analyze call patterns across modules so that I can identify architectural improvements.

## Functional Requirements

### Core Requirements (Must-Have)

1. **AST Query Enhancement**
   - The system must enhance Tree-sitter queries to detect direct function calls: `function_name()`
   - The system must detect method calls: `object.method()`
   - The system must detect self method calls: `self.method()`
   - The system must capture call locations (line numbers) for debugging

2. **Attribute Call Analysis**
   - The system must resolve attribute method calls: `self.progress_tracker.set_total_items()`
   - The system must distinguish between property access and method calls
   - The system must resolve chained method calls: `obj.service.method()`

3. **Call Weight System**
   - The system must assign weights to different call types (direct: 1.0, method: 0.9, attribute: 0.7)
   - The system must calculate frequency factors for repeated calls
   - The system must provide confidence scores for each detected relationship
   - The system must filter calls below configurable weight thresholds

### Important Requirements (Should-Have)

4. **Async Call Detection**
   - The system should detect `await function()` patterns
   - The system should identify `asyncio.gather()` usage
   - The system should recognize `asyncio.create_task()` patterns
   - The system should handle `async with` and `async for` constructs

5. **Dynamic Call Detection**
   - The system should detect `getattr()` based calls
   - The system should identify callback patterns
   - The system should recognize decorator-based calls

### Enhanced Requirements (Could-Have)

6. **Multi-Language Support**
   - The system could extend support to JavaScript/TypeScript
   - The system could support Java method calls
   - The system could handle Go function calls

7. **Advanced Analysis**
   - The system could detect recursive call patterns
   - The system could identify circular dependencies
   - The system could suggest refactoring opportunities

## Non-Goals (Out of Scope)

1. **Backward Compatibility**: No need to maintain compatibility with existing Graph RAG tool interfaces
2. **UI/UX Optimization**: User interface improvements are out of scope for this version
3. **Real-time Analysis**: Live code analysis during editing is not required
4. **IDE Integration**: Direct integration with IDEs is not included
5. **Performance Profiling**: Runtime performance measurement is not included

## Design Considerations

### Technical Architecture
- **Parser Enhancement**: Extend existing Tree-sitter integration with new query patterns
- **Weight Calculator**: New component for calculating call relationship weights
- **Call Resolver**: Service to resolve function names to breadcrumbs
- **Graph Builder**: Enhanced to incorporate function call edges alongside import edges

### Data Models
```python
@dataclass
class FunctionCall:
    source_breadcrumb: str
    target_breadcrumb: str
    call_type: CallType  # direct, method, attribute, async
    line_number: int
    confidence: float
    weight: float
```

### Configuration
- Configurable weight thresholds per call type
- Enable/disable specific detection types
- Language-specific parsing rules

## Technical Considerations

### Tree-sitter Capabilities
Based on research, Tree-sitter provides:
- Robust query system for pattern matching function calls
- Millisecond-level performance for parsing
- Support for incremental parsing
- Native Python and JavaScript support
- Call graph construction capabilities

### Implementation Approach
1. **Phase 1**: Extend existing `CodeParserService` with call detection queries
2. **Phase 2**: Implement `FunctionCallExtractor` and `CallWeightCalculator`
3. **Phase 3**: Integrate with `GraphBuilder` to create enhanced structure graphs
4. **Phase 4**: Add async and dynamic call detection

### Performance Considerations
- Parsing time expected to increase by 20-30%
- Memory usage may increase due to additional edge storage
- Caching strategies for resolved breadcrumbs
- Configurable depth limits for call tracing

### Dependencies
- Current Tree-sitter integration (sufficient for requirements)
- Existing breadcrumb resolution system
- Graph storage infrastructure

## Success Metrics

### Primary Metrics
1. **Detection Accuracy**: Successfully detect 70%+ of function calls in test codebases
2. **Coverage Improvement**: Increase graph edge count by 200-400% compared to import-only graphs
3. **Performance**: Parse time increase limited to 30% of baseline

### Secondary Metrics
4. **User Satisfaction**: Developers can successfully trace function flows in their use cases
5. **False Positive Rate**: Keep incorrectly detected calls below 10%
6. **Memory Usage**: Memory increase limited to 50% of baseline

### Test Cases
- **process_specific_files trace**: Successfully detect → `_reset_counters`, `progress_tracker.set_total_items`, `_process_single_file`, `asyncio.gather`
- **Async patterns**: Detect async/await relationships in real codebases
- **Large codebase**: Process 10k+ function codebase within reasonable time

## Open Questions & Decisions

1. **Multi-language Priority**: Which languages should be implemented after Python?
   **DECISION**: JavaScript/TypeScript (priority), Java (secondary), Go (third) - based on ecosystem adoption and implementation similarity to Python patterns.

2. **Call Depth Limits**: What maximum call depth should be traced to prevent infinite recursion?
   **DECISION**: Default 15 levels with configurable range (5-50). Implement cycle detection for recursion prevention. Provide shallow/medium/deep preset modes (5/15/30 levels).

3. **Integration Testing**: How should this integrate with existing MCP tools?
   **DECISION**: Backward compatible enhancement - new call edges augment existing import edges. Configurable feature toggle for gradual adoption. All existing MCP tools must remain functional.

4. **Caching Strategy**: How should resolved call relationships be cached across sessions?
   **DECISION**: Multi-layer caching: L1 (AST+calls per file), L2 (breadcrumb resolution), L3 (call graph). TTL based on file mtime for intelligent cache invalidation.

5. **Error Handling**: How should ambiguous or unresolvable calls be handled?
   **DECISION**: Graded handling - unresolvable (confidence=0.1), ambiguous (confidence=0.5-0.7 with context heuristics), syntax errors (graceful degradation), missing imports (cross-file resolution with confidence reduction).

6. **Performance Tuning**: What specific performance optimizations will be needed for large codebases?
   **DECISION**: Priority order - Query optimization > Incremental updates = Parallel processing > Memory management. Focus on Tree-sitter query pattern matching optimization first.

---

**Document Version**: 1.0
**Created**: 2025-07-18
**Target Implementation**: Q1 2025
**Primary Language**: Python (with multi-language expansion planned)
