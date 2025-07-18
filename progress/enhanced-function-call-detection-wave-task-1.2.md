# Task 1.2: Tree-sitter Query Patterns for Function Call Detection

## Overview
Created comprehensive Tree-sitter query patterns for detecting different types of function calls in Python code, including direct calls, method calls, and self method calls.

## Deliverables

### 1. PythonCallPatterns Class (`src/utils/python_call_patterns.py`)

Created a comprehensive class containing Tree-sitter query patterns for:

#### Basic Function Call Types
- **Direct Function Calls**: `function_name(args)`
  - Pattern captures function name and arguments
  - Examples: `print()`, `len(data)`, `calculate_sum(a, b)`

- **Method Calls**: `object.method(args)`
  - Pattern captures object, method name, and arguments
  - Examples: `obj.method()`, `data.append(item)`

- **Self Method Calls**: `self.method(args)`
  - Specific pattern for self method calls within classes
  - Uses predicate `(#eq? @call.self "self")` for precise matching
  - Examples: `self.process()`, `self.validate()`

- **Module Function Calls**: `module.function(args)`
  - Pattern for module-level function calls
  - Examples: `os.path.join()`, `json.dumps(data)`

#### Advanced Function Call Types
- **Chained Attribute Calls**: `obj.attr.method(args)`
  - Handles multi-level attribute access with method calls
  - Examples: `config.database.connection.execute()`

- **Subscript Method Calls**: `obj[key].method(args)`
  - Handles method calls on subscripted objects
  - Examples: `items[0].process()`, `data['key'].execute()`

- **Super Method Calls**: `super().method(args)`
  - Special handling for super() calls in inheritance
  - Examples: `super().__init__()`, `super().setup()`

- **Class Method Calls**: `cls.method(args)` or `Class.method(args)`
  - Handles static and class method calls
  - Examples: `MyClass.class_method()`, `User.from_dict(data)`

- **Dynamic Attribute Calls**: `getattr(obj, 'method')(args)`
  - Handles dynamic attribute access patterns
  - Examples: `getattr(obj, 'method')()`

- **Unpacking Calls**: `callable(*args, **kwargs)`
  - Handles calls with argument unpacking
  - Examples: `function(*args)`, `method(**kwargs)`

### 2. PythonCallNodeTypes Class

Defined Tree-sitter node types for function call detection:

#### Core Node Types
- `"call"` - Primary function/method call nodes
- `"attribute"` - Attribute access nodes

#### Extended Node Types
- `"subscript"` - Subscript access patterns
- `"argument_list"` - Function arguments
- `"identifier"` - Function/method names

### 3. Test Validation (`src/utils/test_python_call_patterns.py`)

Created comprehensive test cases including:

#### Test Categories
- **Direct Function Calls**: 7 expected patterns
- **Method Calls**: 7 expected patterns
- **Self Method Calls**: 3 expected patterns
- **Chained Attribute Calls**: 3 expected patterns
- **Module Function Calls**: 4 expected patterns
- **Super Method Calls**: 3 expected patterns
- **Class Method Calls**: 3 expected patterns
- **Dynamic Attribute Calls**: 2 expected patterns
- **Nested Calls**: 3 expected patterns

#### Edge Cases Covered
- Complex nested calls
- Calls in comprehensions
- Calls in conditional expressions
- Calls as arguments
- Calls with generators
- Calls in exception handling
- Calls in with statements
- Multiple calls on same line

## Pattern Design Principles

### 1. Capture Group Strategy
Each pattern uses specific capture groups:
- `@call.function.name` - Function being called
- `@call.object` - Object for method calls
- `@call.method.name` - Method name for method calls
- `@call.arguments` - Arguments passed to call
- Pattern-specific captures for context

### 2. Predicate Usage
- `(#eq? @call.self "self")` - Ensures self method detection
- `(#eq? @call.super "super")` - Ensures super call detection
- `(#eq? @call.getattr "getattr")` - Ensures dynamic call detection

### 3. Hierarchical Matching
Patterns are designed to capture the full calling context:
- Base object identification
- Intermediate attribute access
- Final method/function name
- Complete argument list

## Integration Points

### 1. Chunking Strategy Integration
Patterns designed to integrate with `PythonChunkingStrategy.get_node_mappings()`:
```python
# New mappings to add:
ChunkType.FUNCTION_CALL: ["call"]
ChunkType.ATTRIBUTE_ACCESS: ["attribute"]
```

### 2. Query Execution Strategy
- **Basic Patterns**: For initial implementation and performance
- **Advanced Patterns**: For comprehensive call detection
- **Combined Queries**: For single-pass detection of multiple pattern types

### 3. Metadata Extraction
Each pattern captures sufficient information for:
- Source location (line numbers)
- Call type classification
- Target function/method identification
- Relationship building (caller → callee)

## Performance Considerations

### 1. Pattern Complexity
- Basic patterns optimized for common cases
- Advanced patterns for comprehensive coverage
- Ability to choose pattern subset based on requirements

### 2. Query Combination
- `get_combined_query()` method combines multiple patterns efficiently
- Avoids multiple Tree-sitter traversals
- Maintains pattern identification through comments

### 3. Node Type Filtering
- Pre-filtering by node types before pattern matching
- Reduces unnecessary pattern evaluation
- Focuses on relevant AST nodes

## Next Steps for Task 1.3
1. Extend patterns with async call detection
2. Add patterns for `asyncio.gather()`, `asyncio.create_task()`
3. Handle `await` expressions with function calls
4. Integrate async patterns with existing synchronous patterns

## Validation Strategy
1. Test patterns against real codebases
2. Verify capture group correctness
3. Measure detection accuracy vs. false positives
4. Performance testing on large files

## Architecture Notes
- Patterns are self-contained and reusable
- Easy to extend with new call types
- Compatible with existing Tree-sitter infrastructure
- Designed for integration with existing chunking system
