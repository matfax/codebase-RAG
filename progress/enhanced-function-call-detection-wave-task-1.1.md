# Task 1.1: Tree-sitter Query Syntax Research for Python Function Calls

## Overview
Researched Tree-sitter query syntax and Python grammar node types for detecting different types of function calls in Python code.

## Key Findings

### Python Function Call Node Types in Tree-sitter

1. **Basic Function Calls (`call`)**
   - Node type: `"call"`
   - Child fields:
     - `function`: The function being called (identifier or more complex expression)
     - `arguments`: Argument list passed to the function
   - Examples: `print()`, `len(data)`, `calculate_sum(a, b)`

2. **Method/Attribute Calls (`attribute`)**
   - Node type: `"attribute"`
   - Child fields:
     - `object`: The base object
     - `attribute`: The method/attribute name
   - Combined with `call` for method invocation
   - Examples: `obj.method()`, `self.process()`, `data.append(item)`

3. **Async Function Calls (`await`)**
   - Node type: `"await"`
   - Contains primary expression that can be a call
   - Examples: `await fetch_data()`, `await obj.async_method()`

### Tree-sitter Query Patterns Discovered

#### Basic Query Syntax
- Patterns use S-expressions: `(node_type field_name: child_pattern)`
- Field names provide specificity: `function:`, `object:`, `attribute:`
- Captures use `@name` syntax for extracting matched nodes

#### Function Call Patterns
```scheme
; Direct function calls
(call function: (identifier) @function.name
      arguments: (argument_list) @function.args)

; Method calls
(call function: (attribute object: (_) @object.name
                         attribute: (identifier) @method.name)
      arguments: (argument_list) @method.args)

; Attribute access (may or may not be call)
(attribute object: (_) @object
           attribute: (identifier) @attribute)

; Async calls
(await (call function: (identifier) @async.function
             arguments: (argument_list) @async.args))
```

### Current Infrastructure Analysis

#### Existing Tree-sitter Integration
- **File**: `/src/utils/tree_sitter_manager.py`
- Supports 8+ languages including Python
- Manages parsers and language objects centrally
- Python parser available via `tree_sitter_python` module

#### Current Chunking Strategy
- **File**: `/src/services/chunking_strategies.py`
- `PythonChunkingStrategy.get_node_mappings()` currently maps:
  - `ChunkType.FUNCTION: ["function_definition"]`
  - `ChunkType.CLASS: ["class_definition"]`
  - `ChunkType.IMPORT: ["import_statement", "import_from_statement"]`
  - No function call detection currently implemented

#### AST Extraction Service
- **File**: `/src/services/ast_extraction_service.py`
- Provides core AST traversal and chunk extraction
- Ready to be extended with call detection capabilities

## Identified Function Call Patterns to Support

### 1. Direct Function Calls
- `function_name()` - Simple function invocation
- `module.function()` - Module-level function calls
- `package.module.function()` - Nested module calls

### 2. Method Calls
- `object.method()` - Instance method calls
- `self.method()` - Self method calls within classes
- `cls.classmethod()` - Class method calls
- `Type.staticmethod()` - Static method calls

### 3. Attribute Calls
- `obj.attr.method()` - Chained attribute access with method call
- `obj.property()` - Property that returns callable
- `obj[key].method()` - Subscript with method call

### 4. Async Call Patterns
- `await function()` - Awaited function calls
- `await obj.method()` - Awaited method calls
- `asyncio.gather()` - Async utility functions
- `asyncio.create_task()` - Task creation functions

### 5. Special Python Patterns
- `super().method()` - Super method calls
- `callable(*args, **kwargs)` - Dynamic calls with unpacking
- `getattr(obj, 'method')()` - Dynamic attribute access calls

## Next Steps for Task 1.2
1. Create Tree-sitter query patterns for each call type
2. Define capture groups for extracting call metadata
3. Handle edge cases like nested calls and complex expressions
4. Integrate with existing chunking infrastructure

## Architecture Implications
- Need new `ChunkType.FUNCTION_CALL` for tracking call relationships
- Extend `get_node_mappings()` to include call node types
- Function calls will be extracted as separate chunks with metadata linking to their context
- Call detection should complement, not replace, existing function/class extraction

## References
- Tree-sitter Python grammar: https://github.com/tree-sitter/tree-sitter-python
- Tree-sitter query syntax: https://tree-sitter.github.io/tree-sitter/using-parsers/queries/1-syntax.html
