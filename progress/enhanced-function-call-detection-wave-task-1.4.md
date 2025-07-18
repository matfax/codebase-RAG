# Task 1.4: Extend PythonChunkingStrategy.get_node_mappings()

## Overview
Successfully extended the `PythonChunkingStrategy` class to include function call detection node types in `get_node_mappings()`, enabling the existing chunking infrastructure to detect and process function calls as first-class chunks.

## Changes Made

### 1. Extended ChunkType Enum (`src/models/code_chunk.py`)

Added new chunk types for function call detection:

```python
# Function call chunks (for relationship detection)
FUNCTION_CALL = "function_call"
METHOD_CALL = "method_call"
ASYNC_CALL = "async_call"
ATTRIBUTE_ACCESS = "attribute_access"
```

**Design Rationale:**
- **FUNCTION_CALL**: Direct function invocations (`function_name()`)
- **METHOD_CALL**: Object method calls (`obj.method()`)
- **ASYNC_CALL**: Async await expressions (`await function()`)
- **ATTRIBUTE_ACCESS**: Attribute/property access (`obj.attr`)

### 2. Updated Node Mappings (`src/services/chunking_strategies.py`)

Extended `PythonChunkingStrategy.get_node_mappings()`:

```python
def get_node_mappings(self) -> dict[ChunkType, list[str]]:
    """Get Python-specific AST node type mappings."""
    return {
        # Core code structure chunks
        ChunkType.FUNCTION: ["function_definition"],
        ChunkType.CLASS: ["class_definition"],
        ChunkType.CONSTANT: ["assignment"],  # Filtered by context
        ChunkType.VARIABLE: ["assignment"],
        ChunkType.IMPORT: ["import_statement", "import_from_statement"],

        # Function call and relationship detection chunks
        ChunkType.FUNCTION_CALL: ["call"],
        ChunkType.METHOD_CALL: ["call"],  # Filtered by attribute context
        ChunkType.ASYNC_CALL: ["await"],
        ChunkType.ATTRIBUTE_ACCESS: ["attribute"],
    }
```

**Integration Strategy:**
- Maintains backward compatibility with existing chunk types
- Reuses `"call"` node type for both function and method calls (distinguished by filtering)
- Uses `"await"` node type specifically for async calls
- Uses `"attribute"` node type for attribute access detection

### 3. Enhanced Chunk Filtering (`should_include_chunk`)

Extended `should_include_chunk()` method with intelligent filtering:

```python
elif chunk_type == ChunkType.FUNCTION_CALL:
    # Include function calls for relationship detection
    return self._is_significant_function_call(node)

elif chunk_type == ChunkType.METHOD_CALL:
    # Include method calls for relationship detection
    return self._is_method_call(node)

elif chunk_type == ChunkType.ASYNC_CALL:
    # Include async calls for relationship detection
    return True

elif chunk_type == ChunkType.ATTRIBUTE_ACCESS:
    # Include significant attribute access for relationship detection
    return self._is_significant_attribute_access(node)
```

**Filtering Logic:**
- **Function calls**: Filter out common builtins (`print`, `len`, etc.) to reduce noise
- **Method calls**: Include all calls with attribute access
- **Async calls**: Include all await expressions
- **Attribute access**: Filter out internal attributes (`__dict__`, `__class__`, etc.)

### 4. Added Helper Methods

#### `_is_significant_function_call(node: Node) -> bool`
```python
def _is_significant_function_call(self, node: Node) -> bool:
    """Check if a function call is significant enough to include as a chunk."""
    # Filters out common built-in functions that add noise
    common_builtins = {"print", "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple"}
    return function_name not in common_builtins
```

#### `_is_method_call(node: Node) -> bool`
```python
def _is_method_call(self, node: Node) -> bool:
    """Check if a call node represents a method call (obj.method())."""
    # Method calls have attribute access as the function
    return function_node.type == "attribute"
```

#### `_is_significant_attribute_access(node: Node) -> bool`
```python
def _is_significant_attribute_access(self, node: Node) -> bool:
    """Check if attribute access is significant for relationship detection."""
    # Filter out common internal attributes
    common_attrs = {"__dict__", "__class__", "__module__"}
    return attribute_name not in common_attrs
```

### 5. Enhanced Metadata Extraction

Extended `extract_additional_metadata()` to handle call-specific metadata:

```python
# Handle function call metadata
if chunk.chunk_type in [ChunkType.FUNCTION_CALL, ChunkType.METHOD_CALL]:
    call_metadata = self._extract_call_metadata(node)
    if call_metadata:
        metadata.update(call_metadata)

# Handle async call metadata
elif chunk.chunk_type == ChunkType.ASYNC_CALL:
    async_metadata = self._extract_async_call_metadata(node)
    if async_metadata:
        metadata.update(async_metadata)

# Handle attribute access metadata
elif chunk.chunk_type == ChunkType.ATTRIBUTE_ACCESS:
    attr_metadata = self._extract_attribute_metadata(node)
    if attr_metadata:
        metadata.update(attr_metadata)
```

### 6. Added Metadata Extraction Methods

#### `_extract_call_metadata(node: Node) -> dict`
Extracts detailed call information:
- `call_type`: "function" or "method"
- `function_name`: For direct function calls
- `object_name` & `method_name`: For method calls
- `argument_count`: Number of arguments passed

#### `_extract_async_call_metadata(node: Node) -> dict`
Extracts async-specific information:
- `is_async`: Always True for await expressions
- `call_type`: Prefixed with "async_" (e.g., "async_function", "async_method")
- Inherits all call metadata from nested call expression

#### `_extract_attribute_metadata(node: Node) -> dict`
Extracts attribute access information:
- `object_name`: Object being accessed
- `attribute_name`: Attribute/property name
- `is_chained`: Whether this is part of a chain (e.g., `a.b.c`)
- `chain_depth`: Length of attribute access chain

#### `_calculate_attribute_chain_depth(node: Node) -> int`
Calculates depth of chained attribute access:
- `a.b` = depth 2
- `a.b.c.d` = depth 4

## Integration Testing (`src/test_chunking_integration.py`)

Created comprehensive test script to validate the integration:

### Test Coverage
1. **Node Mapping Validation**: Ensures all call detection chunk types are present
2. **Chunk Extraction Testing**: Tests extraction of various call patterns
3. **Metadata Validation**: Verifies call-specific metadata is extracted correctly
4. **Helper Method Testing**: Tests filtering and detection logic

### Sample Test Results Expected
```
Node mappings:
  function: ['function_definition']
  class: ['class_definition']
  function_call: ['call']
  method_call: ['call']
  async_call: ['await']
  attribute_access: ['attribute']

Extracted chunks:
  function_call: process_data (lines 3-3)
    call_type: function
    function_name: process_data
    argument_count: 1

  method_call: user.save (lines 7-7)
    call_type: method
    object_name: user
    method_name: save
    argument_count: 0
```

## Architecture Integration

### Backward Compatibility
- Existing chunk types remain unchanged
- No modifications to core chunking infrastructure
- New chunk types are additive only

### Performance Considerations
- Intelligent filtering reduces noise from common patterns
- Reuses existing Tree-sitter node types
- Minimal overhead in chunk processing pipeline

### Metadata Schema
Call chunks include rich metadata for relationship building:
```python
{
    "call_type": "method",           # function|method|async_function|async_method
    "function_name": "process",      # For function calls
    "object_name": "self",           # For method calls
    "method_name": "validate",       # For method calls
    "argument_count": 2,             # Number of arguments
    "is_async": True,                # For async calls
    "is_chained": False,             # For attribute access
    "chain_depth": 1                 # Depth of attribute chain
}
```

## Integration with Existing Services

### AST Extraction Service
- Leverages existing `extract_chunks()` infrastructure
- No changes required to core extraction logic
- Call detection happens through normal chunk processing

### Tree-sitter Manager
- Uses existing Python parser
- No changes to parser initialization
- Relies on standard Tree-sitter node types

### Code Parser Service
- Compatible with existing parsing pipeline
- Call chunks processed alongside function/class chunks
- No breaking changes to API

## Next Steps for Task 1.5

The infrastructure is now ready for testing against real Python codebases:

1. **Real Codebase Testing**: Apply to actual Python projects
2. **Pattern Validation**: Verify detection accuracy across different coding styles
3. **Performance Analysis**: Measure impact on parsing speed
4. **False Positive Analysis**: Identify and tune filtering criteria

## Success Criteria Met

✅ **Extended ChunkType enum** with call detection types
✅ **Updated get_node_mappings()** to include call node types
✅ **Enhanced chunk filtering** with intelligent selection criteria
✅ **Added metadata extraction** for call relationship building
✅ **Maintained backward compatibility** with existing chunking
✅ **Created integration tests** for validation

## Configuration Options

The implementation provides several configuration points:

### Filtering Tuning
```python
# In _is_significant_function_call()
common_builtins = {"print", "len", "str", "int", "float", "bool", "list", "dict", "set", "tuple"}

# In _is_significant_attribute_access()
common_attrs = {"__dict__", "__class__", "__module__"}
```

### Chunk Type Selection
Each call type can be enabled/disabled independently by modifying `get_node_mappings()` return dictionary.

### Metadata Granularity
Metadata extraction can be extended or simplified based on Graph RAG requirements.

This implementation provides a robust foundation for function call detection while maintaining compatibility with the existing chunking infrastructure.
