# Task 1.3: Async Call Detection Patterns

## Overview
Extended the Tree-sitter query patterns to include comprehensive async function call detection, covering await expressions, asyncio module functions, and async method calls.

## Async Patterns Added

### 1. Basic Await Patterns

#### Await Function Calls (`AWAIT_FUNCTION_CALL`)
```scheme
; Await function calls: await function(args)
(await
    (call
        function: (identifier) @async_call.function.name
        arguments: (argument_list) @async_call.arguments)) @async_call.await_function
```
- **Captures**: `@async_call.function.name`, `@async_call.arguments`
- **Examples**: `await fetch_data()`, `await process_async()`

#### Await Method Calls (`AWAIT_METHOD_CALL`)
```scheme
; Await method calls: await obj.method(args)
(await
    (call
        function: (attribute
            object: (_) @async_call.object
            attribute: (identifier) @async_call.method.name)
        arguments: (argument_list) @async_call.arguments)) @async_call.await_method
```
- **Captures**: `@async_call.object`, `@async_call.method.name`, `@async_call.arguments`
- **Examples**: `await obj.async_method()`, `await client.get_data()`

#### Await Self Method Calls (`AWAIT_SELF_METHOD_CALL`)
```scheme
; Await self method calls: await self.method(args)
(await
    (call
        function: (attribute
            object: (identifier) @async_call.self
            attribute: (identifier) @async_call.method.name)
        arguments: (argument_list) @async_call.arguments)) @async_call.await_self_method
(#eq? @async_call.self "self")
```
- **Predicate**: Ensures object is specifically "self"
- **Examples**: `await self.async_process()`, `await self.save_async()`

#### Await Chained Calls (`AWAIT_CHAINED_CALL`)
```scheme
; Await chained method calls: await obj.attr.method(args)
(await
    (call
        function: (attribute
            object: (attribute
                object: (_) @async_call.base_object
                attribute: (identifier) @async_call.intermediate_attr)
            attribute: (identifier) @async_call.method.name)
        arguments: (argument_list) @async_call.arguments)) @async_call.await_chained
```
- **Examples**: `await user.profile.async_update()`, `await config.database.async_connect()`

### 2. Asyncio Module Patterns

#### Asyncio.gather (`ASYNCIO_GATHER_CALL`)
```scheme
; Asyncio gather calls: asyncio.gather(coro1, coro2, ...)
(call
    function: (attribute
        object: (identifier) @asyncio_call.module
        attribute: (identifier) @asyncio_call.function)
    arguments: (argument_list) @asyncio_call.arguments) @asyncio_call.gather
(#eq? @asyncio_call.module "asyncio")
(#eq? @asyncio_call.function "gather")
```
- **Examples**: `asyncio.gather(task1, task2)`, `await asyncio.gather(*tasks)`

#### Asyncio.create_task (`ASYNCIO_CREATE_TASK_CALL`)
```scheme
; Asyncio create_task calls: asyncio.create_task(coro)
(call
    function: (attribute
        object: (identifier) @asyncio_call.module
        attribute: (identifier) @asyncio_call.function)
    arguments: (argument_list) @asyncio_call.arguments) @asyncio_call.create_task
(#eq? @asyncio_call.module "asyncio")
(#eq? @asyncio_call.function "create_task")
```
- **Examples**: `asyncio.create_task(coroutine)`, `asyncio.create_task(async_function())`

#### Asyncio.run (`ASYNCIO_RUN_CALL`)
```scheme
; Asyncio run calls: asyncio.run(coro)
(call
    function: (attribute
        object: (identifier) @asyncio_call.module
        attribute: (identifier) @asyncio_call.function)
    arguments: (argument_list) @asyncio_call.arguments) @asyncio_call.run
(#eq? @asyncio_call.module "asyncio")
(#eq? @asyncio_call.function "run")
```
- **Examples**: `asyncio.run(async_main())`, `asyncio.run(main_coroutine())`

#### Asyncio.wait (`ASYNCIO_WAIT_CALL`)
```scheme
; Asyncio wait calls: asyncio.wait(tasks)
(call
    function: (attribute
        object: (identifier) @asyncio_call.module
        attribute: (identifier) @asyncio_call.function)
    arguments: (argument_list) @asyncio_call.arguments) @asyncio_call.wait
(#eq? @asyncio_call.module "asyncio")
(#eq? @asyncio_call.function "wait")
```
- **Examples**: `await asyncio.wait(tasks)`, `completed, pending = await asyncio.wait(tasks)`

#### Asyncio.wait_for (`ASYNCIO_WAIT_FOR_CALL`)
```scheme
; Asyncio wait_for calls: asyncio.wait_for(coro, timeout)
(call
    function: (attribute
        object: (identifier) @asyncio_call.module
        attribute: (identifier) @asyncio_call.function)
    arguments: (argument_list) @asyncio_call.arguments) @asyncio_call.wait_for
(#eq? @asyncio_call.module "asyncio")
(#eq? @asyncio_call.function "wait_for")
```
- **Examples**: `await asyncio.wait_for(coro, timeout=5.0)`

#### Generic Asyncio Calls (`ASYNCIO_GENERIC_CALL`)
```scheme
; Generic asyncio function calls: asyncio.function(args)
(call
    function: (attribute
        object: (identifier) @asyncio_call.module
        attribute: (identifier) @asyncio_call.function)
    arguments: (argument_list) @asyncio_call.arguments) @asyncio_call.generic
(#eq? @asyncio_call.module "asyncio")
```
- **Examples**: `asyncio.sleep()`, `asyncio.get_event_loop()`, `asyncio.set_event_loop()`

### 3. Combined Await + Asyncio Patterns

#### Await Asyncio Calls (`AWAIT_ASYNCIO_CALL`)
```scheme
; Await asyncio calls: await asyncio.function(args)
(await
    (call
        function: (attribute
            object: (identifier) @async_call.module
            attribute: (identifier) @async_call.function)
        arguments: (argument_list) @async_call.arguments)) @async_call.await_asyncio
(#eq? @async_call.module "asyncio")
```
- **Examples**: `await asyncio.sleep(1.0)`, `await asyncio.create_task(coro)`

## Updated Method Categories

### New Method Groups Added

1. **`get_async_patterns()`**: All await-based patterns
   - `await_function_call`
   - `await_method_call`
   - `await_self_method_call`
   - `await_chained_call`
   - `await_asyncio_call`

2. **`get_asyncio_patterns()`**: All asyncio module patterns
   - `asyncio_gather_call`
   - `asyncio_create_task_call`
   - `asyncio_run_call`
   - `asyncio_wait_call`
   - `asyncio_wait_for_call`
   - `asyncio_generic_call`

### Enhanced Basic Patterns
Updated `get_basic_patterns()` to include most common async patterns:
- Added `await_function_call`
- Added `await_method_call`

### Enhanced Advanced Patterns
Updated `get_advanced_patterns()` to include complex async patterns:
- Added `await_self_method_call`
- Added `await_chained_call`
- Added `await_asyncio_call`

## Node Type Updates

### Extended Node Types
Added `"await"` to both:
- `CALL_NODE_TYPES`: Primary async call detection
- `EXTENDED_CALL_NODE_TYPES`: Comprehensive async detection

Updated node type mappings support async patterns through the `"await"` node type.

## Test Case Coverage

### Added Async Test Cases

1. **Basic Await Calls**: 2 patterns
   - `await fetch_data()`
   - `await process_async()`

2. **Await Method Calls**: 3 patterns
   - `await obj.async_method()`
   - `await client.get_data()`
   - `await asyncio.sleep(1.0)`

3. **Await Self Method Calls**: 2 patterns
   - `await self.async_process()`
   - `await self.save_async()`

4. **Await Chained Calls**: 2 patterns
   - `await user.profile.async_update()`
   - `await config.database.async_connect()`

5. **Asyncio Calls**: 5 patterns
   - `asyncio.create_task(coroutine)`
   - `asyncio.create_task(async_function())`
   - `asyncio.run(async_main())`
   - `asyncio.get_event_loop()`
   - `asyncio.set_event_loop(new_loop)`

6. **Asyncio Await Calls**: 5 patterns
   - `await asyncio.gather(task1, task2, task3)`
   - `await asyncio.wait(tasks)`
   - `await asyncio.wait_for(coro, timeout=5.0)`
   - `await asyncio.create_task(coro)`
   - `await asyncio.gather(*tasks)`

## Tree-sitter Grammar Integration

### Await Node Structure
Based on Tree-sitter Python grammar research:
- `"await"` is a named node type
- Requires `primary_expression` as child
- Supports all expression types that can be awaited

### Async Function Context
- Patterns work within `async def` function contexts
- Compatible with async comprehensions
- Handles nested async calls correctly

## Performance Considerations

### Pattern Efficiency
- Specific predicates reduce false matches
- Hierarchical matching avoids redundant traversals
- Separate async/asyncio grouping enables selective detection

### Query Optimization
- Async patterns can be enabled/disabled independently
- Basic vs advanced async pattern separation
- Module-specific patterns (asyncio) for targeted detection

## Integration Strategy

### Chunking Strategy Integration
Async patterns integrate with existing `PythonChunkingStrategy`:
```python
# Additional node mappings for async
ChunkType.ASYNC_CALL: ["await"]
ChunkType.ASYNCIO_CALL: ["call"]  # With asyncio module filter
```

### Metadata Enhancement
Async call detection provides:
- **Call Type**: `await_function`, `await_method`, `asyncio_*`
- **Async Context**: Whether call is within await expression
- **Module Context**: Specific asyncio function identification
- **Chaining Level**: For complex async call chains

## Next Steps for Task 1.4
1. Integrate all call patterns (sync + async) into `PythonChunkingStrategy`
2. Extend `get_node_mappings()` to include call detection node types
3. Add `ChunkType.FUNCTION_CALL` enum value
4. Update chunking strategy to handle function call chunks

## Architecture Notes
- Async patterns maintain same capture group conventions
- Compatible with existing synchronous patterns
- Supports mixed async/sync codebases
- Designed for Graph RAG integration with call relationship detection

## Validation Requirements
1. Test patterns against real async Python codebases
2. Verify await expression detection accuracy
3. Validate asyncio module call identification
4. Performance testing on large async applications
5. False positive/negative analysis for async patterns
