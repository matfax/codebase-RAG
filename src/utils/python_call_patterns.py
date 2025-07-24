"""
Tree-sitter query patterns for detecting Python function calls.

This module contains Tree-sitter query patterns specifically designed for detecting
various types of function calls in Python code, including direct calls, method calls,
attribute calls, and async patterns.
"""

# from typing import Dict, List  # Not needed with modern Python


class PythonCallPatterns:
    """
    Tree-sitter query patterns for Python function call detection.

    This class contains pre-defined query patterns that can be used with Tree-sitter
    to detect different types of function calls in Python ASTs.
    """

    # Direct function call patterns
    DIRECT_FUNCTION_CALL = """
    ; Direct function calls: function_name(args)
    (call
        function: (identifier) @call.function.name
        arguments: (argument_list) @call.arguments) @call.direct
    """

    # Method call patterns (object.method())
    METHOD_CALL = """
    ; Method calls: object.method(args)
    (call
        function: (attribute
            object: (_) @call.object
            attribute: (identifier) @call.method.name)
        arguments: (argument_list) @call.arguments) @call.method
    """

    # Self method call patterns (self.method())
    SELF_METHOD_CALL = """
    ; Self method calls: self.method(args)
    (call
        function: (attribute
            object: (identifier) @call.self
            attribute: (identifier) @call.method.name)
        arguments: (argument_list) @call.arguments) @call.self_method
    (#eq? @call.self "self")
    """

    # Chained attribute call patterns (obj.attr.method())
    CHAINED_ATTRIBUTE_CALL = """
    ; Chained attribute calls: obj.attr.method(args)
    (call
        function: (attribute
            object: (attribute
                object: (_) @call.base_object
                attribute: (identifier) @call.intermediate_attr)
            attribute: (identifier) @call.method.name)
        arguments: (argument_list) @call.arguments) @call.chained
    """

    # Module function call patterns (module.function())
    MODULE_FUNCTION_CALL = """
    ; Module function calls: module.function(args)
    (call
        function: (attribute
            object: (identifier) @call.module.name
            attribute: (identifier) @call.function.name)
        arguments: (argument_list) @call.arguments) @call.module_function
    """

    # Subscript method call patterns (obj[key].method())
    SUBSCRIPT_METHOD_CALL = """
    ; Subscript method calls: obj[key].method(args)
    (call
        function: (attribute
            object: (subscript
                value: (_) @call.subscript.object
                subscript: (_) @call.subscript.key)
            attribute: (identifier) @call.method.name)
        arguments: (argument_list) @call.arguments) @call.subscript_method
    """

    # Super method call patterns (super().method())
    SUPER_METHOD_CALL = """
    ; Super method calls: super().method(args)
    (call
        function: (attribute
            object: (call
                function: (identifier) @call.super
                arguments: (argument_list))
            attribute: (identifier) @call.method.name)
        arguments: (argument_list) @call.arguments) @call.super_method
    (#eq? @call.super "super")
    """

    # Class method call patterns (cls.method() or Class.method())
    CLASS_METHOD_CALL = """
    ; Class method calls: cls.method(args) or Class.method(args)
    (call
        function: (attribute
            object: (identifier) @call.class.name
            attribute: (identifier) @call.method.name)
        arguments: (argument_list) @call.arguments) @call.class_method
    """

    # Dynamic attribute call patterns (getattr(obj, 'method')())
    DYNAMIC_ATTRIBUTE_CALL = """
    ; Dynamic attribute calls: getattr(obj, 'method')(args)
    (call
        function: (call
            function: (identifier) @call.getattr
            arguments: (argument_list
                (_) @call.target_object
                (string) @call.attribute_name))
        arguments: (argument_list) @call.arguments) @call.dynamic
    (#eq? @call.getattr "getattr")
    """

    # Callable with unpacking patterns (callable(*args, **kwargs))
    UNPACKING_CALL = """
    ; Calls with argument unpacking: callable(*args, **kwargs)
    (call
        function: (_) @call.function
        arguments: (argument_list
            (list_splat) @call.args_unpack
            (dictionary_splat) @call.kwargs_unpack)) @call.unpacking
    """

    # === ASYNC CALL PATTERNS ===

    # Basic await function call patterns (await function())
    AWAIT_FUNCTION_CALL = """
    ; Await function calls: await function(args)
    (await
        (call
            function: (identifier) @async_call.function.name
            arguments: (argument_list) @async_call.arguments)) @async_call.await_function
    """

    # Await method call patterns (await obj.method())
    AWAIT_METHOD_CALL = """
    ; Await method calls: await obj.method(args)
    (await
        (call
            function: (attribute
                object: (_) @async_call.object
                attribute: (identifier) @async_call.method.name)
            arguments: (argument_list) @async_call.arguments)) @async_call.await_method
    """

    # Await self method call patterns (await self.method())
    AWAIT_SELF_METHOD_CALL = """
    ; Await self method calls: await self.method(args)
    (await
        (call
            function: (attribute
                object: (identifier) @async_call.self
                attribute: (identifier) @async_call.method.name)
            arguments: (argument_list) @async_call.arguments)) @async_call.await_self_method
    (#eq? @async_call.self "self")
    """

    # Await chained method call patterns (await obj.attr.method())
    AWAIT_CHAINED_CALL = """
    ; Await chained method calls: await obj.attr.method(args)
    (await
        (call
            function: (attribute
                object: (attribute
                    object: (_) @async_call.base_object
                    attribute: (identifier) @async_call.intermediate_attr)
                attribute: (identifier) @async_call.method.name)
            arguments: (argument_list) @async_call.arguments)) @async_call.await_chained
    """

    # Asyncio.gather patterns - asyncio.gather(coroutines)
    ASYNCIO_GATHER_CALL = """
    ; Asyncio gather calls: asyncio.gather(coro1, coro2, ...)
    (call
        function: (attribute
            object: (identifier) @asyncio_call.module
            attribute: (identifier) @asyncio_call.function)
        arguments: (argument_list) @asyncio_call.arguments) @asyncio_call.gather
    (#eq? @asyncio_call.module "asyncio")
    (#eq? @asyncio_call.function "gather")
    """

    # Asyncio.create_task patterns - asyncio.create_task(coroutine)
    ASYNCIO_CREATE_TASK_CALL = """
    ; Asyncio create_task calls: asyncio.create_task(coro)
    (call
        function: (attribute
            object: (identifier) @asyncio_call.module
            attribute: (identifier) @asyncio_call.function)
        arguments: (argument_list) @asyncio_call.arguments) @asyncio_call.create_task
    (#eq? @asyncio_call.module "asyncio")
    (#eq? @asyncio_call.function "create_task")
    """

    # Asyncio.run patterns - asyncio.run(main())
    ASYNCIO_RUN_CALL = """
    ; Asyncio run calls: asyncio.run(coro)
    (call
        function: (attribute
            object: (identifier) @asyncio_call.module
            attribute: (identifier) @asyncio_call.function)
        arguments: (argument_list) @asyncio_call.arguments) @asyncio_call.run
    (#eq? @asyncio_call.module "asyncio")
    (#eq? @asyncio_call.function "run")
    """

    # Asyncio.wait patterns - asyncio.wait(tasks)
    ASYNCIO_WAIT_CALL = """
    ; Asyncio wait calls: asyncio.wait(tasks)
    (call
        function: (attribute
            object: (identifier) @asyncio_call.module
            attribute: (identifier) @asyncio_call.function)
        arguments: (argument_list) @asyncio_call.arguments) @asyncio_call.wait
    (#eq? @asyncio_call.module "asyncio")
    (#eq? @asyncio_call.function "wait")
    """

    # Asyncio.wait_for patterns - asyncio.wait_for(coro, timeout)
    ASYNCIO_WAIT_FOR_CALL = """
    ; Asyncio wait_for calls: asyncio.wait_for(coro, timeout)
    (call
        function: (attribute
            object: (identifier) @asyncio_call.module
            attribute: (identifier) @asyncio_call.function)
        arguments: (argument_list) @asyncio_call.arguments) @asyncio_call.wait_for
    (#eq? @asyncio_call.module "asyncio")
    (#eq? @asyncio_call.function "wait_for")
    """

    # Generic asyncio function calls (asyncio.*)
    ASYNCIO_GENERIC_CALL = """
    ; Generic asyncio function calls: asyncio.function(args)
    (call
        function: (attribute
            object: (identifier) @asyncio_call.module
            attribute: (identifier) @asyncio_call.function)
        arguments: (argument_list) @asyncio_call.arguments) @asyncio_call.generic
    (#eq? @asyncio_call.module "asyncio")
    """

    # Await with asyncio calls - await asyncio.function()
    AWAIT_ASYNCIO_CALL = """
    ; Await asyncio calls: await asyncio.function(args)
    (await
        (call
            function: (attribute
                object: (identifier) @async_call.module
                attribute: (identifier) @async_call.function)
            arguments: (argument_list) @async_call.arguments)) @async_call.await_asyncio
    (#eq? @async_call.module "asyncio")
    """

    @classmethod
    def get_all_patterns(cls) -> dict[str, str]:
        """
        Get all function call patterns as a dictionary.

        Returns:
            Dictionary mapping pattern names to query strings
        """
        return {
            # Synchronous call patterns
            "direct_function_call": cls.DIRECT_FUNCTION_CALL,
            "method_call": cls.METHOD_CALL,
            "self_method_call": cls.SELF_METHOD_CALL,
            "chained_attribute_call": cls.CHAINED_ATTRIBUTE_CALL,
            "module_function_call": cls.MODULE_FUNCTION_CALL,
            "subscript_method_call": cls.SUBSCRIPT_METHOD_CALL,
            "super_method_call": cls.SUPER_METHOD_CALL,
            "class_method_call": cls.CLASS_METHOD_CALL,
            "dynamic_attribute_call": cls.DYNAMIC_ATTRIBUTE_CALL,
            "unpacking_call": cls.UNPACKING_CALL,
            # Asynchronous call patterns
            "await_function_call": cls.AWAIT_FUNCTION_CALL,
            "await_method_call": cls.AWAIT_METHOD_CALL,
            "await_self_method_call": cls.AWAIT_SELF_METHOD_CALL,
            "await_chained_call": cls.AWAIT_CHAINED_CALL,
            "asyncio_gather_call": cls.ASYNCIO_GATHER_CALL,
            "asyncio_create_task_call": cls.ASYNCIO_CREATE_TASK_CALL,
            "asyncio_run_call": cls.ASYNCIO_RUN_CALL,
            "asyncio_wait_call": cls.ASYNCIO_WAIT_CALL,
            "asyncio_wait_for_call": cls.ASYNCIO_WAIT_FOR_CALL,
            "asyncio_generic_call": cls.ASYNCIO_GENERIC_CALL,
            "await_asyncio_call": cls.AWAIT_ASYNCIO_CALL,
        }

    @classmethod
    def get_basic_patterns(cls) -> dict[str, str]:
        """
        Get basic function call patterns (most common use cases).

        Returns:
            Dictionary mapping basic pattern names to query strings
        """
        return {
            "direct_function_call": cls.DIRECT_FUNCTION_CALL,
            "method_call": cls.METHOD_CALL,
            "self_method_call": cls.SELF_METHOD_CALL,
            "module_function_call": cls.MODULE_FUNCTION_CALL,
            "await_function_call": cls.AWAIT_FUNCTION_CALL,
            "await_method_call": cls.AWAIT_METHOD_CALL,
        }

    @classmethod
    def get_advanced_patterns(cls) -> dict[str, str]:
        """
        Get advanced function call patterns (less common, more complex).

        Returns:
            Dictionary mapping advanced pattern names to query strings
        """
        return {
            "chained_attribute_call": cls.CHAINED_ATTRIBUTE_CALL,
            "subscript_method_call": cls.SUBSCRIPT_METHOD_CALL,
            "super_method_call": cls.SUPER_METHOD_CALL,
            "class_method_call": cls.CLASS_METHOD_CALL,
            "dynamic_attribute_call": cls.DYNAMIC_ATTRIBUTE_CALL,
            "unpacking_call": cls.UNPACKING_CALL,
            "await_self_method_call": cls.AWAIT_SELF_METHOD_CALL,
            "await_chained_call": cls.AWAIT_CHAINED_CALL,
            "await_asyncio_call": cls.AWAIT_ASYNCIO_CALL,
        }

    @classmethod
    def get_async_patterns(cls) -> dict[str, str]:
        """
        Get async-specific function call patterns.

        Returns:
            Dictionary mapping async pattern names to query strings
        """
        return {
            "await_function_call": cls.AWAIT_FUNCTION_CALL,
            "await_method_call": cls.AWAIT_METHOD_CALL,
            "await_self_method_call": cls.AWAIT_SELF_METHOD_CALL,
            "await_chained_call": cls.AWAIT_CHAINED_CALL,
            "await_asyncio_call": cls.AWAIT_ASYNCIO_CALL,
        }

    @classmethod
    def get_asyncio_patterns(cls) -> dict[str, str]:
        """
        Get asyncio-specific function call patterns.

        Returns:
            Dictionary mapping asyncio pattern names to query strings
        """
        return {
            "asyncio_gather_call": cls.ASYNCIO_GATHER_CALL,
            "asyncio_create_task_call": cls.ASYNCIO_CREATE_TASK_CALL,
            "asyncio_run_call": cls.ASYNCIO_RUN_CALL,
            "asyncio_wait_call": cls.ASYNCIO_WAIT_CALL,
            "asyncio_wait_for_call": cls.ASYNCIO_WAIT_FOR_CALL,
            "asyncio_generic_call": cls.ASYNCIO_GENERIC_CALL,
        }

    @classmethod
    def get_combined_query(cls, pattern_types: list[str] = None) -> str:
        """
        Combine multiple patterns into a single query.

        Args:
            pattern_types: List of pattern names to include. If None, includes all basic patterns.

        Returns:
            Combined query string with all specified patterns
        """
        if pattern_types is None:
            patterns = cls.get_basic_patterns()
        else:
            all_patterns = cls.get_all_patterns()
            patterns = {name: all_patterns[name] for name in pattern_types if name in all_patterns}

        # Combine all patterns with OR logic
        combined_queries = []
        for pattern_name, query in patterns.items():
            # Add comment for pattern identification
            commented_query = f"  ; {pattern_name.replace('_', ' ').title()}\n{query.strip()}"
            combined_queries.append(commented_query)

        return "\n\n".join(combined_queries)


class PythonCallNodeTypes:
    """
    Python AST node types related to function calls.

    This class defines the Tree-sitter node types that should be included
    when detecting function calls in Python code.
    """

    # Primary call-related node types
    CALL_NODE_TYPES = [
        "call",  # Function/method calls
        "attribute",  # Attribute access (may lead to calls)
        "await",  # Async await expressions
    ]

    # Extended node types for comprehensive call detection
    EXTENDED_CALL_NODE_TYPES = [
        "call",  # Function/method calls
        "attribute",  # Attribute access
        "await",  # Async await expressions
        "subscript",  # Subscript access (obj[key])
        "argument_list",  # Function arguments
        "identifier",  # Function/method names
    ]

    # Node types that indicate potential callable objects
    CALLABLE_NODE_TYPES = [
        "function_definition",  # Function definitions
        "lambda",  # Lambda expressions
        "call",  # Calls that return callables
    ]

    @classmethod
    def get_node_mapping_for_calls(cls) -> dict[str, list[str]]:
        """
        Get node type mappings for function call detection.

        Returns:
            Dictionary mapping semantic types to Tree-sitter node types
        """
        return {
            "function_calls": cls.CALL_NODE_TYPES,
            "extended_calls": cls.EXTENDED_CALL_NODE_TYPES,
            "callable_objects": cls.CALLABLE_NODE_TYPES,
        }


# Example usage and testing functions
def demo_pattern_usage():
    """
    Demonstrate how to use the patterns with Tree-sitter.

    This function shows example usage of the patterns with Tree-sitter
    for educational purposes.
    """
    # patterns = PythonCallPatterns()  # Demo disabled for linting

    # Get all basic patterns
    # basic_patterns = patterns.get_basic_patterns()
    # print("Basic Patterns:")  # Disabled for linting
    # for name, pattern in basic_patterns.items():
    #     print(f"\n{name}:")
    #     print(pattern)

    # Get combined query
    # combined_query = patterns.get_combined_query(["direct_function_call", "method_call"])
    # print(f"\nCombined Query:\n{combined_query}")  # Disabled for linting

    # Get node types for chunking strategies
    # node_types = PythonCallNodeTypes.get_node_mapping_for_calls()
    # print(f"\nNode Type Mappings:\n{node_types}")  # Disabled for linting
    pass  # Demo function disabled for linting


if __name__ == "__main__":
    demo_pattern_usage()
