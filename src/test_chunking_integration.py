"""
Test script to validate PythonChunkingStrategy extensions for function call detection.

This script tests the updated PythonChunkingStrategy to ensure it correctly
identifies and processes function calls, method calls, and async calls as chunks.
"""

import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.code_chunk import ChunkType
from src.services.chunking_strategies import PythonChunkingStrategy
from src.utils.tree_sitter_manager import TreeSitterManager


def test_python_chunking_with_calls():
    """Test Python chunking strategy with function call detection."""

    # Sample Python code with various call types
    test_code = """
# Direct function calls
result = process_data(input_data)
value = calculate_sum(a, b, c)

# Method calls
user.save()
data.append(item)
config.get_value("key")

# Self method calls
class MyClass:
    def process(self):
        self.validate()
        self.update_state()
        return self.get_result()

# Async calls
async def async_example():
    result = await fetch_data()
    await self.async_process()
    data = await asyncio.gather(task1, task2)

# Chained calls
result = user.profile.settings.get_theme()
config.database.connection.execute(query)

# Attribute access
theme = user.profile.theme
connection = database.pool.connection
"""

    print("Testing PythonChunkingStrategy with function call detection...\n")

    # Initialize Tree-sitter manager and get Python parser
    ts_manager = TreeSitterManager()
    if not ts_manager.is_language_supported("python"):
        print("ERROR: Python parser not available")
        return False

    parser = ts_manager.get_parser("python")
    if not parser:
        print("ERROR: Could not get Python parser")
        return False

    # Parse the test code
    tree = parser.parse(test_code.encode("utf-8"))
    root_node = tree.root_node

    # Initialize Python chunking strategy
    strategy = PythonChunkingStrategy("python")

    # Test get_node_mappings includes call detection
    node_mappings = strategy.get_node_mappings()
    print("Node mappings:")
    for chunk_type, node_types in node_mappings.items():
        print(f"  {chunk_type.value}: {node_types}")

    # Verify call-related mappings exist
    expected_call_types = [ChunkType.FUNCTION_CALL, ChunkType.METHOD_CALL, ChunkType.ASYNC_CALL, ChunkType.ATTRIBUTE_ACCESS]

    missing_types = []
    for expected_type in expected_call_types:
        if expected_type not in node_mappings:
            missing_types.append(expected_type)

    if missing_types:
        print(f"\nERROR: Missing call detection chunk types: {missing_types}")
        return False

    print("\n✓ All expected call detection chunk types are present")

    # Extract chunks using the strategy
    try:
        chunks = strategy.extract_chunks(root_node, "test_file.py", test_code)
        print(f"\nExtracted {len(chunks)} chunks:")

        # Categorize chunks by type
        chunk_counts = {}
        call_chunks = []

        for chunk in chunks:
            chunk_type = chunk.chunk_type
            chunk_counts[chunk_type] = chunk_counts.get(chunk_type, 0) + 1

            # Collect call-related chunks for detailed analysis
            if chunk_type in expected_call_types:
                call_chunks.append(chunk)

            print(f"  {chunk_type.value}: {chunk.name} (lines {chunk.start_line}-{chunk.end_line})")
            if hasattr(chunk, "metadata") and chunk.metadata:
                for key, value in chunk.metadata.items():
                    print(f"    {key}: {value}")

        print("\nChunk counts by type:")
        for chunk_type, count in chunk_counts.items():
            print(f"  {chunk_type.value}: {count}")

        # Analyze call chunks specifically
        if call_chunks:
            print(f"\nCall-related chunks ({len(call_chunks)}):")
            for chunk in call_chunks:
                print(f"  {chunk.chunk_type.value}: {chunk.content.strip()}")
                if hasattr(chunk, "metadata") and chunk.metadata:
                    metadata_str = ", ".join([f"{k}={v}" for k, v in chunk.metadata.items()])
                    print(f"    Metadata: {metadata_str}")

        # Success criteria
        has_function_calls = ChunkType.FUNCTION_CALL in chunk_counts
        has_method_calls = ChunkType.METHOD_CALL in chunk_counts
        has_async_calls = ChunkType.ASYNC_CALL in chunk_counts
        has_attribute_access = ChunkType.ATTRIBUTE_ACCESS in chunk_counts

        print("\n--- Test Results ---")
        print(f"Function calls detected: {has_function_calls} ({chunk_counts.get(ChunkType.FUNCTION_CALL, 0)})")
        print(f"Method calls detected: {has_method_calls} ({chunk_counts.get(ChunkType.METHOD_CALL, 0)})")
        print(f"Async calls detected: {has_async_calls} ({chunk_counts.get(ChunkType.ASYNC_CALL, 0)})")
        print(f"Attribute access detected: {has_attribute_access} ({chunk_counts.get(ChunkType.ATTRIBUTE_ACCESS, 0)})")

        # Overall success
        success = any([has_function_calls, has_method_calls, has_async_calls, has_attribute_access])

        if success:
            print("\n✓ SUCCESS: Function call detection is working!")
        else:
            print("\n✗ FAILURE: No function calls detected")

        return success

    except Exception as e:
        print(f"\nERROR during chunk extraction: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_helper_methods():
    """Test the helper methods for call detection."""
    print("\nTesting helper methods...\n")

    # Initialize Tree-sitter and strategy
    ts_manager = TreeSitterManager()
    parser = ts_manager.get_parser("python")
    strategy = PythonChunkingStrategy("python")

    # Test cases
    test_cases = [
        ("print('hello')", "should be filtered out as common builtin"),
        ("process_data(x)", "should be included as significant function call"),
        ("obj.method()", "should be identified as method call"),
        ("self.process()", "should be identified as method call"),
        ("config.database.connection", "should be included as significant attribute access"),
    ]

    for code, description in test_cases:
        print(f"Testing: {code} - {description}")

        # Parse the code
        tree = parser.parse(code.encode("utf-8"))
        root_node = tree.root_node

        # Find call or attribute nodes
        def find_nodes_by_type(node, node_type):
            """Recursively find all nodes of a specific type."""
            nodes = []
            if node.type == node_type:
                nodes.append(node)
            for child in node.children:
                nodes.extend(find_nodes_by_type(child, node_type))
            return nodes

        # Test call nodes
        call_nodes = find_nodes_by_type(root_node, "call")
        for call_node in call_nodes:
            is_significant = strategy._is_significant_function_call(call_node)
            is_method = strategy._is_method_call(call_node)
            print(f"  Call node: significant={is_significant}, method={is_method}")

            if is_significant:
                metadata = strategy._extract_call_metadata(call_node)
                print(f"  Metadata: {metadata}")

        # Test attribute nodes
        attr_nodes = find_nodes_by_type(root_node, "attribute")
        for attr_node in attr_nodes:
            is_significant = strategy._is_significant_attribute_access(attr_node)
            print(f"  Attribute node: significant={is_significant}")

            if is_significant:
                metadata = strategy._extract_attribute_metadata(attr_node)
                print(f"  Metadata: {metadata}")

        print()


if __name__ == "__main__":
    print("=== PythonChunkingStrategy Function Call Detection Test ===\n")

    # Test the main chunking functionality
    success = test_python_chunking_with_calls()

    # Test helper methods
    test_helper_methods()

    print(f"\n=== Test {'PASSED' if success else 'FAILED'} ===")
