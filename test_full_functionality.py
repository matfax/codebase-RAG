#!/usr/bin/env python3
"""Full functionality test for the MCP server."""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import app


async def test_full_functionality():
    """Test complete MCP server functionality."""
    print("Testing full MCP Server functionality...")

    try:
        # Test 1: Health check
        print("\n1. Testing health check...")
        health_result = await app.call_tool("health_check", {})
        print(f"Health check: {health_result}")

        # Test 2: Create a small test directory
        print("\n2. Creating test directory...")
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text(
                """
def hello_world():
    \"\"\"A simple hello world function.\"\"\"
    return "Hello, World!"

class TestClass:
    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value
"""
            )

            # Test 3: Index the test directory
            print(f"\n3. Indexing test directory: {temp_dir}")
            index_result = await app.call_tool("index_directory", {"directory": temp_dir, "clear_existing": True})
            print(f"Indexing result: {index_result}")

            # Test 4: Search the indexed content
            print("\n4. Searching indexed content...")
            search_result = await app.call_tool(
                "search",
                {
                    "query": "hello world function",
                    "n_results": 3,
                    "cross_project": True,
                },
            )
            print(f"Search result: {search_result}")

            # Test 5: Search for class
            print("\n5. Searching for class...")
            class_search = await app.call_tool("search", {"query": "TestClass", "n_results": 2, "cross_project": True})
            print(f"Class search result: {class_search}")

        print("\n✅ All tests completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    success = await test_full_functionality()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
