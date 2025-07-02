#!/usr/bin/env python3
"""Demo script showing MCP server usage."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run_mcp_command(method, params=None):
    """Run a single MCP command and return the result."""
    request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}

    process = subprocess.Popen(
        [".venv/bin/python", "src/run_mcp.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        stdout, stderr = process.communicate(input=json.dumps(request) + "\n", timeout=30)
        if stderr:
            print(f"Server logs: {stderr}", file=sys.stderr)
        return json.loads(stdout.strip()) if stdout.strip() else None
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return None


def demo_mcp_functionality():
    """Demonstrate complete MCP server functionality."""
    print("🚀 Codebase RAG MCP Server Demo")
    print("=" * 50)

    # Initialize
    print("\n1. Initializing MCP Server...")
    response = run_mcp_command(
        "initialize",
        {
            "protocolVersion": "1.0.0",
            "capabilities": {},
            "clientInfo": {"name": "demo-client", "version": "1.0.0"},
        },
    )

    if response and "result" in response:
        print("✅ Server initialized successfully")
        print(f"   Server: {response['result']['serverInfo']['name']}")
        print(f"   Version: {response['result']['serverInfo']['version']}")
    else:
        print("❌ Initialization failed")
        return False

    print("\n2. Testing available tools...")
    print("   Available MCP tools:")
    print("   - health_check: Check server health")
    print("   - index_directory: Index files in a directory")
    print("   - search: Search indexed content")

    print("\n3. Creating test project...")
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_py = Path(temp_dir) / "example.py"
        test_py.write_text(
            '''
def fibonacci(n):
    """Calculate fibonacci sequence."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def add(self, a, b):
        return a + b

    def multiply(self, a, b):
        return a * b
'''
        )

        readme_md = Path(temp_dir) / "README.md"
        readme_md.write_text(
            """
# Example Project

This is a demo project with:
- Fibonacci function
- Calculator class
"""
        )

        print(f"   Created test project in: {temp_dir}")
        print(f"   Files: {list(Path(temp_dir).glob('*'))}")

        print("\n4. Indexing test project...")
        index_response = run_mcp_command(
            "tools/call",
            {
                "name": "index_directory",
                "arguments": {"directory": temp_dir, "clear_existing": True},
            },
        )

        if index_response and "result" in index_response:
            result = json.loads(index_response["result"]["content"][0]["text"])
            if "error" not in result:
                print("✅ Indexing successful")
                print(f"   Indexed files: {len(result.get('indexed_files', []))}")
                print(f"   Collections: {result.get('collections', [])}")
            else:
                print(f"❌ Indexing failed: {result['error']}")
                return False
        else:
            print("❌ Indexing request failed")
            return False

        print("\n5. Searching for 'fibonacci'...")
        search_response = run_mcp_command(
            "tools/call",
            {
                "name": "search",
                "arguments": {
                    "query": "fibonacci function",
                    "n_results": 3,
                    "cross_project": True,
                },
            },
        )

        if search_response and "result" in search_response:
            result = json.loads(search_response["result"]["content"][0]["text"])
            if "results" in result and result["results"]:
                print("✅ Search successful")
                print(f"   Found {len(result['results'])} results")
                best_result = result["results"][0]
                print(f"   Best match (score: {best_result['score']:.3f}):")
                print(f"   File: {best_result['display_path']}")
                print(f"   Content preview: {best_result['content'][:100]}...")
            else:
                print("⚠️  No search results found")
        else:
            print("❌ Search request failed")

    print("\n🎉 Demo completed successfully!")
    print("\nTo use this MCP server with Claude Code:")
    print("1. Run: ./register_mcp.sh")
    print("2. Start Claude Code")
    print("3. Use @codebase-rag-mcp tools in your prompts")

    return True


if __name__ == "__main__":
    success = demo_mcp_functionality()
    sys.exit(0 if success else 1)
