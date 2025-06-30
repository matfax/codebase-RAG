#!/usr/bin/env python3
"""Test indexing this project itself."""

import asyncio
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import app

async def test_self_indexing():
    """Test indexing this project itself."""
    print("Testing self-indexing of Agentic-RAG project...")
    
    try:
        project_root = str(Path(__file__).parent)
        
        # Index this project
        print(f"\nIndexing project: {project_root}")
        index_result = await app.call_tool("index_directory", {
            "directory": project_root,
            "clear_existing": True
        })
        print(f"Indexing result: {index_result}")
        
        # Search for specific functionality
        print("\nSearching for 'MCP server'...")
        search_result = await app.call_tool("search", {
            "query": "MCP server",
            "n_results": 5
        })
        print(f"Search result: {search_result}")
        
        # Search for FastMCP usage
        print("\nSearching for 'FastMCP'...")
        fastmcp_search = await app.call_tool("search", {
            "query": "FastMCP",
            "n_results": 3
        })
        print(f"FastMCP search: {fastmcp_search}")
        
        print("\n✅ Self-indexing test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during self-indexing test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_self_indexing())
    sys.exit(0 if success else 1)