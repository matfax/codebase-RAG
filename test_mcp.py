#!/usr/bin/env python3
"""Test script for MCP server functionality."""

import asyncio
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import app


async def test_mcp_server():
    """Test MCP server tools and functionality."""
    print("Testing MCP Server...")

    try:
        # Test listing tools
        tools = await app.list_tools()
        print(f"Available tools: {tools}")

        # Test health check tool
        if hasattr(app, "call_tool"):
            health_result = await app.call_tool("health_check", {})
            print(f"Health check result: {health_result}")

        print("MCP Server test completed successfully!")
        return True

    except Exception as e:
        print(f"Error testing MCP server: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_mcp_server())
    sys.exit(0 if success else 1)
