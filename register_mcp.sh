#!/bin/bash

# Script to register this MCP server with Claude Code
# Run this script to add the codebase RAG MCP server to Claude Code

echo "Registering Codebase RAG MCP Server with Claude Code..."

# Get the absolute path to the MCP runner
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PATH="$SCRIPT_DIR/.venv/bin/python"
MCP_RUNNER="$SCRIPT_DIR/src/run_mcp.py"

echo "Project directory: $SCRIPT_DIR"
echo "Python path: $PYTHON_PATH"
echo "MCP runner: $MCP_RUNNER"

# Verify files exist
if [ ! -f "$PYTHON_PATH" ]; then
    echo "❌ Error: Python virtual environment not found at $PYTHON_PATH"
    echo "Please run 'poetry install' first."
    exit 1
fi

if [ ! -f "$MCP_RUNNER" ]; then
    echo "❌ Error: MCP runner not found at $MCP_RUNNER"
    exit 1
fi

# Test the MCP server can start
echo ""
echo "Testing MCP server startup..."
echo '{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "1.0.0", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}}}' | "$PYTHON_PATH" "$MCP_RUNNER" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ MCP server test successful"
else
    echo "❌ MCP server test failed"
    exit 1
fi

echo ""
echo "📋 Claude Code Registration Instructions:"
echo ""
echo "🎯 Recommended: Use the wrapper script (this should work reliably):"
echo ""
echo "claude mcp add codebase-rag-mcp '$SCRIPT_DIR/mcp_server'"
echo ""
echo "🔧 Alternative methods if the above doesn't work:"
echo "claude mcp add codebase-rag-mcp '$PYTHON_PATH' '$MCP_RUNNER'"
echo "claude mcp add codebase-rag-mcp '$PYTHON_PATH $MCP_RUNNER'"
echo ""
echo "After registration, you can use these tools in Claude Code:"
echo "- @codebase-rag-mcp:index_directory - Index files in a directory"
echo "- @codebase-rag-mcp:search - Search indexed content with natural language"
echo "- @codebase-rag-mcp:health_check - Check server status"
echo ""
echo "💡 If you encounter ENOENT errors, make sure to use the wrapper script path!"