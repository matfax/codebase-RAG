"# Integration Guide for Codebase RAG MCP Server

## IDE Integration

This MCP server integrates seamlessly with various AI development environments:

### Claude Code Integration

Register the server with Claude Code for use in conversations:

```bash
git clone https://github.com/bluewings1211/codebase-RAG.git
claude mcp add codebase-rag-mcp -- /path/to/this/repo/mcp_server
```

Once registered, you can use natural language queries like:
- "Find functions that handle file uploads"
- "Show me React components that use hooks"
- "Locate error handling patterns in this codebase"

### Gemini CLI Integration

Edit ~/.gemini/settings.json

```
{
  "mcpServers": {
    "codebaseRAG": {
      "command": "/Users/jeff/Documents/personal/Agentic-RAG/trees/query-caching-layer-wave/mcp_server"
    }
  }
}
```

### VS Code Integration

Open VS Code and Ctrl + Shift + P and search MCP

```
{
  "mcpServers": {
    "codebaseRAG": {
      "command": "/Users/jeff/Documents/personal/Agentic-RAG/trees/query-caching-layer-wave/mcp_server"
    }
  }
}
```

## MCP Tools

The server provides powerful MCP tools for intelligent codebase search and analysis:

### Primary Tool: `search` - Semantic Code Search

Search indexed codebases using natural language queries with function-level precision.

**Example Queries:**
- "Find functions that handle file uploads"
- "Show me React components that use useState hook"
- "Locate error handling patterns in Python"
- "Find database connection initialization code"

**Key Features:**
- **🔍 Function-Level Precision**: Returns specific functions, classes, and methods instead of entire files
- **📝 Natural Language**: Use conversational queries to find code
- **🌐 Cross-Project Search**: Search across all projects or target specific projects with `target_projects`
- **📚 Rich Context**: Include surrounding code for better understanding
- **⚡ Multiple Search Modes**: Semantic, keyword, or hybrid search strategies

### Additional Tools

For comprehensive functionality, additional tools are available:
- **`index_directory`**: Index a codebase for intelligent searching
- **`health_check`**: Verify server connectivity and status
- **`analyze_repository_tool`**: Get repository statistics and analysis

For complete tool documentation with parameters and examples, see [MCP_TOOLS.md](MCP_TOOLS.md).

## Integrating with AI Assistants

This MCP server can be integrated with various AI assistants and development tools.

### Other MCP Clients

For other MCP-compatible clients, configure them to run:
```bash
uv run python /path/to/this/repo/src/run_mcp.py
```

Refer to your specific MCP client documentation for configuration details."
