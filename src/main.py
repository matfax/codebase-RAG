from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP


load_dotenv() # Load environment variables from .env file

app = FastMCP("codebase-rag-mcp")

import mcp_tools
mcp_tools.register_mcp_tools(app)