from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP


load_dotenv() # Load environment variables from .env file

app = FastMCP("codebase-rag-mcp")

# Register tools using the new modular system
from tools import register_tools
register_tools(app)

# Legacy registration for tools not yet migrated
import mcp_tools
mcp_tools.register_mcp_tools(app)