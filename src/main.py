from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from tools import register_tools

load_dotenv()  # Load environment variables from .env file

app = FastMCP("codebase-rag-mcp")

# Register all tools and prompts using the new modular system

register_tools(app)
