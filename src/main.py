from fastapi import FastAPI
from dotenv import load_dotenv

from .api.endpoints import router

load_dotenv() # Load environment variables from .env file

app = FastAPI(
    title="Codebase RAG MCP Server",
    description="API for indexing and querying codebases using RAG and Qdrant."
)

app.include_router(router)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def get_mcp_manifest():
    return {
        "name": "Codebase RAG MCP",
        "description": "A Model-Controller-Provider server for RAG-based codebase understanding.",
        "endpoints": [
            {
                "path": "/index",
                "method": "POST",
                "description": "Indexes a local directory or Git repository into a Qdrant collection."
            },
            {
                "path": "/query",
                "method": "POST",
                "description": "Queries an indexed codebase using a natural language question."
            }
        ]
    }
