from fastapi import FastAPI
from .api.endpoints import router

app = FastAPI(
    title="Codebase RAG MCP Server",
    description="API for indexing and querying codebases using RAG and Qdrant."
)

app.include_router(router)

@app.get("/health")
async def health_check():
    return {"status": "ok"}
