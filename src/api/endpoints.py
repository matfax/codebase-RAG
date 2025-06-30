from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.indexing_service import IndexingService
from ..services.embedding_service import EmbeddingService
from ..services.qdrant_service import QdrantService

router = APIRouter()
indexing_service = IndexingService() # Initialize the service
embedding_service = EmbeddingService()
qdrant_service = QdrantService()

class IndexRequest(BaseModel):
    source_path: str
    collection_name: str
    embedding_model: str

class QueryRequest(BaseModel):
    query_text: str
    collection_name: str
    embedding_model: str
    limit: int = 5

@router.post("/index")
async def index_codebase(request: IndexRequest):
    try:
        indexing_service.index_codebase(
            source_path=request.source_path,
            collection_name=request.collection_name,
            embedding_model=request.embedding_model
        )
        return {"message": "Indexing initiated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query")
async def query_codebase(request: QueryRequest):
    try:
        query_embedding = embedding_service.generate_embeddings(request.embedding_model, request.query_text)
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding.")

        search_result = qdrant_service.client.search(
            collection_name=request.collection_name,
            query_vector=query_embedding,
            limit=request.limit,
            with_payload=True
        )

        results = []
        for hit in search_result:
            results.append({
                "file_path": hit.payload["file_path"],
                "content": hit.payload["content"],
                "score": hit.score
            })
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
