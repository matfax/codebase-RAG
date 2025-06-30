from qdrant_client import QdrantClient, models

class QdrantService:
    def __init__(self, host='localhost', port=6333):
        self.client = QdrantClient(host=host, port=port)

    def create_collection(self, collection_name: str, vector_size: int):
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def add_points(self, collection_name: str, points: list):
        self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True
        )
