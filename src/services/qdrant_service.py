from qdrant_client import QdrantClient

class QdrantService:
    def __init__(self, host='localhost', port=6333):
        self.client = QdrantClient(host=host, port=port)