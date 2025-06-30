import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.main import app

client = TestClient(app)

@pytest.fixture
def mock_indexing_service():
    with patch('src.api.endpoints.indexing_service') as mock_service:
        yield mock_service

@pytest.fixture
def mock_embedding_service():
    with patch('src.api.endpoints.embedding_service') as mock_service:
        yield mock_service

@pytest.fixture
def mock_qdrant_service():
    with patch('src.api.endpoints.qdrant_service') as mock_service:
        yield mock_service

def test_index_endpoint_success(mock_indexing_service):
    mock_indexing_service.index_codebase.return_value = None
    response = client.post(
        "/index",
        json={
            "source_path": "/tmp/test_repo",
            "collection_name": "test_collection",
            "embedding_model": "ollama/nomic-embed-text"
        }
    )
    assert response.status_code == 200
    assert response.json() == {"message": "Indexing initiated successfully."}
    mock_indexing_service.index_codebase.assert_called_once_with(
        source_path="/tmp/test_repo",
        collection_name="test_collection",
        embedding_model="ollama/nomic-embed-text"
    )

def test_index_endpoint_failure(mock_indexing_service):
    mock_indexing_service.index_codebase.side_effect = Exception("Indexing failed")
    response = client.post(
        "/index",
        json={
            "source_path": "/tmp/test_repo",
            "collection_name": "test_collection",
            "embedding_model": "ollama/nomic-embed-text"
        }
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Indexing failed"}

def test_query_endpoint_success(mock_embedding_service, mock_qdrant_service):
    mock_embedding_service.generate_embeddings.return_value = [0.1, 0.2, 0.3]
    
    mock_hit = MagicMock()
    mock_hit.payload = {"file_path": "/src/main.py", "content": "def main(): pass"}
    mock_hit.score = 0.95
    mock_qdrant_service.client.search.return_value = [mock_hit]

    response = client.post(
        "/query",
        json={
            "query_text": "What is the main function?",
            "collection_name": "test_collection",
            "embedding_model": "ollama/nomic-embed-text",
            "limit": 1
        }
    )
    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "file_path": "/src/main.py",
                "content": "def main(): pass",
                "score": 0.95
            }
        ]
    }
    mock_embedding_service.generate_embeddings.assert_called_once_with(
        "ollama/nomic-embed-text", "What is the main function?"
    )
    mock_qdrant_service.client.search.assert_called_once()

def test_query_endpoint_embedding_failure(mock_embedding_service):
    mock_embedding_service.generate_embeddings.return_value = None
    response = client.post(
        "/query",
        json={
            "query_text": "What is the main function?",
            "collection_name": "test_collection",
            "embedding_model": "ollama/nomic-embed-text"
        }
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "500: Failed to generate query embedding."}

def test_query_endpoint_qdrant_failure(mock_embedding_service, mock_qdrant_service):
    mock_embedding_service.generate_embeddings.return_value = [0.1, 0.2, 0.3]
    mock_qdrant_service.client.search.side_effect = Exception("Qdrant search error")
    response = client.post(
        "/query",
        json={
            "query_text": "What is the main function?",
            "collection_name": "test_collection",
            "embedding_model": "ollama/nomic-embed-text"
        }
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Qdrant search error"}
