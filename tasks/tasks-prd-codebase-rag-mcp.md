## Relevant Files

- `src/main.py` - Main application entry point.
- `src/config.py` - Configuration management (e.g., for Qdrant, Ollama, Metal acceleration).
- `src/services/qdrant_service.py` - Service for interacting with Qdrant.
- `src/services/embedding_service.py` - Service for generating embeddings (interfacing with Ollama and handling Metal acceleration).
- `src/services/indexing_service.py` - Service for handling the codebase indexing logic, including automatic file detection.
- `src/services/project_analysis_service.py` - Service to analyze project structure and identify relevant files.
- `src/api/endpoints.py` - API endpoint definitions.
- `tests/test_indexing_service.py` - Unit tests for the indexing service.
- `tests/test_project_analysis_service.py` - Unit tests for the project analysis service.
- `tests/test_api.py` - Tests for the API endpoints.

### Notes

- Unit tests should be placed in the `tests/` directory.
- Use `pytest` to run tests.

## Tasks

- [x] 1.0 Setup Project Structure and Dependencies
  - [x] 1.1 Create the main project directory and `src` and `tests` subdirectories.
  - [x] 1.2 Initialize a Python project with `pyproject.toml` and add dependencies like `fastapi`, `uvicorn`, `qdrant-client`, `ollama`, and `torch`.
  - [x] 1.3 Create empty `__init__.py` files where necessary to define Python packages.
- [x] 2.0 Implement Core Services (Qdrant, Embedding with Metal Support)
  - [x] 2.1 Implement the `QdrantService` to connect to the Qdrant server, create collections, and add points.
  - [x] 2.2 Implement the `EmbeddingService` to generate embeddings using a specified model from Ollama.
  - [x] 2.3 In the `EmbeddingService`, add logic to detect if the OS is `darwin` (macOS) and if so, set the PyTorch device to `mps` for Metal acceleration.
- [x] 3.0 Implement Project Analysis Service
  - [x] 3.1 Implement the `ProjectAnalysisService` to scan a directory.
  - [x] 3.2 Add logic to the service to identify project type by looking for files like `package.json`, `requirements.txt`, `pom.xml`, etc.
  - [x] 3.3 Add logic to the service to create a list of relevant source code files to be indexed, ignoring files listed in `.gitignore`.
- [x] 4.0 Implement Codebase Indexing Logic
  - [x] 4.1 Implement the `IndexingService` that uses the other services to manage the end-to-end indexing process.
  - [x] 4.2 The service should take a directory or Git URL, use the `ProjectAnalysisService` to get the file list, read the files, and use the `EmbeddingService` to generate embeddings.
  - [x] 4.3 The service should then use the `QdrantService` to store the embeddings in the vector database.
- [x] 5.0 Implement the API Endpoints
  - [x] 5.1 In `api/endpoints.py`, create an `/index` endpoint that takes a path or URL and triggers the `IndexingService`.
  - [x] 5.2 Create a `/query` endpoint that takes a natural language question, generates an embedding for it, and queries Qdrant for relevant results.
  - [x] 5.3 Integrate the endpoints into the FastAPI application in `main.py`.
- [x] 6.0 Create Tests
  - [x] 6.1 Create unit tests for the `ProjectAnalysisService` to ensure it correctly identifies project files.
  - [x] 6.2 Create unit tests for the `EmbeddingService`, mocking the Ollama call and verifying Metal support is triggered on macOS.
  - [x] 6.3 Create integration tests for the `/index` and `/query` API endpoints.
