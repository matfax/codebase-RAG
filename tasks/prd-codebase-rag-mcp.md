# Product Requirements Document: Codebase RAG MCP Server

## 1. Introduction/Overview

This document outlines the requirements for a Retrieval-Augmented Generation (RAG) MCP (Model-Controller-Provider) server designed to help AI assistants and junior developers quickly understand and navigate codebases. The server will use Qdrant as its vector database to store and retrieve codebase embeddings. Users will be able to index local directories or GitHub repositories and then query the codebase using natural language.

## 2. Goals

*   **G-1:** Provide a fast and efficient way for AI assistants and developers to understand the architecture and implementation details of a codebase.
*   **G-2:** Enable natural language queries to explore and understand a codebase, reducing the learning curve for new developers.
*   **G-3:** Allow users to index new codebases (from local directories or GitHub) easily.
*   **G-4:** Offer flexibility by allowing users to select different embedding models, including local models available through Ollama.

## 3. User Stories

*   **U-1:** As an AI assistant, I want to be able to query a codebase with natural language to find specific functions, understand class structures, and trace dependencies, so that I can provide accurate and context-aware coding assistance.
*   **U-2:** As a junior developer joining a new project, I want to index the project's GitHub repository and ask questions like "Where is the authentication logic handled?" or "What does the `ApiService` class do?" to quickly get up to speed.
*   **U-3:** As a developer, I want to index my local project directory and specify which embedding model to use (e.g., a model from Ollama), so I can experiment with different models to find the best one for my codebase.
*   **U-4:** As a developer, when I point the MCP server to a project directory, I want it to automatically identify the main programming language, dependencies (like `package.json` or `requirements.txt`), and source code files, so that I don't have to manually configure what to index.

## 4. Functional Requirements

*   **F-1:** The system must provide an endpoint to index a codebase from a local directory path.
*   **F-2:** The system must provide an endpoint to index a codebase from a GitHub repository URL.
*   **F-3:** The system must allow the user to specify the embedding model to be used for indexing. It should support models available via Ollama.
*   **F-4:** The system must parse the codebase, generate vector embeddings for code files, and store them in a Qdrant collection.
*   **F-5:** The system must provide a query endpoint that accepts a natural language question.
*   **F-6:** The query endpoint must convert the user's question into a vector, search for the most relevant code snippets in Qdrant, and return them as context.
*   **F-7:** The system should be able to handle various programming languages commonly found in modern software projects.
*   **F-8:** When indexing a local directory, the system must analyze its contents (e.g., package manager files, file extensions) to automatically identify the project type and the most relevant files to include in the index.
*   **F-9:** If the server is running on a macOS environment, it must automatically attempt to use Metal Performance Shaders (MPS) to accelerate embedding generation and other machine learning computations.

## 5. Non-Goals (Out of Scope)

*   This version will not provide a graphical user interface (GUI). Interaction will be via API endpoints.
*   This version will not manage user authentication or access control for the indexed codebases.
*   This version will not support real-time, continuous indexing. Indexing is an on-demand process initiated by the user.

## 6. Design Considerations (Optional)

*   The API should be well-documented (e.g., using OpenAPI/Swagger) to facilitate its use by AI assistants and developers.
*   The system should be designed to be extensible, allowing for the potential addition of new vector databases or embedding model providers in the future.

## 7. Technical Considerations (Optional)

*   The server will be built using a modern web framework (e.g., FastAPI for Python, Express for Node.js).
*   The system will interact with Qdrant via its official client library.
*   The system will need to be able to interact with a running Ollama instance to use local embedding models.
*   The system must include logic to detect the host operating system. If `darwin` (macOS) is detected, it should configure the underlying machine learning libraries (e.g., PyTorch) to use the `mps` device for hardware acceleration.

## 8. Success Metrics

*   **S-1:** Time required for a new developer to answer a specific question about the codebase is reduced by 50%.
*   **S-2:** The accuracy of the retrieved code snippets for a set of benchmark questions is above 85%.
*   **S-3:** The server can successfully index a 100,000-line-of-code repository in under 10 minutes.

## 9. Open Questions

*   What is the preferred programming language and framework for the server? (e.g., Python with FastAPI, Node.js with Express)
*   How should the server handle dependencies and environment setup for analyzing different programming languages in the codebases?
*   What is the strategy for chunking code files before generating embeddings to ensure optimal retrieval?
