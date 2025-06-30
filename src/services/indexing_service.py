import os
import tempfile
import shutil
from git import Repo, GitCommandError

from .project_analysis_service import ProjectAnalysisService
from .embedding_service import EmbeddingService
from .qdrant_service import QdrantService
from qdrant_client import models

class IndexingService:
    def __init__(self, qdrant_host: str = 'localhost', qdrant_port: int = 6333):
        self.project_analysis_service = ProjectAnalysisService()
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService(host=qdrant_host, port=qdrant_port)

    def index_codebase(self, source_path: str, collection_name: str, embedding_model: str):
        print(f"Indexing codebase from: {source_path}")

        is_git_url = source_path.startswith(('http://', 'https://', 'git@'))

        if is_git_url:
            temp_dir = tempfile.mkdtemp()
            try:
                print(f"Cloning {source_path} into {temp_dir}")
                Repo.clone_from(source_path, temp_dir)
                directory_to_index = temp_dir
            except GitCommandError as e:
                print(f"Error cloning repository: {e}")
                shutil.rmtree(temp_dir)
                return
        else:
            directory_to_index = source_path

        relevant_files = self.project_analysis_service.get_relevant_files(directory_to_index)

        if not relevant_files:
            print("No relevant files found to index.")
            if is_git_url: shutil.rmtree(temp_dir)
            return

        # For simplicity, we'll assume a fixed vector size for now. 
        # In a real application, this should be determined by the embedding model.
        # A common size for many models is 768 or 1536.
        # For Ollama, it depends on the model. Let's use a placeholder for now.
        vector_size = 768 # Placeholder, will need to be dynamic based on model

        self.qdrant_service.create_collection(collection_name, vector_size)

        points = []
        for file_path in relevant_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Generate embedding for the file content
                embedding = self.embedding_service.generate_embeddings(embedding_model, content)
                
                if embedding:
                    points.append(
                        models.PointStruct(
                            id=hash(file_path), # Simple hash for ID, consider more robust solution
                            vector=embedding,
                            payload={
                                "file_path": file_path,
                                "content": content # Storing content for retrieval, consider truncation for large files
                            }
                        )
                    )
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
        if points:
            self.qdrant_service.add_points(collection_name, points)
            print(f"Successfully indexed {len(points)} files into collection {collection_name}.")
        else:
            print("No points to add to Qdrant.")

        if is_git_url:
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
