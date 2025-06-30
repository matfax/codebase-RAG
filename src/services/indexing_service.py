import os
import tempfile
import shutil
from git import Repo, GitCommandError
from typing import List, Dict, Any
from dataclasses import dataclass

from services.project_analysis_service import ProjectAnalysisService

@dataclass
class Chunk:
    content: str
    metadata: Dict[str, Any]

class IndexingService:
    def __init__(self):
        self.project_analysis_service = ProjectAnalysisService()

    def process_codebase_for_indexing(self, source_path: str) -> List[Chunk]:
        print(f"Processing codebase from: {source_path}")

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
                return []
        else:
            directory_to_index = source_path

        relevant_files = self.project_analysis_service.get_relevant_files(directory_to_index)

        if not relevant_files:
            print("No relevant files found to process.")
            if is_git_url: shutil.rmtree(temp_dir)
            return []

        chunks = []
        for file_path in relevant_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple chunking: treat entire file as one chunk for now
                # More sophisticated chunking can be added here (e.g., based on AST, lines, etc.)
                chunks.append(
                    Chunk(
                        content=content,
                        metadata={
                            "file_path": file_path,
                            "chunk_index": 0, # Assuming single chunk for now
                            "line_start": 1,
                            "line_end": len(content.splitlines()),
                            "language": self._detect_language(file_path) # Add language detection
                        }
                    )
                )
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
        if is_git_url:
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

        return chunks

    def _detect_language(self, file_path: str) -> str:
        # Basic language detection based on file extension
        extension = os.path.splitext(file_path)[1].lower()
        if extension == ".py":
            return "python"
        elif extension == ".js":
            return "javascript"
        elif extension == ".ts":
            return "typescript"
        elif extension == ".java":
            return "java"
        elif extension == ".go":
            return "go"
        elif extension == ".rs":
            return "rust"
        elif extension == ".md":
            return "markdown"
        elif extension == ".json":
            return "json"
        elif extension == ".yaml" or extension == ".yml":
            return "yaml"
        else:
            return "unknown"