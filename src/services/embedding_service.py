import ollama
import torch
import platform
import os

class EmbeddingService:
    def __init__(self):
        self.device = self._get_device()
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434") # Default Ollama host

    def _get_device(self):
        if platform.system() == 'Darwin':
            if torch.backends.mps.is_available():
                print("MPS is available. Using Metal for acceleration.")
                return torch.device("mps")
            else:
                print("MPS not available, using CPU.")
        return torch.device("cpu")

    def generate_embeddings(self, model: str, text: str):
        try:
            # Note: The Ollama library itself doesn't directly expose device selection.
            # The torch device is set for potential future use with local models that might run via PyTorch.
            # For now, this primarily serves the requirement of detecting and acknowledging MPS support.
            
            # Create Ollama client with host configuration
            client = ollama.Client(host=self.ollama_host)
            response = client.embeddings(model=model, prompt=text)
            
            # Convert to torch tensor for consistency
            import numpy as np
            embedding_array = np.array(response["embedding"])
            return torch.tensor(embedding_array, dtype=torch.float32)
            
        except Exception as e:
            print(f"An error occurred while generating embeddings: {e}")
            return None
