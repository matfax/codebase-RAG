import ollama
import torch
import platform
import os
import logging
import time
import random
import functools
from typing import List, Union, Optional, Tuple, Callable

class EmbeddingService:
    def __init__(self):
        self.device = self._get_device()
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434") # Default Ollama host
        self.embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
        self.max_batch_chars = int(os.getenv("MAX_BATCH_CHARS", "50000"))  # Max chars per batch
        
        # Retry configuration
        self.max_retries = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
        self.base_delay = float(os.getenv("EMBEDDING_RETRY_BASE_DELAY", "1.0"))  # Base delay in seconds
        self.max_delay = float(os.getenv("EMBEDDING_RETRY_MAX_DELAY", "60.0"))   # Max delay in seconds
        self.backoff_multiplier = float(os.getenv("EMBEDDING_BACKOFF_MULTIPLIER", "2.0"))
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._setup_logging()

    def _get_device(self):
        if platform.system() == 'Darwin':
            if torch.backends.mps.is_available():
                self.logger.info("MPS is available. Using Metal for acceleration.")
                return torch.device("mps")
            else:
                self.logger.info("MPS not available, using CPU.")
        return torch.device("cpu")

    def _retry_with_exponential_backoff(self, func: Callable) -> Callable:
        """Decorator for implementing exponential backoff retry logic."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_retries + 1):  # +1 for initial attempt
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Don't retry on final attempt
                    if attempt == self.max_retries:
                        break
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(
                        self.base_delay * (self.backoff_multiplier ** attempt),
                        self.max_delay
                    )
                    
                    # Add jitter (random variation) to prevent thundering herd
                    jitter = random.uniform(0.1, 0.3) * delay
                    total_delay = delay + jitter
                    
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{self.max_retries + 1} failed: {e}. "
                        f"Retrying in {total_delay:.2f}s..."
                    )
                    
                    time.sleep(total_delay)
            
            # If we get here, all retries failed
            self.logger.error(f"All {self.max_retries + 1} attempts failed. Final error: {last_exception}")
            raise last_exception
        
        return wrapper

    def _should_retry_exception(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry."""
        # Define retryable exceptions (connection errors, timeouts, rate limits)
        retryable_errors = [
            "connection",
            "timeout",
            "rate limit",
            "server error",
            "503",
            "502", 
            "504",
            "429"  # Too Many Requests
        ]
        
        error_str = str(exception).lower()
        return any(error_type in error_str for error_type in retryable_errors)

    def _split_oversized_batch(self, texts: List[str]) -> List[List[str]]:
        """Split a batch that's too large for processing."""
        if len(texts) <= 1:
            return [texts]  # Can't split further
        
        mid = len(texts) // 2
        left_batch = texts[:mid]
        right_batch = texts[mid:]
        
        self.logger.info(f"Splitting batch of {len(texts)} into batches of {len(left_batch)} and {len(right_batch)}")
        
        return [left_batch, right_batch]

    def generate_embeddings(self, model: str, text: Union[str, List[str]]) -> Optional[Union[torch.Tensor, List[torch.Tensor]]]:
        """Generate embeddings for single text or batch of texts.
        
        Args:
            model: The embedding model to use
            text: Single text string or list of text strings
            
        Returns:
            Single tensor for single text, list of tensors for batch, or None on error
        """
        # Handle both single text and batch processing
        if isinstance(text, str):
            return self._generate_single_embedding(model, text)
        elif isinstance(text, list):
            return self._generate_batch_embeddings(model, text)
        else:
            self.logger.error(f"Invalid input type for text: {type(text)}")
            return None
    
    def _generate_single_embedding(self, model: str, text: str) -> Optional[torch.Tensor]:
        """Generate embedding for a single text (backward compatibility)."""
        try:
            # Handle empty or whitespace-only text
            if not text or not text.strip():
                self.logger.warning("Empty or whitespace-only text provided for embedding")
                return None
            
            # Note: The Ollama library itself doesn't directly expose device selection.
            # The torch device is set for potential future use with local models that might run via PyTorch.
            # For now, this primarily serves the requirement of detecting and acknowledging MPS support.
            
            # Create Ollama client with host configuration
            client = ollama.Client(host=self.ollama_host)
            response = client.embeddings(model=model, prompt=text)
            
            # Check if embedding is empty
            if not response.get("embedding") or len(response["embedding"]) == 0:
                self.logger.warning(f"Received empty embedding for text: {text[:50]}...")
                return None
            
            # Convert to torch tensor for consistency
            import numpy as np
            embedding_array = np.array(response["embedding"])
            return torch.tensor(embedding_array, dtype=torch.float32)
            
        except Exception as e:
            self.logger.error(f"An error occurred while generating single embedding: {e}")
            return None
    
    def _generate_batch_embeddings(self, model: str, texts: List[str]) -> Optional[List[torch.Tensor]]:
        """Generate embeddings for multiple texts using intelligent batching."""
        if not texts:
            self.logger.warning("Empty text list provided for batch embedding")
            return []
        
        self.logger.info(f"Generating embeddings for batch of {len(texts)} texts using intelligent batching")
        
        try:
            # Split texts into optimal batches
            batches = self._create_intelligent_batches(texts)
            self.logger.info(f"Created {len(batches)} intelligent batches for processing")
            
            all_embeddings = []
            
            for batch_idx, batch_texts in enumerate(batches):
                self.logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} with {len(batch_texts)} texts")
                batch_embeddings = self._process_single_batch(model, batch_texts)
                
                if batch_embeddings is None:
                    self.logger.error(f"Failed to process batch {batch_idx + 1}")
                    # Add None placeholders for failed batch
                    all_embeddings.extend([None] * len(batch_texts))
                else:
                    all_embeddings.extend(batch_embeddings)
            
            successful_count = sum(1 for emb in all_embeddings if emb is not None)
            self.logger.info(f"Successfully generated {successful_count}/{len(texts)} embeddings")
            
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"An error occurred while generating batch embeddings: {e}")
            return None
    
    def _create_intelligent_batches(self, texts: List[str]) -> List[List[str]]:
        """Create intelligent batches based on content size and batch limits."""
        if not texts:
            return []
        
        batches = []
        current_batch = []
        current_batch_chars = 0
        
        for text in texts:
            text_length = len(text) if text else 0
            
            # Check if adding this text would exceed limits
            would_exceed_chars = current_batch_chars + text_length > self.max_batch_chars
            would_exceed_count = len(current_batch) >= self.embedding_batch_size
            
            # If current batch would be exceeded, start a new batch
            if current_batch and (would_exceed_chars or would_exceed_count):
                batches.append(current_batch)
                current_batch = []
                current_batch_chars = 0
            
            # Add text to current batch
            current_batch.append(text)
            current_batch_chars += text_length
            
            # Handle oversized single texts by putting them in their own batch
            if text_length > self.max_batch_chars:
                self.logger.warning(f"Text exceeds max batch size ({text_length} > {self.max_batch_chars} chars), processing individually")
                if len(current_batch) > 1:
                    # Move the oversized text to its own batch
                    oversized_text = current_batch.pop()
                    current_batch_chars -= text_length
                    
                    # Save current batch and create new batch for oversized text
                    batches.append(current_batch)
                    batches.append([oversized_text])
                    current_batch = []
                    current_batch_chars = 0
        
        # Add remaining texts
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _process_single_batch(self, model: str, texts: List[str]) -> Optional[List[torch.Tensor]]:
        """Process a single batch of texts with retry logic and subdivision on failure."""
        return self._process_batch_with_retry(model, texts)
    
    def _process_batch_with_retry(self, model: str, texts: List[str], attempt_subdivision: bool = True) -> Optional[List[torch.Tensor]]:
        """Process batch with retry logic and optional subdivision on failure."""
        try:
            # Try processing the batch normally first
            return self._process_batch_core(model, texts)
            
        except Exception as e:
            # Check if we should retry this exception
            if not self._should_retry_exception(e):
                self.logger.error(f"Non-retryable error processing batch: {e}")
                return None
            
            # If batch subdivision is enabled and batch has more than 1 item, try splitting
            if attempt_subdivision and len(texts) > 1:
                self.logger.warning(f"Batch processing failed: {e}. Attempting batch subdivision...")
                
                try:
                    # Split the batch into smaller batches
                    sub_batches = self._split_oversized_batch(texts)
                    all_embeddings = []
                    
                    for sub_batch in sub_batches:
                        # Process each sub-batch without further subdivision to avoid infinite recursion
                        sub_embeddings = self._process_batch_with_retry(model, sub_batch, attempt_subdivision=False)
                        
                        if sub_embeddings is None:
                            # If sub-batch fails, add None placeholders
                            all_embeddings.extend([None] * len(sub_batch))
                        else:
                            all_embeddings.extend(sub_embeddings)
                    
                    return all_embeddings
                    
                except Exception as subdivision_error:
                    self.logger.error(f"Batch subdivision also failed: {subdivision_error}")
                    return None
            else:
                # Cannot subdivide further or subdivision disabled
                self.logger.error(f"Batch processing failed and cannot subdivide further: {e}")
                return None
    
    def _process_batch_core(self, model: str, texts: List[str]) -> Optional[List[torch.Tensor]]:
        """Core batch processing logic with retry decorator applied."""
        # Apply retry decorator to the core processing logic
        @self._retry_with_exponential_backoff
        def _core_processing():
            client = ollama.Client(host=self.ollama_host)
            embeddings = []
            
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    self.logger.warning(f"Skipping empty text at index {i}")
                    embeddings.append(None)
                    continue
                
                # Individual text processing with its own error handling
                try:
                    response = client.embeddings(model=model, prompt=text)
                    
                    if not response.get("embedding") or len(response["embedding"]) == 0:
                        self.logger.warning(f"Received empty embedding for text at index {i}: {text[:50]}...")
                        embeddings.append(None)
                        continue
                    
                    # Convert to torch tensor
                    import numpy as np
                    embedding_array = np.array(response["embedding"])
                    tensor = torch.tensor(embedding_array, dtype=torch.float32)
                    embeddings.append(tensor)
                    
                except Exception as text_error:
                    # For individual text errors, don't retry the whole batch
                    self.logger.error(f"Error generating embedding for text at index {i}: {text_error}")
                    embeddings.append(None)
            
            return embeddings
        
        try:
            return _core_processing()
        except Exception as e:
            self.logger.error(f"Core batch processing failed after all retries: {e}")
            return None
    
    def _setup_logging(self) -> None:
        """Setup logging configuration for embedding service."""
        if not self.logger.handlers:
            # Set level from environment or default to INFO
            log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
            self.logger.setLevel(getattr(logging, log_level, logging.INFO))
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # Prevent duplicate logging
            self.logger.propagate = False
    
    # Backward compatibility method (deprecated)
    def generate_embedding(self, model: str, text: str) -> Optional[torch.Tensor]:
        """Deprecated: Use generate_embeddings instead."""
        self.logger.warning("generate_embedding is deprecated, use generate_embeddings instead")
        return self._generate_single_embedding(model, text)
