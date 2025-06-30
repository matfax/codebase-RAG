import os
import time
import logging
import gc
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

@dataclass
class BatchInsertionStats:
    """Statistics for batch insertion operations."""
    total_points: int = 0
    successful_insertions: int = 0
    failed_insertions: int = 0
    batch_count: int = 0
    total_duration: float = 0.0
    average_batch_size: float = 0.0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_points <= 0:
            return 0.0
        return (self.successful_insertions / self.total_points) * 100
    
    @property
    def insertions_per_second(self) -> float:
        """Calculate insertions per second rate."""
        if self.total_duration <= 0:
            return 0.0
        return self.successful_insertions / self.total_duration

class QdrantService:
    def __init__(self, host='localhost', port=6333):
        self.client = QdrantClient(host=host, port=port)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Configuration
        self.default_batch_size = int(os.getenv("QDRANT_BATCH_SIZE", "500"))
        self.max_retries = int(os.getenv("QDRANT_MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("QDRANT_RETRY_DELAY", "1.0"))
        
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration for Qdrant service."""
        if not self.logger.handlers:
            log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
            self.logger.setLevel(getattr(logging, log_level, logging.INFO))
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            self.logger.propagate = False
    
    def calculate_optimal_batch_size(self, points: List[PointStruct], base_batch_size: Optional[int] = None) -> int:
        """
        Calculate optimal batch size based on point dimensions and metadata size.
        
        Args:
            points: List of points to analyze
            base_batch_size: Base batch size to start from (uses default if None)
            
        Returns:
            Optimized batch size
        """
        if not points:
            return base_batch_size or self.default_batch_size
        
        if base_batch_size is None:
            base_batch_size = self.default_batch_size
        
        # Estimate memory usage per point
        sample_point = points[0]
        
        # Vector dimension (primary memory consumer)
        vector_size = len(sample_point.vector) if sample_point.vector else 768  # Default dimension
        
        # Estimate payload size (rough approximation)
        payload_size = len(str(sample_point.payload)) if sample_point.payload else 100
        
        # Estimate memory per point in bytes
        # Float32 vectors: 4 bytes per dimension + payload overhead
        estimated_memory_per_point = (vector_size * 4) + payload_size + 100  # 100 bytes overhead
        
        # Target maximum memory per batch (default: 50MB)
        max_memory_per_batch = int(os.getenv("QDRANT_MAX_BATCH_MEMORY_MB", "50")) * 1024 * 1024
        
        # Calculate optimal batch size based on memory
        memory_based_batch_size = max_memory_per_batch // estimated_memory_per_point
        
        # Use the smaller of memory-based and configured batch size
        optimal_size = min(memory_based_batch_size, base_batch_size)
        
        # Ensure minimum batch size of 1
        optimal_size = max(1, optimal_size)
        
        # Log optimization if size was adjusted
        if optimal_size != base_batch_size:
            self.logger.info(
                f"Optimized batch size: {base_batch_size} -> {optimal_size} "
                f"(vector_dim: {vector_size}, estimated_memory_per_point: {estimated_memory_per_point} bytes)"
            )
        
        return optimal_size
    
    def batch_upsert_with_retry(
        self, 
        collection_name: str, 
        points: List[PointStruct], 
        batch_size: Optional[int] = None,
        enable_optimization: bool = True
    ) -> BatchInsertionStats:
        """
        Perform batch upsert with automatic retry, optimization, and comprehensive error handling.
        
        Args:
            collection_name: Name of the collection to insert into
            points: List of points to insert
            batch_size: Override default batch size (optional)
            enable_optimization: Whether to optimize batch size based on point characteristics
            
        Returns:
            BatchInsertionStats with detailed insertion statistics
        """
        stats = BatchInsertionStats()
        stats.total_points = len(points)
        start_time = time.time()
        
        if not points:
            self.logger.warning("No points provided for batch upsert")
            return stats
        
        # Determine optimal batch size
        if enable_optimization:
            effective_batch_size = self.calculate_optimal_batch_size(points, batch_size)
        else:
            effective_batch_size = batch_size or self.default_batch_size
        
        stats.average_batch_size = effective_batch_size
        
        self.logger.info(
            f"Starting batch upsert for {len(points)} points to {collection_name} "
            f"(batch_size: {effective_batch_size})"
        )
        
        # Process points in batches
        for i in range(0, len(points), effective_batch_size):
            batch_points = points[i:i + effective_batch_size]
            batch_num = (i // effective_batch_size) + 1
            stats.batch_count += 1
            
            # Attempt batch insertion with retry
            success = self._insert_batch_with_retry(
                collection_name, 
                batch_points, 
                batch_num, 
                stats
            )
            
            if success:
                stats.successful_insertions += len(batch_points)
            else:
                # Try individual point insertion for failed batch
                individual_successes = self._retry_individual_points(
                    collection_name, 
                    batch_points, 
                    batch_num, 
                    stats
                )
                stats.successful_insertions += individual_successes
                stats.failed_insertions += (len(batch_points) - individual_successes)
            
            # Memory cleanup after each batch
            del batch_points
            gc.collect()
        
        stats.total_duration = time.time() - start_time
        
        # Log final statistics
        self.logger.info(
            f"Batch upsert complete: {stats.successful_insertions}/{stats.total_points} points inserted "
            f"({stats.success_rate:.1f}% success rate, {stats.insertions_per_second:.1f} points/sec)"
        )
        
        if stats.errors:
            self.logger.warning(f"Encountered {len(stats.errors)} errors during batch insertion")
        
        return stats
    
    def _insert_batch_with_retry(
        self, 
        collection_name: str, 
        batch_points: List[PointStruct], 
        batch_num: int, 
        stats: BatchInsertionStats
    ) -> bool:
        """
        Insert a single batch with retry logic.
        
        Returns:
            True if batch insertion succeeded, False otherwise
        """
        for attempt in range(self.max_retries + 1):
            try:
                batch_start = time.time()
                self.client.upsert(collection_name=collection_name, points=batch_points)
                batch_duration = time.time() - batch_start
                
                self.logger.debug(
                    f"Batch {batch_num} inserted successfully: {len(batch_points)} points "
                    f"(attempt {attempt + 1}, duration: {batch_duration:.2f}s)"
                )
                return True
                
            except Exception as e:
                error_msg = f"Batch {batch_num} attempt {attempt + 1} failed: {e}"
                
                if attempt < self.max_retries:
                    self.logger.warning(f"{error_msg}. Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    self.logger.error(f"{error_msg}. All retry attempts exhausted.")
                    stats.errors.append(error_msg)
        
        return False
    
    def _retry_individual_points(
        self, 
        collection_name: str, 
        batch_points: List[PointStruct], 
        batch_num: int, 
        stats: BatchInsertionStats
    ) -> int:
        """
        Retry failed batch by inserting individual points.
        
        Returns:
            Number of successfully inserted individual points
        """
        self.logger.warning(
            f"Retrying batch {batch_num} with individual point insertion "
            f"({len(batch_points)} points)"
        )
        
        successful_individual = 0
        
        for point_idx, point in enumerate(batch_points):
            try:
                self.client.upsert(collection_name=collection_name, points=[point])
                successful_individual += 1
                
            except Exception as e:
                error_msg = (
                    f"Individual point insertion failed "
                    f"(batch {batch_num}, point {point_idx + 1}): {e}"
                )
                self.logger.error(error_msg)
                stats.errors.append(error_msg)
        
        self.logger.info(
            f"Individual retry for batch {batch_num}: "
            f"{successful_individual}/{len(batch_points)} points succeeded"
        )
        
        return successful_individual
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a collection.
        
        Returns:
            Dictionary with collection statistics and configuration
        """
        try:
            collection_info = self.client.get_collection(collection_name)
            point_count = self.client.count(collection_name).count
            
            return {
                "name": collection_name,
                "points_count": point_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.value,
                "status": collection_info.status.value,
                "optimizer_status": collection_info.optimizer_status,
                "indexed_vectors_count": getattr(collection_info, 'indexed_vectors_count', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collection info for {collection_name}: {e}")
            return {"error": str(e), "collection_name": collection_name}