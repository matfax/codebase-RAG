"""Qdrant database utilities for MCP tools.

This module provides Qdrant-specific database operations, health checks, and retry logic.
"""

import os
import time
import hashlib
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct, Distance, VectorParams, Filter, FieldCondition, MatchValue
)

from tools.core.errors import QdrantConnectionError

logger = logging.getLogger(__name__)

# Global client instance for lazy initialization
_qdrant_client = None


def check_qdrant_health(client) -> Dict[str, Any]:
    """Check Qdrant connection health and return status information.
    
    Args:
        client: Qdrant client instance
        
    Returns:
        Dict containing health status, response time, and collection count
    """
    try:
        start_time = time.time()
        collections = client.get_collections()
        response_time = time.time() - start_time
        
        return {
            "healthy": True,
            "response_time_ms": response_time * 1000,
            "collections_count": len(collections.collections),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def retry_qdrant_operation(operation_func, operation_name: str, max_retries: int = None) -> Any:
    """Retry Qdrant operations with exponential backoff.
    
    Args:
        operation_func: Function to retry
        operation_name: Name of operation for logging
        max_retries: Maximum number of retry attempts (from env if not specified)
        
    Returns:
        Result of the operation function
        
    Raises:
        Exception: The last exception if all retries fail
    """
    max_retries = max_retries or int(os.getenv("DB_RETRY_ATTEMPTS", "3"))
    retry_delay = float(os.getenv("DB_RETRY_DELAY", "1.0"))
    
    for attempt in range(max_retries + 1):
        try:
            return operation_func()
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"{operation_name} failed after {max_retries + 1} attempts: {e}")
                raise e
            
            delay = retry_delay * (2 ** attempt)  # Exponential backoff
            logger.warning(f"{operation_name} attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
            time.sleep(delay)


def retry_individual_points(client, collection_name: str, points: List[PointStruct]) -> Tuple[int, int]:
    """Retry individual points when batch insertion fails.
    
    This function attempts to insert points one-by-one when a batch operation fails,
    allowing partial success rather than complete failure.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        points: List of points to retry individually
        
    Returns:
        Tuple of (successful_count, failed_count)
    """
    successful_count = 0
    failed_count = 0
    
    for point in points:
        try:
            client.upsert(collection_name=collection_name, points=[point])
            successful_count += 1
        except Exception as e:
            failed_count += 1
            logger.debug(f"Individual point insertion failed for {point.id}: {e}")
            # Don't retry individual points to avoid infinite loops
    
    return successful_count, failed_count


def log_database_metrics(stats: Dict[str, Any], operation_time: float, operation_type: str) -> None:
    """Log detailed database operation metrics.
    
    Args:
        stats: Dictionary containing operation statistics
        operation_time: Total operation time in seconds
        operation_type: Type of operation (e.g., "insertion", "search")
    """
    points_per_second = stats["successful_insertions"] / operation_time if operation_time > 0 else 0
    success_rate = (stats["successful_insertions"] / stats["total_points"]) * 100 if stats["total_points"] > 0 else 0
    
    logger.info(
        f"Database {operation_type} metrics: "
        f"{stats['successful_insertions']}/{stats['total_points']} points "
        f"({success_rate:.1f}% success), "
        f"{points_per_second:.1f} points/sec, "
        f"{stats['batch_count']} batches, "
        f"total time: {operation_time:.2f}s"
    )
    
    if stats["failed_insertions"] > 0:
        logger.warning(f"Database operation had {stats['failed_insertions']} failed insertions")
    
    if stats["errors"]:
        logger.error(f"Database operation errors: {len(stats['errors'])} error(s)")
        for i, error in enumerate(stats["errors"][:3]):  # Log first 3 errors
            logger.error(f"  Error {i+1}: {error}")
        if len(stats["errors"]) > 3:
            logger.error(f"  ... and {len(stats['errors']) - 3} more errors")


def get_qdrant_client() -> QdrantClient:
    """Get or create Qdrant client instance with connection validation.
    
    Returns:
        QdrantClient instance
        
    Raises:
        QdrantConnectionError: If connection fails
    """
    global _qdrant_client
    if _qdrant_client is None:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        try:
            _qdrant_client = QdrantClient(host=host, port=port)
            # Test connection
            _qdrant_client.get_collections()
        except Exception as e:
            raise QdrantConnectionError(
                f"Failed to connect to Qdrant at {host}:{port}", details=str(e)
            )
    return _qdrant_client


def ensure_collection(collection_name: str, embedding_dimension: Optional[int] = None, 
                     embedding_model_name: Optional[str] = None, content_type: str = "general") -> None:
    """Ensure a collection exists, creating it if necessary.
    
    Args:
        collection_name: Name of the collection to ensure exists
        embedding_dimension: Dimension of embedding vectors (default: 768)
        embedding_model_name: Name of embedding model used
        content_type: Type of content stored in collection
    """
    client = get_qdrant_client()
    
    def check_and_create():
        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            # Use provided values or defaults
            final_dimension = embedding_dimension or 768  # Default for nomic-embed-text
            final_model_name = embedding_model_name or os.getenv("OLLAMA_DEFAULT_EMBEDDING_MODEL", "nomic-embed-text")

            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=final_dimension,
                    distance=Distance.COSINE
                )
            )
            
            # Store metadata about the collection
            metadata_payload = {
                "type": "collection_metadata",
                "embedding_model": final_model_name,
                "embedding_dimension": final_dimension,
                "content_type": content_type,
                "created_at": datetime.now().isoformat(),
            }
            metadata_point_id = hashlib.md5(f"{collection_name}_metadata".encode()).hexdigest()
            metadata_point = PointStruct(
                id=metadata_point_id,
                vector=[0.0] * final_dimension,
                payload=metadata_payload
            )
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=[metadata_point]
                )
                logger.info(f"Created collection {collection_name} with model {final_model_name}")
            except Exception as e:
                logger.warning(f"Failed to store metadata for collection {collection_name}: {e}")
    
    retry_qdrant_operation(check_and_create, f"ensure collection {collection_name}")


def check_existing_index(project_info: Dict[str, str]) -> Dict[str, Any]:
    """Check if the project already has indexed data.
    
    Args:
        project_info: Project information from get_current_project()
        
    Returns:
        Dictionary with existing index information
    """
    try:
        client = get_qdrant_client()
        existing_collections = [c.name for c in client.get_collections().collections]
        
        project_collections = []
        total_points = 0
        collection_info = []
        
        if project_info:
            prefix = project_info['collection_prefix']
            for collection_type in ['code', 'config', 'documentation', 'file_metadata']:
                collection_name = f"{prefix}_{collection_type}"
                if collection_name in existing_collections:
                    try:
                        # Get collection info
                        collection_info_response = client.get_collection(collection_name)
                        points_count = collection_info_response.points_count
                        
                        project_collections.append(collection_name)
                        total_points += points_count
                        collection_info.append({
                            "name": collection_name,
                            "type": collection_type,
                            "points_count": points_count
                        })
                    except Exception as e:
                        logger.warning(f"Could not get info for collection {collection_name}: {e}")
        
        return {
            "has_existing_data": len(project_collections) > 0,
            "collections": project_collections,
            "total_points": total_points,
            "collection_details": collection_info,
            "project_name": project_info.get('name', 'unknown') if project_info else 'unknown'
        }
        
    except Exception as e:
        logger.error(f"Error checking existing index: {e}")
        return {
            "has_existing_data": False,
            "error": str(e)
        }


def estimate_indexing_time(file_count: int, existing_points: int = 0) -> Dict[str, Any]:
    """Estimate indexing time and provide recommendations.
    
    Args:
        file_count: Number of files to be indexed
        existing_points: Number of existing points in collections
        
    Returns:
        Dictionary with time estimates and recommendations
    """
    # Rough estimates based on typical performance
    seconds_per_file = 0.5  # Including file reading, embedding, and DB insertion
    estimated_seconds = file_count * seconds_per_file
    estimated_minutes = estimated_seconds / 60
    
    if existing_points > 0:
        # Estimate time saved by not re-indexing
        avg_points_per_file = existing_points / max(file_count, 1)
        time_saved_seconds = existing_points / avg_points_per_file * seconds_per_file if avg_points_per_file > 0 else 0
        time_saved_minutes = time_saved_seconds / 60
    else:
        time_saved_minutes = 0
    
    # Smart recommendation logic
    if existing_points > 0 and estimated_minutes > 5:
        recommendation = "keep_existing_recommended"
    elif estimated_minutes > 30:
        recommendation = "large_operation_confirm"
    elif estimated_minutes > 10:
        recommendation = "medium_operation"
    else:
        recommendation = "quick_operation"
    
    return {
        "estimated_time_minutes": round(estimated_minutes, 1),
        "time_saved_by_keeping_existing_minutes": round(time_saved_minutes, 1),
        "recommendation": recommendation,
        "file_count": file_count,
        "existing_points": existing_points
    }