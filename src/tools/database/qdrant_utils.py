"""Qdrant database utilities for MCP tools.

This module provides Qdrant-specific database operations, health checks, and retry logic.
"""

import os
import time
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime

from qdrant_client.http.models import PointStruct

logger = logging.getLogger(__name__)


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
            logger.error(f"  ... and {len(stats["errors"]) - 3} more errors")