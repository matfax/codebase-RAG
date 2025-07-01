"""Memory management utilities for MCP tools.

This module provides memory monitoring and management functionality.
"""

import os
import gc
import logging
from typing import Optional, Tuple

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - memory monitoring disabled")

logger = logging.getLogger(__name__)

# Configuration from environment
MEMORY_WARNING_THRESHOLD_MB = int(os.getenv("MEMORY_WARNING_THRESHOLD_MB", "1000"))
MEMORY_CLEANUP_INTERVAL = int(os.getenv("MEMORY_CLEANUP_INTERVAL", "5"))
FORCE_CLEANUP_THRESHOLD_MB = int(os.getenv("FORCE_CLEANUP_THRESHOLD_MB", "1500"))


def get_memory_stats() -> dict:
    """Get current memory statistics.
    
    Returns:
        dict: Memory statistics including RSS, VMS, and available memory
    """
    if not PSUTIL_AVAILABLE:
        return {
            "rss_mb": 0.0,
            "vms_mb": 0.0,
            "available_mb": 0.0,
            "percent": 0.0,
            "psutil_available": False
        }
    
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "available_mb": virtual_memory.available / (1024 * 1024),
            "percent": virtual_memory.percent,
            "psutil_available": True
        }
    except Exception as e:
        logger.warning(f"Failed to get memory stats: {e}")
        return {
            "rss_mb": 0.0,
            "vms_mb": 0.0,
            "available_mb": 0.0,
            "percent": 0.0,
            "psutil_available": False,
            "error": str(e)
        }


def check_memory_usage(context: str = "") -> Tuple[float, bool]:
    """Check current memory usage and log warnings if needed.
    
    Args:
        context: Context string for logging
        
    Returns:
        Tuple[float, bool]: (memory_mb, needs_cleanup)
    """
    stats = get_memory_stats()
    memory_mb = stats["rss_mb"]
    
    if memory_mb > 0:
        logger.info(f"Memory usage{' ' + context if context else ''}: {memory_mb:.1f}MB")
        
        if memory_mb > MEMORY_WARNING_THRESHOLD_MB:
            logger.warning(
                f"Memory usage ({memory_mb:.1f}MB) exceeds warning threshold ({MEMORY_WARNING_THRESHOLD_MB}MB)"
            )
        
        if memory_mb > FORCE_CLEANUP_THRESHOLD_MB:
            return memory_mb, True
    
    return memory_mb, False


def force_memory_cleanup(context: str = "") -> None:
    """Force garbage collection and log memory usage.
    
    Args:
        context: Context string for logging
    """
    logger.info(f"Forcing memory cleanup{' ' + context if context else ''}...")
    
    before_mb, _ = check_memory_usage("before cleanup")
    
    # Force garbage collection
    gc.collect()
    
    after_mb, _ = check_memory_usage("after cleanup")
    
    if before_mb > 0 and after_mb > 0:
        freed_mb = before_mb - after_mb
        logger.info(f"Memory cleanup freed {freed_mb:.1f}MB")


def should_cleanup_memory(batch_count: int, force_check: bool = False) -> bool:
    """Determine if memory cleanup should be performed.
    
    Args:
        batch_count: Number of batches processed
        force_check: Force memory check regardless of batch count
        
    Returns:
        bool: True if cleanup should be performed
    """
    if force_check or (batch_count > 0 and batch_count % MEMORY_CLEANUP_INTERVAL == 0):
        _, needs_cleanup = check_memory_usage()
        return needs_cleanup
    return False


def get_adaptive_batch_size(base_batch_size: int, memory_usage_mb: float) -> int:
    """Calculate adaptive batch size based on memory usage.
    
    Args:
        base_batch_size: Base batch size
        memory_usage_mb: Current memory usage in MB
        
    Returns:
        int: Adjusted batch size
    """
    if memory_usage_mb > FORCE_CLEANUP_THRESHOLD_MB:
        # Critical memory usage - use minimum batch size
        return max(1, base_batch_size // 4)
    elif memory_usage_mb > MEMORY_WARNING_THRESHOLD_MB:
        # High memory usage - reduce batch size
        return max(1, base_batch_size // 2)
    else:
        # Normal memory usage
        return base_batch_size


def clear_processing_variables(*variables) -> None:
    """Clear processing variables to free memory.
    
    Args:
        *variables: Variables to clear
    """
    for var in variables:
        if var is not None:
            if hasattr(var, 'clear'):
                var.clear()
            elif hasattr(var, 'close'):
                var.close()
    gc.collect()