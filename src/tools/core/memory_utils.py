"""Memory management utilities for MCP tools.

This module provides memory monitoring and management functionality with
integration to cache memory leak detection.
"""

import gc
import logging
import os
from typing import Optional

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - memory monitoring disabled")

logger = logging.getLogger(__name__)

# Optional integration with cache memory leak detector
_leak_detector = None

# Configuration from environment
MEMORY_WARNING_THRESHOLD_MB = int(os.getenv("MEMORY_WARNING_THRESHOLD_MB", "1000"))
MEMORY_CLEANUP_INTERVAL = int(os.getenv("MEMORY_CLEANUP_INTERVAL", "5"))
FORCE_CLEANUP_THRESHOLD_MB = int(os.getenv("FORCE_CLEANUP_THRESHOLD_MB", "1500"))


def get_memory_stats() -> dict:
    """Get comprehensive memory statistics for both process and system.

    Returns:
        dict: Memory statistics including process and system memory info
    """
    if not PSUTIL_AVAILABLE:
        return {
            "process_memory_mb": 0.0,
            "rss_mb": 0.0,
            "vms_mb": 0.0,
            "system_memory": {
                "total_mb": 0.0,
                "available_mb": 0.0,
                "used_mb": 0.0,
                "percent_used": 0.0,
            },
            "psutil_available": False,
        }

    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()

        return {
            "process_memory_mb": memory_info.rss / (1024 * 1024),
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "system_memory": {
                "total_mb": virtual_memory.total / (1024 * 1024),
                "available_mb": virtual_memory.available / (1024 * 1024),
                "used_mb": virtual_memory.used / (1024 * 1024),
                "percent_used": virtual_memory.percent,
            },
            "psutil_available": True,
        }
    except Exception as e:
        logger.warning(f"Failed to get memory stats: {e}")
        return {
            "process_memory_mb": 0.0,
            "rss_mb": 0.0,
            "vms_mb": 0.0,
            "system_memory": {
                "total_mb": 0.0,
                "available_mb": 0.0,
                "used_mb": 0.0,
                "percent_used": 0.0,
            },
            "psutil_available": False,
            "error": str(e),
        }


def check_memory_usage(context: str = "") -> tuple[float, bool]:
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
            logger.warning(f"Memory usage ({memory_mb:.1f}MB) exceeds warning threshold ({MEMORY_WARNING_THRESHOLD_MB}MB)")

        if memory_mb > FORCE_CLEANUP_THRESHOLD_MB:
            return memory_mb, True

    return memory_mb, False


def force_memory_cleanup(context: str = "") -> None:
    """Force comprehensive memory cleanup.

    Args:
        context: Context string for logging
    """
    logger.info(f"Forcing memory cleanup{' for ' + context if context else ''}")

    # Force garbage collection multiple times for thoroughness
    for i in range(3):
        collected = gc.collect()
        if collected > 0:
            logger.debug(f"GC cycle {i + 1}: collected {collected} objects")

    # Clear any cached objects if possible
    if hasattr(gc, "set_threshold"):
        # Temporarily lower GC thresholds to be more aggressive
        original_thresholds = gc.get_threshold()
        gc.set_threshold(100, 10, 10)
        gc.collect()
        gc.set_threshold(*original_thresholds)

    memory_after = log_memory_usage("after cleanup")

    if memory_after > FORCE_CLEANUP_THRESHOLD_MB:
        logger.warning(
            f"Memory still high ({memory_after:.1f}MB) after cleanup. " f"Consider reducing batch sizes or processing fewer files."
        )


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
            if hasattr(var, "clear") and callable(var.clear):
                var.clear()
            elif hasattr(var, "close") and callable(var.close):
                var.close()
            del var
    gc.collect()


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB.

    Returns:
        float: Current memory usage in MB, or 0.0 if unavailable
    """
    if not PSUTIL_AVAILABLE:
        return 0.0
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert bytes to MB
    except Exception as e:
        logger.warning(f"Failed to get memory usage: {e}")
        return 0.0


def log_memory_usage(context: str = "") -> float:
    """Log current memory usage and return the value in MB.

    Args:
        context: Context string for logging

    Returns:
        float: Current memory usage in MB
    """
    memory_mb = get_memory_usage_mb()
    if memory_mb > 0:
        logger.info(f"Memory usage{' ' + context if context else ''}: {memory_mb:.1f}MB")
        if memory_mb > MEMORY_WARNING_THRESHOLD_MB:
            logger.warning(f"Memory usage ({memory_mb:.1f}MB) exceeds warning threshold ({MEMORY_WARNING_THRESHOLD_MB}MB)")

        # Integrate with leak detector if available
        _trigger_leak_detection_snapshot(context, memory_mb)

    return memory_mb


async def setup_leak_detector_integration():
    """Setup integration with cache memory leak detector."""
    global _leak_detector

    try:
        # Import here to avoid circular dependencies
        from src.services.cache_memory_leak_detector import get_leak_detector

        _leak_detector = await get_leak_detector()
        logger.info("Cache memory leak detector integration enabled")
    except Exception as e:
        logger.debug(f"Cache memory leak detector not available: {e}")
        _leak_detector = None


def _trigger_leak_detection_snapshot(context: str, memory_mb: float):
    """Trigger a memory snapshot for leak detection if detector is available."""
    global _leak_detector

    if _leak_detector is None:
        return

    try:
        # Extract cache name from context if possible
        cache_name = _extract_cache_name_from_context(context)
        if cache_name:
            # This would need to be called from an async context in real usage
            # For now, we just log that we would take a snapshot
            logger.debug(f"Would trigger leak detection snapshot for cache: {cache_name}")
    except Exception as e:
        logger.debug(f"Failed to trigger leak detection snapshot: {e}")


def _extract_cache_name_from_context(context: str) -> str | None:
    """Extract cache name from context string."""
    # Simple heuristic to extract cache name from context
    if "cache" in context.lower():
        # Look for patterns like "cache_name" or "for cache_name"
        parts = context.lower().split()
        for i, part in enumerate(parts):
            if "cache" in part:
                if i + 1 < len(parts):
                    return parts[i + 1].strip("_-.,:")
                return part
    return None


async def check_memory_leaks_for_cache(cache_name: str) -> dict:
    """Check for memory leaks in a specific cache.

    Args:
        cache_name: Name of the cache to check

    Returns:
        dict: Memory leak analysis results
    """
    global _leak_detector

    if _leak_detector is None:
        return {"status": "detector_not_available", "leaks": []}

    try:
        from src.services.cache_memory_leak_detector import analyze_cache_memory_leaks

        leaks = await analyze_cache_memory_leaks(cache_name)

        return {
            "status": "success",
            "cache_name": cache_name,
            "leak_count": len(leaks),
            "leaks": [
                {
                    "leak_id": leak.leak_id,
                    "leak_type": leak.leak_type.value,
                    "severity": leak.severity.value,
                    "memory_growth_mb": leak.memory_growth_mb,
                    "growth_rate_mb_per_minute": leak.growth_rate_mb_per_minute,
                    "detected_at": leak.detected_at.isoformat(),
                    "recommendations": leak.recommendations,
                }
                for leak in leaks
            ],
        }
    except Exception as e:
        logger.error(f"Failed to check memory leaks for cache {cache_name}: {e}")
        return {"status": "error", "error": str(e), "leaks": []}
