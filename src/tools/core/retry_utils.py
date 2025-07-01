"""Retry utilities for handling transient failures.

This module provides retry logic for database and network operations.
"""

import time
import logging
from typing import Callable, Any, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


def retry_operation(
    func: Callable[..., T], 
    max_attempts: int = 3, 
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    operation_name: Optional[str] = None
) -> T:
    """Retry an operation with exponential backoff.
    
    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts in seconds
        backoff_factor: Factor to multiply delay by after each attempt
        operation_name: Name of operation for logging
        
    Returns:
        Result of the function call
        
    Raises:
        Exception: The last exception if all attempts fail
    """
    name = operation_name or func.__name__
    current_delay = delay
    
    for attempt in range(max_attempts):
        try:
            result = func()
            if attempt > 0:
                logger.info(f"{name} succeeded after {attempt + 1} attempts")
            return result
        except Exception as e:
            if attempt == max_attempts - 1:
                logger.error(f"{name} failed after {max_attempts} attempts: {e}")
                raise
            else:
                logger.warning(
                    f"{name} attempt {attempt + 1}/{max_attempts} failed: {e}. "
                    f"Retrying in {current_delay:.1f}s..."
                )
                time.sleep(current_delay)
                current_delay *= backoff_factor