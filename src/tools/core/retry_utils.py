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


def retry_with_context(
    func: Callable[..., T],
    operation_name: str,
    max_retries: int = None,
    retry_delay: float = None,
    backoff_factor: float = 2.0
) -> T:
    """Retry operation with environment-based configuration.
    
    This is a more sophisticated retry function that uses environment variables
    for configuration and provides enhanced logging.
    
    Args:
        func: Function to retry
        operation_name: Name of operation for detailed logging
        max_retries: Maximum retry attempts (uses DB_RETRY_ATTEMPTS from env if None)
        retry_delay: Initial delay between retries (uses DB_RETRY_DELAY from env if None)
        backoff_factor: Exponential backoff multiplier
        
    Returns:
        Result of the function call
        
    Raises:
        Exception: The last exception if all attempts fail
    """
    import os
    
    max_retries = max_retries or int(os.getenv("DB_RETRY_ATTEMPTS", "3"))
    retry_delay = retry_delay or float(os.getenv("DB_RETRY_DELAY", "1.0"))
    
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            result = func()
            if attempt > 0:
                logger.info(f"{operation_name} succeeded after {attempt + 1} attempts")
            return result
        except Exception as e:
            last_error = e
            if attempt == max_retries:
                logger.error(f"{operation_name} failed after {max_retries + 1} attempts: {e}")
                break
            
            delay = retry_delay * (backoff_factor ** attempt)  # Exponential backoff
            logger.warning(f"{operation_name} attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
            time.sleep(delay)
    
    raise last_error if last_error else Exception("Unknown error during retry")