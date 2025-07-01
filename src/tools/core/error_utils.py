"""Error handling utilities for MCP tools.

This module provides comprehensive error handling, logging, and recovery utilities.
"""

import logging
import traceback
import functools
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union
from datetime import datetime

from .errors import MCPToolError, ServiceError, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar('T')


def log_and_raise_error(
    error_class: Type[MCPToolError],
    message: str,
    details: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None,
    **kwargs
) -> None:
    """Log an error and raise the appropriate exception.
    
    Args:
        error_class: The error class to raise
        message: Primary error message
        details: Additional error details
        logger_instance: Logger to use (defaults to module logger)
        **kwargs: Additional arguments for the error class constructor
    """
    log = logger_instance or logger
    
    # Create full error message for logging
    full_message = message
    if details:
        full_message += f" Details: {details}"
    
    # Log the error
    log.error(full_message)
    
    # Raise the appropriate exception
    if details:
        raise error_class(message, details=details, **kwargs)
    else:
        raise error_class(message, **kwargs)


def handle_service_error(
    func: Callable[..., T],
    service_name: str,
    operation_name: str = None,
    logger_instance: Optional[logging.Logger] = None
) -> Callable[..., T]:
    """Decorator to handle service errors with standardized logging.
    
    Args:
        func: Function to wrap
        service_name: Name of the service
        operation_name: Name of the operation (defaults to function name)
        logger_instance: Logger to use
    
    Returns:
        Wrapped function with error handling
    """
    def decorator(wrapper_func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(wrapper_func)
        def wrapper(*args, **kwargs) -> T:
            op_name = operation_name or wrapper_func.__name__
            log = logger_instance or logger
            
            try:
                log.debug(f"Starting {service_name}.{op_name}")
                result = wrapper_func(*args, **kwargs)
                log.debug(f"Completed {service_name}.{op_name}")
                return result
            except MCPToolError:
                # Re-raise MCP tool errors as-is
                raise
            except Exception as e:
                error_msg = f"{service_name}.{op_name} failed: {str(e)}"
                log.error(error_msg)
                log.error(f"Traceback: {traceback.format_exc()}")
                raise ServiceError(
                    message=error_msg,
                    service_name=service_name,
                    details=str(e)
                )
        
        return wrapper
    
    return decorator(func)


def validate_required_fields(data: Dict[str, Any], required_fields: list, context: str = "") -> None:
    """Validate that required fields are present and not None.
    
    Args:
        data: Dictionary to validate
        required_fields: List of required field names
        context: Context for error messages
        
    Raises:
        ValidationError: If any required fields are missing or None
    """
    missing_fields = []
    
    for field in required_fields:
        if field not in data or data[field] is None:
            missing_fields.append(field)
    
    if missing_fields:
        context_str = f" in {context}" if context else ""
        raise ValidationError(
            message=f"Missing required fields{context_str}: {', '.join(missing_fields)}",
            field_name=missing_fields[0] if len(missing_fields) == 1 else None,
            details=f"Required fields: {required_fields}, Missing: {missing_fields}"
        )


def validate_field_types(data: Dict[str, Any], field_types: Dict[str, Type], context: str = "") -> None:
    """Validate that fields have the correct types.
    
    Args:
        data: Dictionary to validate
        field_types: Dictionary mapping field names to expected types
        context: Context for error messages
        
    Raises:
        ValidationError: If any fields have incorrect types
    """
    type_errors = []
    
    for field, expected_type in field_types.items():
        if field in data and data[field] is not None:
            if not isinstance(data[field], expected_type):
                actual_type = type(data[field]).__name__
                expected_type_name = expected_type.__name__
                type_errors.append(f"{field}: expected {expected_type_name}, got {actual_type}")
    
    if type_errors:
        context_str = f" in {context}" if context else ""
        raise ValidationError(
            message=f"Type validation failed{context_str}",
            details=f"Type errors: {'; '.join(type_errors)}"
        )


def safe_execute(
    func: Callable[..., T],
    default_value: T = None,
    error_message: str = None,
    log_errors: bool = True,
    logger_instance: Optional[logging.Logger] = None
) -> T:
    """Execute a function safely, returning a default value on error.
    
    Args:
        func: Function to execute
        default_value: Value to return on error
        error_message: Custom error message prefix
        log_errors: Whether to log errors
        logger_instance: Logger to use
        
    Returns:
        Function result or default value on error
    """
    log = logger_instance or logger
    
    try:
        return func()
    except Exception as e:
        if log_errors:
            prefix = error_message or f"Error in {func.__name__}"
            log.error(f"{prefix}: {str(e)}")
            log.debug(f"Traceback: {traceback.format_exc()}")
        return default_value


def create_error_context(
    operation: str,
    **context_data
) -> Dict[str, Any]:
    """Create standardized error context information.
    
    Args:
        operation: Name of the operation
        **context_data: Additional context data
        
    Returns:
        Dictionary with error context
    """
    context = {
        "operation": operation,
        "timestamp": datetime.now().isoformat(),
        "traceback": traceback.format_stack()[-3:-1]  # Get relevant stack frames
    }
    
    # Add any additional context data
    context.update(context_data)
    
    return context


def format_error_details(error: Exception, include_traceback: bool = False) -> str:
    """Format error details for logging or user display.
    
    Args:
        error: Exception to format
        include_traceback: Whether to include full traceback
        
    Returns:
        Formatted error string
    """
    details = [
        f"Error Type: {type(error).__name__}",
        f"Message: {str(error)}"
    ]
    
    # Add custom error attributes if it's an MCPToolError
    if isinstance(error, MCPToolError):
        if hasattr(error, 'details') and error.details:
            details.append(f"Details: {error.details}")
        
        # Add specific attributes based on error type
        if hasattr(error, 'file_path') and error.file_path:
            details.append(f"File: {error.file_path}")
        if hasattr(error, 'service_name') and error.service_name:
            details.append(f"Service: {error.service_name}")
        if hasattr(error, 'collection_name') and error.collection_name:
            details.append(f"Collection: {error.collection_name}")
    
    if include_traceback:
        details.append(f"Traceback: {traceback.format_exc()}")
    
    return " | ".join(details)


def chain_exceptions(new_error: Exception, original_error: Exception) -> Exception:
    """Chain exceptions to preserve the original error context.
    
    Args:
        new_error: New exception to raise
        original_error: Original exception that caused the problem
        
    Returns:
        New exception with chained context
    """
    new_error.__cause__ = original_error
    return new_error