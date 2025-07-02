"""Error handling utilities for MCP tools.

This module provides comprehensive error handling, logging, and recovery utilities.
"""

import functools
import logging
import traceback
from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar

from tools.core.errors import MCPToolError, ServiceError, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T")


def log_and_raise_error(
    error_class: type[MCPToolError],
    message: str,
    details: str | None = None,
    logger_instance: logging.Logger | None = None,
    **kwargs,
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
    logger_instance: logging.Logger | None = None,
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
                raise ServiceError(message=error_msg, service_name=service_name, details=str(e))

        return wrapper

    return decorator(func)


def validate_required_fields(data: dict[str, Any], required_fields: list, context: str = "") -> None:
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
            details=f"Required fields: {required_fields}, Missing: {missing_fields}",
        )


def validate_field_types(data: dict[str, Any], field_types: dict[str, type], context: str = "") -> None:
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
            details=f"Type errors: {'; '.join(type_errors)}",
        )


def safe_execute(
    func: Callable[..., T],
    default_value: T = None,
    error_message: str = None,
    log_errors: bool = True,
    logger_instance: logging.Logger | None = None,
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


def create_error_context(operation: str, **context_data) -> dict[str, Any]:
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
        "traceback": traceback.format_stack()[-3:-1],  # Get relevant stack frames
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
    details = [f"Error Type: {type(error).__name__}", f"Message: {str(error)}"]

    # Add custom error attributes if it's an MCPToolError
    if isinstance(error, MCPToolError):
        if hasattr(error, "details") and error.details:
            details.append(f"Details: {error.details}")

        # Add specific attributes based on error type
        if hasattr(error, "file_path") and error.file_path:
            details.append(f"File: {error.file_path}")
        if hasattr(error, "service_name") and error.service_name:
            details.append(f"Service: {error.service_name}")
        if hasattr(error, "collection_name") and error.collection_name:
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


def handle_tool_error(func: Callable[..., T], *args, **kwargs) -> dict[str, Any]:
    """Handle tool execution with standardized error handling.

    Args:
        func: Function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Dictionary with function result or error information
    """
    try:
        return func(*args, **kwargs)
    except MCPToolError as e:
        # Handle known MCP tool errors
        error_info = {
            "error": str(e),
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat(),
        }

        if hasattr(e, "details") and e.details:
            error_info["details"] = e.details

        # Add specific error attributes
        for attr in [
            "file_path",
            "service_name",
            "collection_name",
            "query",
            "language",
        ]:
            if hasattr(e, attr):
                value = getattr(e, attr)
                if value:
                    error_info[attr] = value

        logger.error(f"MCP Tool Error in {func.__name__}: {format_error_details(e)}")
        return error_info

    except Exception as e:
        # Handle unexpected errors
        error_info = {
            "error": f"Unexpected error in {func.__name__}: {str(e)}",
            "error_type": type(e).__name__,
            "timestamp": datetime.now().isoformat(),
            "details": str(e),
        }

        logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return error_info


class log_tool_usage:
    """Context manager for logging tool usage and performance.

    This context manager logs when tools start and complete,
    tracks execution time, and handles any errors.
    """

    def __init__(
        self,
        tool_name: str,
        params: dict[str, Any] = None,
        logger_instance: logging.Logger | None = None,
    ):
        self.tool_name = tool_name
        self.params = params or {}
        self.logger = logger_instance or logger
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        param_str = ""
        if self.params:
            # Create a safe parameter string (avoid logging sensitive data)
            safe_params = {}
            for key, value in self.params.items():
                if isinstance(value, str | int | float | bool) and len(str(value)) < 100:
                    safe_params[key] = value
                else:
                    safe_params[key] = f"<{type(value).__name__}>"
            param_str = f" with params: {safe_params}"

        self.logger.info(f"Starting tool: {self.tool_name}{param_str}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        if exc_type is None:
            # Success
            self.logger.info(f"Completed tool: {self.tool_name} in {duration:.2f}s")
        else:
            # Error occurred
            self.logger.error(f"Tool failed: {self.tool_name} after {duration:.2f}s - {exc_type.__name__}: {exc_val}")

        # Don't suppress exceptions
        return False
