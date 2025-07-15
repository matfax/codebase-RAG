"""
Prompt Error Handling and Logging System

This module provides comprehensive error handling, logging, and recovery
mechanisms for MCP prompt operations.
"""

import logging
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from src.models.prompt_context import PromptContext, PromptType


class ErrorSeverity(Enum):
    """Severity levels for prompt errors."""

    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ErrorCategory(Enum):
    """Categories of errors that can occur in prompts."""

    VALIDATION_ERROR = "validation_error"
    PARAMETER_ERROR = "parameter_error"
    SERVICE_ERROR = "service_error"
    SEARCH_ERROR = "search_error"
    ANALYSIS_ERROR = "analysis_error"
    WORKFLOW_ERROR = "workflow_error"
    SYSTEM_ERROR = "system_error"
    TIMEOUT_ERROR = "timeout_error"
    PERMISSION_ERROR = "permission_error"
    RESOURCE_ERROR = "resource_error"


@dataclass
class PromptError:
    """
    Structured error information for prompt operations.

    Captures detailed error context for debugging and user feedback.
    """

    # Error identification
    error_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Error classification
    severity: ErrorSeverity = ErrorSeverity.ERROR
    category: ErrorCategory = ErrorCategory.SYSTEM_ERROR
    error_code: str | None = None

    # Error details
    message: str = ""
    details: str = ""
    exception_type: str | None = None
    stack_trace: str | None = None

    # Context information
    prompt_type: PromptType | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    context: PromptContext | None = None

    # Recovery information
    recoverable: bool = True
    suggested_fixes: list[str] = field(default_factory=list)
    retry_recommended: bool = False

    # User communication
    user_message: str = ""
    technical_message: str = ""

    # Metadata
    session_id: str | None = None
    workflow_id: str | None = None
    correlation_id: str | None = None


@dataclass
class ErrorStats:
    """Statistics about prompt errors for monitoring and improvement."""

    total_errors: int = 0
    errors_by_category: dict[str, int] = field(default_factory=dict)
    errors_by_prompt_type: dict[str, int] = field(default_factory=dict)
    errors_by_severity: dict[str, int] = field(default_factory=dict)

    # Time-based tracking
    errors_last_hour: int = 0
    errors_last_day: int = 0
    error_rate_per_minute: float = 0.0

    # Recovery statistics
    recovered_errors: int = 0
    unrecoverable_errors: int = 0
    retry_success_rate: float = 0.0

    # Recent errors for analysis
    recent_errors: list[PromptError] = field(default_factory=list)

    # Performance impact
    avg_error_handling_time_ms: float = 0.0
    total_error_handling_time_ms: float = 0.0


class PromptErrorHandler:
    """
    Comprehensive error handling system for MCP prompts.

    Provides error classification, logging, recovery suggestions,
    and statistics tracking for prompt operations.
    """

    def __init__(self, logger_name: str = "mcp_prompts"):
        self.logger = logging.getLogger(logger_name)
        self.stats = ErrorStats()
        self.error_history: list[PromptError] = []
        self.recovery_strategies: dict[ErrorCategory, Callable] = {}

        # Configure error handling
        self._setup_recovery_strategies()
        self._setup_error_logging()

    def handle_error(
        self,
        exception: Exception,
        prompt_type: PromptType | None = None,
        parameters: dict[str, Any] | None = None,
        context: PromptContext | None = None,
        correlation_id: str | None = None,
    ) -> PromptError:
        """
        Handle and classify a prompt error.

        Args:
            exception: The exception that occurred
            prompt_type: Type of prompt that failed
            parameters: Parameters used when error occurred
            context: Prompt context when error occurred
            correlation_id: Correlation ID for tracking

        Returns:
            PromptError: Structured error information
        """
        start_time = time.time()

        # Create error object
        error = PromptError(
            error_id=f"error_{int(time.time() * 1000)}",
            prompt_type=prompt_type,
            parameters=parameters or {},
            context=context,
            correlation_id=correlation_id,
            session_id=context.session_id if context else None,
            exception_type=type(exception).__name__,
            stack_trace=traceback.format_exc(),
        )

        # Classify the error
        self._classify_error(error, exception)

        # Generate user-friendly messages
        self._generate_error_messages(error, exception)

        # Determine recovery options
        self._analyze_recovery_options(error, exception)

        # Log the error
        self._log_error(error)

        # Update statistics
        self._update_error_stats(error)

        # Store in history
        self.error_history.append(error)
        self._trim_error_history()

        # Track error handling performance
        handling_time = (time.time() - start_time) * 1000
        self.stats.total_error_handling_time_ms += handling_time
        error_count = self.stats.total_errors
        if error_count > 0:
            self.stats.avg_error_handling_time_ms = self.stats.total_error_handling_time_ms / error_count

        return error

    def attempt_recovery(self, error: PromptError, **recovery_params) -> bool:
        """
        Attempt to recover from an error using appropriate strategy.

        Args:
            error: The error to recover from
            **recovery_params: Additional parameters for recovery

        Returns:
            True if recovery was successful, False otherwise
        """
        if not error.recoverable:
            self.logger.info(f"Error {error.error_id} marked as unrecoverable")
            return False

        recovery_strategy = self.recovery_strategies.get(error.category)
        if not recovery_strategy:
            self.logger.warning(f"No recovery strategy for error category {error.category}")
            return False

        try:
            self.logger.info(f"Attempting recovery for error {error.error_id} using {error.category} strategy")
            success = recovery_strategy(error, **recovery_params)

            if success:
                self.stats.recovered_errors += 1
                self.logger.info(f"Successfully recovered from error {error.error_id}")
            else:
                self.stats.unrecoverable_errors += 1
                self.logger.warning(f"Recovery failed for error {error.error_id}")

            # Update retry success rate
            total_recoveries = self.stats.recovered_errors + self.stats.unrecoverable_errors
            if total_recoveries > 0:
                self.stats.retry_success_rate = self.stats.recovered_errors / total_recoveries

            return success

        except Exception as recovery_exception:
            self.logger.error(f"Recovery attempt failed with exception: {recovery_exception}")
            self.stats.unrecoverable_errors += 1
            return False

    def get_error_context(self, error_id: str) -> PromptError | None:
        """Get full context for a specific error."""
        for error in self.error_history:
            if error.error_id == error_id:
                return error
        return None

    def get_error_statistics(self) -> dict[str, Any]:
        """Get comprehensive error statistics."""
        return {
            "total_errors": self.stats.total_errors,
            "error_rate_per_minute": self.stats.error_rate_per_minute,
            "errors_by_category": dict(self.stats.errors_by_category),
            "errors_by_prompt_type": dict(self.stats.errors_by_prompt_type),
            "errors_by_severity": dict(self.stats.errors_by_severity),
            "recovery_stats": {
                "recovered_errors": self.stats.recovered_errors,
                "unrecoverable_errors": self.stats.unrecoverable_errors,
                "success_rate": round(self.stats.retry_success_rate * 100, 1),
            },
            "performance": {
                "avg_handling_time_ms": round(self.stats.avg_error_handling_time_ms, 2),
                "total_handling_time_ms": round(self.stats.total_error_handling_time_ms, 2),
            },
            "recent_error_types": [error.category.value for error in self.stats.recent_errors[-10:]],
        }

    def get_user_friendly_error(self, error: PromptError) -> dict[str, Any]:
        """Get user-friendly error information."""
        return {
            "error_id": error.error_id,
            "message": error.user_message or error.message,
            "severity": error.severity.value,
            "recoverable": error.recoverable,
            "suggested_fixes": error.suggested_fixes,
            "retry_recommended": error.retry_recommended,
            "timestamp": error.timestamp.isoformat(),
        }

    @contextmanager
    def error_context(
        self,
        prompt_type: PromptType | None = None,
        parameters: dict[str, Any] | None = None,
        context: PromptContext | None = None,
        correlation_id: str | None = None,
    ):
        """
        Context manager for automatic error handling.

        Usage:
            with error_handler.error_context(prompt_type=PromptType.EXPLORE_PROJECT):
                # Your prompt code here
                pass
        """
        try:
            yield
        except Exception as e:
            error = self.handle_error(e, prompt_type, parameters, context, correlation_id)
            raise PromptOperationError(error) from e

    def _classify_error(self, error: PromptError, exception: Exception):
        """Classify error based on exception type and context."""
        exception_name = type(exception).__name__
        exception_message = str(exception).lower()

        # Classify by exception type
        if "validation" in exception_message or "invalid" in exception_message:
            error.category = ErrorCategory.VALIDATION_ERROR
            error.severity = ErrorSeverity.WARNING
        elif "parameter" in exception_message or "argument" in exception_message:
            error.category = ErrorCategory.PARAMETER_ERROR
            error.severity = ErrorSeverity.WARNING
        elif "timeout" in exception_message or "timed out" in exception_message:
            error.category = ErrorCategory.TIMEOUT_ERROR
            error.severity = ErrorSeverity.ERROR
        elif "permission" in exception_message or "access denied" in exception_message:
            error.category = ErrorCategory.PERMISSION_ERROR
            error.severity = ErrorSeverity.ERROR
        elif "search" in exception_message or "query" in exception_message:
            error.category = ErrorCategory.SEARCH_ERROR
            error.severity = ErrorSeverity.WARNING
        elif "service" in exception_message or "connection" in exception_message:
            error.category = ErrorCategory.SERVICE_ERROR
            error.severity = ErrorSeverity.ERROR
        elif "memory" in exception_message or "resource" in exception_message:
            error.category = ErrorCategory.RESOURCE_ERROR
            error.severity = ErrorSeverity.ERROR
        else:
            error.category = ErrorCategory.SYSTEM_ERROR
            error.severity = ErrorSeverity.ERROR

        # Set error code
        error.error_code = f"{error.category.value.upper()}_{error.error_id[-6:]}"

        # Set basic details
        error.message = str(exception)
        error.details = f"{exception_name}: {exception}"

    def _generate_error_messages(self, error: PromptError, exception: Exception):
        """Generate user-friendly and technical error messages."""
        category_messages = {
            ErrorCategory.VALIDATION_ERROR: "Invalid input parameters provided",
            ErrorCategory.PARAMETER_ERROR: "Incorrect parameters specified",
            ErrorCategory.SERVICE_ERROR: "External service is unavailable",
            ErrorCategory.SEARCH_ERROR: "Search operation failed",
            ErrorCategory.ANALYSIS_ERROR: "Analysis operation failed",
            ErrorCategory.TIMEOUT_ERROR: "Operation timed out",
            ErrorCategory.PERMISSION_ERROR: "Permission denied for requested operation",
            ErrorCategory.RESOURCE_ERROR: "Insufficient system resources",
            ErrorCategory.SYSTEM_ERROR: "Internal system error occurred",
        }

        error.user_message = category_messages.get(error.category, "An error occurred")
        error.technical_message = f"{error.category.value}: {error.message}"

    def _analyze_recovery_options(self, error: PromptError, exception: Exception):
        """Analyze and suggest recovery options."""
        recovery_suggestions = {
            ErrorCategory.VALIDATION_ERROR: [
                "Check parameter values and formats",
                "Ensure all required parameters are provided",
                "Verify parameter types match expected formats",
            ],
            ErrorCategory.PARAMETER_ERROR: [
                "Review parameter documentation",
                "Check for typos in parameter names",
                "Ensure parameter values are within valid ranges",
            ],
            ErrorCategory.SERVICE_ERROR: [
                "Check if external services are running",
                "Verify network connectivity",
                "Wait a moment and retry the operation",
            ],
            ErrorCategory.SEARCH_ERROR: [
                "Try simpler search terms",
                "Check if the project has been indexed",
                "Verify search parameters are correct",
            ],
            ErrorCategory.TIMEOUT_ERROR: [
                "Try with a smaller dataset or scope",
                "Increase timeout settings if configurable",
                "Break the operation into smaller steps",
            ],
            ErrorCategory.PERMISSION_ERROR: [
                "Check file and directory permissions",
                "Ensure the path exists and is accessible",
                "Run with appropriate user permissions",
            ],
            ErrorCategory.RESOURCE_ERROR: [
                "Free up system memory",
                "Reduce operation scope or batch size",
                "Wait for system resources to become available",
            ],
        }

        error.suggested_fixes = recovery_suggestions.get(
            error.category,
            [
                "Review the error details",
                "Check the prompt parameters",
                "Try the operation again",
            ],
        )

        # Determine if retry is recommended
        retry_categories = {
            ErrorCategory.SERVICE_ERROR,
            ErrorCategory.TIMEOUT_ERROR,
            ErrorCategory.RESOURCE_ERROR,
        }
        error.retry_recommended = error.category in retry_categories

        # Determine if error is recoverable
        unrecoverable_categories = {ErrorCategory.PERMISSION_ERROR}
        error.recoverable = error.category not in unrecoverable_categories

    def _log_error(self, error: PromptError):
        """Log error with appropriate level and detail."""
        log_message = f"Prompt error {error.error_id}: {error.message}"

        if error.prompt_type:
            log_message += f" (prompt: {error.prompt_type.value})"

        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
            self.logger.critical(f"Error details: {error.details}")
        elif error.severity == ErrorSeverity.ERROR:
            self.logger.error(log_message)
            self.logger.debug(f"Error details: {error.details}")
        elif error.severity == ErrorSeverity.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

        # Log stack trace for debugging if severity is error or above
        if error.severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL] and error.stack_trace:
            self.logger.debug(f"Stack trace for {error.error_id}:\n{error.stack_trace}")

    def _update_error_stats(self, error: PromptError):
        """Update error statistics."""
        self.stats.total_errors += 1

        # Update category counts
        category_key = error.category.value
        self.stats.errors_by_category[category_key] = self.stats.errors_by_category.get(category_key, 0) + 1

        # Update prompt type counts
        if error.prompt_type:
            prompt_key = error.prompt_type.value
            self.stats.errors_by_prompt_type[prompt_key] = self.stats.errors_by_prompt_type.get(prompt_key, 0) + 1

        # Update severity counts
        severity_key = error.severity.value
        self.stats.errors_by_severity[severity_key] = self.stats.errors_by_severity.get(severity_key, 0) + 1

        # Add to recent errors
        self.stats.recent_errors.append(error)
        if len(self.stats.recent_errors) > 50:  # Keep last 50 errors
            self.stats.recent_errors.pop(0)

        # Update time-based statistics (simplified)
        current_time = datetime.now()
        hour_ago = current_time.replace(minute=0, second=0, microsecond=0)
        day_ago = current_time.replace(hour=0, minute=0, second=0, microsecond=0)

        self.stats.errors_last_hour = len([e for e in self.stats.recent_errors if e.timestamp >= hour_ago])

        self.stats.errors_last_day = len([e for e in self.stats.recent_errors if e.timestamp >= day_ago])

        # Calculate error rate (errors per minute in last hour)
        if self.stats.errors_last_hour > 0:
            self.stats.error_rate_per_minute = self.stats.errors_last_hour / 60.0

    def _trim_error_history(self):
        """Keep error history to a reasonable size."""
        max_history = 1000
        if len(self.error_history) > max_history:
            self.error_history = self.error_history[-max_history:]

    def _setup_recovery_strategies(self):
        """Setup recovery strategies for different error categories."""
        self.recovery_strategies = {
            ErrorCategory.SERVICE_ERROR: self._recover_service_error,
            ErrorCategory.TIMEOUT_ERROR: self._recover_timeout_error,
            ErrorCategory.RESOURCE_ERROR: self._recover_resource_error,
            ErrorCategory.SEARCH_ERROR: self._recover_search_error,
        }

    def _setup_error_logging(self):
        """Setup error-specific logging configuration."""
        # Add custom formatter for errors if needed
        pass

    def _recover_service_error(self, error: PromptError, **params) -> bool:
        """Attempt recovery from service errors."""
        # Implement service-specific recovery logic
        return False

    def _recover_timeout_error(self, error: PromptError, **params) -> bool:
        """Attempt recovery from timeout errors."""
        # Implement timeout recovery logic
        return False

    def _recover_resource_error(self, error: PromptError, **params) -> bool:
        """Attempt recovery from resource errors."""
        # Implement resource recovery logic
        return False

    def _recover_search_error(self, error: PromptError, **params) -> bool:
        """Attempt recovery from search errors."""
        # Implement search recovery logic
        return False


class PromptOperationError(Exception):
    """Exception raised when a prompt operation fails."""

    def __init__(self, prompt_error: PromptError):
        self.prompt_error = prompt_error
        super().__init__(prompt_error.user_message or prompt_error.message)


# Global error handler instance
prompt_error_handler = PromptErrorHandler()
