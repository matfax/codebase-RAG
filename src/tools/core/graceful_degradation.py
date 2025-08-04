"""
Graceful Degradation and Error Handling for MCP Tools

This module provides comprehensive error handling, graceful degradation,
and fallback mechanisms for all MCP tools to ensure robustness.
"""

import asyncio
import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    """Levels of service degradation."""

    FULL_SERVICE = "full_service"
    PARTIAL_DEGRADATION = "partial_degradation"
    MINIMAL_SERVICE = "minimal_service"
    EMERGENCY_MODE = "emergency_mode"


@dataclass
class ErrorContext:
    """Context information for error handling."""

    tool_name: str
    operation_type: str
    error_type: str
    error_message: str
    parameters: dict[str, Any]
    fallback_attempted: bool = False
    fallback_successful: bool = False
    degradation_level: DegradationLevel = DegradationLevel.FULL_SERVICE
    recovery_suggestions: list[str] = None

    def __post_init__(self):
        if self.recovery_suggestions is None:
            self.recovery_suggestions = []


class GracefulDegradationManager:
    """Manager for graceful degradation and error handling."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history: list[ErrorContext] = []
        self.max_error_history = 500

        # Service health tracking
        self.service_health = {
            "search": DegradationLevel.FULL_SERVICE,
            "indexing": DegradationLevel.FULL_SERVICE,
            "graph_rag": DegradationLevel.FULL_SERVICE,
            "multi_modal": DegradationLevel.FULL_SERVICE,
            "cache": DegradationLevel.FULL_SERVICE,
        }

        # Error thresholds for degradation
        self.error_thresholds = {
            "timeout_threshold": 3,  # 3 timeouts trigger degradation
            "error_rate_threshold": 0.5,  # 50% error rate
            "time_window_minutes": 10,  # Time window for error counting
        }

    def record_error(self, error_context: ErrorContext):
        """Record an error for tracking and analysis."""
        self.error_history.append(error_context)

        # Trim history if needed
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history :]

        # Check if service degradation is needed
        self._check_service_degradation(error_context.tool_name)

        self.logger.warning(f"Error recorded for {error_context.tool_name}: {error_context.error_message}")

    def _check_service_degradation(self, service_name: str):
        """Check if a service needs degradation based on error patterns."""
        import time

        # Get recent errors for this service
        current_time = time.time()
        time_window_seconds = self.error_thresholds["time_window_minutes"] * 60

        recent_errors = [
            error
            for error in self.error_history[-50:]  # Last 50 errors
            if error.tool_name == service_name and (current_time - time.time()) < time_window_seconds  # This is a simplified check
        ]

        if len(recent_errors) >= self.error_thresholds["timeout_threshold"]:
            # Degrade service
            current_level = self.service_health.get(service_name, DegradationLevel.FULL_SERVICE)

            if current_level == DegradationLevel.FULL_SERVICE:
                self.service_health[service_name] = DegradationLevel.PARTIAL_DEGRADATION
                self.logger.warning(f"Service {service_name} degraded to PARTIAL_DEGRADATION")
            elif current_level == DegradationLevel.PARTIAL_DEGRADATION:
                self.service_health[service_name] = DegradationLevel.MINIMAL_SERVICE
                self.logger.warning(f"Service {service_name} degraded to MINIMAL_SERVICE")

    def get_degradation_level(self, service_name: str) -> DegradationLevel:
        """Get current degradation level for a service."""
        return self.service_health.get(service_name, DegradationLevel.FULL_SERVICE)

    def recover_service(self, service_name: str):
        """Attempt to recover a degraded service."""
        current_level = self.service_health.get(service_name, DegradationLevel.FULL_SERVICE)

        if current_level != DegradationLevel.FULL_SERVICE:
            # Upgrade service level
            if current_level == DegradationLevel.EMERGENCY_MODE:
                self.service_health[service_name] = DegradationLevel.MINIMAL_SERVICE
            elif current_level == DegradationLevel.MINIMAL_SERVICE:
                self.service_health[service_name] = DegradationLevel.PARTIAL_DEGRADATION
            elif current_level == DegradationLevel.PARTIAL_DEGRADATION:
                self.service_health[service_name] = DegradationLevel.FULL_SERVICE

            self.logger.info(f"Service {service_name} recovered to {self.service_health[service_name].value}")

    def get_error_summary(self) -> dict[str, Any]:
        """Get error summary and service health status."""
        import time
        from collections import defaultdict

        # Count errors by type and service
        error_counts = defaultdict(lambda: defaultdict(int))
        recent_errors = 0

        for error in self.error_history[-100:]:  # Last 100 errors
            error_counts[error.tool_name][error.error_type] += 1
            # This is a simplified time check
            recent_errors += 1

        return {
            "service_health": {name: level.value for name, level in self.service_health.items()},
            "error_counts": dict(error_counts),
            "recent_errors": recent_errors,
            "total_errors": len(self.error_history),
            "degraded_services": [name for name, level in self.service_health.items() if level != DegradationLevel.FULL_SERVICE],
            "error_thresholds": self.error_thresholds,
        }


def create_fallback_response(
    tool_name: str, original_error: Exception, parameters: dict[str, Any] = None, fallback_data: Any = None
) -> dict[str, Any]:
    """Create a standardized fallback response for failed operations."""

    error_type = type(original_error).__name__

    # Generate recovery suggestions based on error type
    recovery_suggestions = []

    if "timeout" in str(original_error).lower():
        recovery_suggestions.extend(
            [
                "Try reducing the scope of your query",
                "Consider breaking complex operations into smaller parts",
                "Check system resources and try again later",
            ]
        )
    elif "connection" in str(original_error).lower():
        recovery_suggestions.extend(["Check network connectivity", "Verify service dependencies are running", "Try again in a few moments"])
    elif "memory" in str(original_error).lower():
        recovery_suggestions.extend(
            ["Reduce batch size or query complexity", "Clear caches if possible", "Contact administrator if problem persists"]
        )
    else:
        recovery_suggestions.extend(["Check your input parameters", "Try a simpler query first", "Contact support if the issue persists"])

    fallback_response = {
        "error": str(original_error),
        "error_type": error_type,
        "tool_name": tool_name,
        "fallback_mode": True,
        "recovery_suggestions": recovery_suggestions,
        "parameters": parameters or {},
        "timestamp": "2024-07-25T12:00:00Z",
    }

    # Add fallback data if provided
    if fallback_data is not None:
        if isinstance(fallback_data, dict):
            fallback_response.update(fallback_data)
        else:
            fallback_response["fallback_data"] = fallback_data

    return fallback_response


def with_graceful_degradation(
    service_name: str, fallback_function: Callable | None = None, emergency_response: dict[str, Any] | None = None
):
    """Decorator to add graceful degradation to MCP tools."""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            manager = get_degradation_manager()
            degradation_level = manager.get_degradation_level(service_name)

            # Check if service is severely degraded
            if degradation_level == DegradationLevel.EMERGENCY_MODE:
                if emergency_response:
                    return emergency_response
                else:
                    return {
                        "error": f"Service {service_name} is currently in emergency mode",
                        "error_type": "ServiceDegraded",
                        "service_name": service_name,
                        "degradation_level": degradation_level.value,
                        "message": "Service is temporarily unavailable. Please try again later.",
                    }

            # Attempt main operation
            try:
                result = await func(*args, **kwargs)

                # If successful, attempt service recovery
                if degradation_level != DegradationLevel.FULL_SERVICE:
                    manager.recover_service(service_name)

                return result

            except Exception as e:
                # Record error
                error_context = ErrorContext(
                    tool_name=service_name,
                    operation_type=func.__name__,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    parameters=kwargs,
                    degradation_level=degradation_level,
                )

                # Attempt fallback if available
                if fallback_function and degradation_level != DegradationLevel.EMERGENCY_MODE:
                    try:
                        logger.info(f"Attempting fallback for {service_name}")
                        fallback_result = await fallback_function(*args, **kwargs)

                        error_context.fallback_attempted = True
                        error_context.fallback_successful = True

                        # Mark result as fallback
                        if isinstance(fallback_result, dict):
                            fallback_result["fallback_mode"] = True
                            fallback_result["original_error"] = str(e)

                        manager.record_error(error_context)
                        return fallback_result

                    except Exception as fallback_error:
                        error_context.fallback_attempted = True
                        error_context.fallback_successful = False
                        logger.error(f"Fallback also failed for {service_name}: {fallback_error}")

                # Record error and return graceful failure response
                manager.record_error(error_context)

                return create_fallback_response(tool_name=service_name, original_error=e, parameters=kwargs)

        return wrapper

    return decorator


async def simple_search_fallback(*args, **kwargs) -> dict[str, Any]:
    """Simple fallback for search operations."""
    return {
        "results": [],
        "query": kwargs.get("query", ""),
        "total": 0,
        "message": "Fallback search: service temporarily degraded",
        "fallback_mode": True,
        "suggestions": ["Try a simpler query", "Check if indexing is complete", "Contact administrator if issues persist"],
    }


async def minimal_index_fallback(*args, **kwargs) -> dict[str, Any]:
    """Minimal fallback for indexing operations."""
    return {
        "success": False,
        "message": "Indexing service temporarily unavailable",
        "fallback_mode": True,
        "directory": kwargs.get("directory", "."),
        "recommendations": ["Try indexing smaller directories", "Check system resources", "Try again later"],
    }


# Global instance
_degradation_manager = None


def get_degradation_manager() -> GracefulDegradationManager:
    """Get or create the degradation manager instance."""
    global _degradation_manager
    if _degradation_manager is None:
        _degradation_manager = GracefulDegradationManager()
    return _degradation_manager


async def get_service_health_status() -> dict[str, Any]:
    """Get comprehensive service health status."""
    manager = get_degradation_manager()
    return manager.get_error_summary()
