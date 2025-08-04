"""
Enhanced Debug Logger for MCP Tools

This module provides structured logging capabilities specifically designed
for debugging MCP tool interactions, multi-modal search modes, and service
call chains.
"""

import json
import logging
import time
import uuid
from contextvars import ContextVar
from functools import wraps
from typing import Any, Dict, Optional

# Import centralized logging configuration
try:
    from ...config.logging_config import get_logger, get_logging_config

    _use_centralized_logging = True
except ImportError:
    _use_centralized_logging = False

# Request tracking context
request_id: ContextVar[str] = ContextVar("request_id", default="")


class StructuredLogger:
    """Enhanced logger with structured context and request tracking."""

    def __init__(self, component_name: str):
        self.component = component_name
        if _use_centralized_logging:
            self.logger = get_logger(component_name)
        else:
            self.logger = logging.getLogger(f"mcp.{component_name}")

    def _get_context(self, **kwargs) -> dict[str, Any]:
        """Build structured log context."""
        req_id = request_id.get()
        return {"component": self.component, "request_id": req_id, "timestamp": time.time(), **kwargs}

    def debug(self, message: str, **kwargs):
        """Debug level with context."""
        context = self._get_context(**kwargs)
        req_id = context["request_id"]
        prefix = f"[REQ-{req_id}] [{self.component}]" if req_id else f"[{self.component}]"
        self.logger.debug(f"{prefix} {message}", extra=context)

    def info(self, message: str, **kwargs):
        """Info level with context."""
        context = self._get_context(**kwargs)
        req_id = context["request_id"]
        prefix = f"[REQ-{req_id}] [{self.component}]" if req_id else f"[{self.component}]"
        self.logger.info(f"{prefix} {message}", extra=context)

    def warning(self, message: str, **kwargs):
        """Warning level with context."""
        context = self._get_context(**kwargs)
        req_id = context["request_id"]
        prefix = f"[REQ-{req_id}] [{self.component}]" if req_id else f"[{self.component}]"
        self.logger.warning(f"{prefix} {message}", extra=context)

    def error(self, message: str, **kwargs):
        """Error level with context."""
        context = self._get_context(**kwargs)
        req_id = context["request_id"]
        prefix = f"[REQ-{req_id}] [{self.component}]" if req_id else f"[{self.component}]"
        self.logger.error(f"{prefix} {message}", extra=context)


class MultiModalDebugLogger:
    """Specialized logger for multi-modal search debugging."""

    def __init__(self):
        self.logger = StructuredLogger("multi_modal")

    def log_mode_decision(self, query: str, requested_mode: str | None, actual_mode: str, decision_reason: str):
        """Log multi-modal mode selection decision."""
        self.logger.info(
            f"Mode Decision - Query: '{query[:30]}...', "
            f"Requested: {requested_mode or 'auto'}, "
            f"Actual: {actual_mode}, "
            f"Reason: {decision_reason}",
            query_preview=query[:50],
            requested_mode=requested_mode,
            actual_mode=actual_mode,
            decision_reason=decision_reason,
        )

    def log_fallback_trigger(self, original_mode: str, fallback_mode: str, error_message: str, fallback_reason: str):
        """Log multi-modal fallback scenarios."""
        self.logger.warning(
            f"Fallback Triggered - {original_mode} → {fallback_mode}, " f"Error: {error_message}, " f"Reason: {fallback_reason}",
            original_mode=original_mode,
            fallback_mode=fallback_mode,
            error_message=error_message,
            fallback_reason=fallback_reason,
        )

    def log_mode_execution(self, mode: str, query: str, execution_path: str, parameters: dict[str, Any]):
        """Log actual mode execution details."""
        self.logger.debug(
            f"Mode Execution - {mode} processing '{query[:30]}...' " f"via {execution_path}",
            mode=mode,
            query_preview=query[:50],
            execution_path=execution_path,
            parameters=parameters,
        )


class ServiceCallLogger:
    """Logger for tracking service-to-service calls."""

    def __init__(self):
        self.logger = StructuredLogger("service_calls")

    def log_call_start(self, from_service: str, to_service: str, method: str, params: dict[str, Any]):
        """Log the start of a service call."""
        self.logger.debug(
            f"{from_service} → {to_service}.{method}({list(params.keys())})",
            from_service=from_service,
            to_service=to_service,
            method=method,
            param_keys=list(params.keys()),
            call_direction="outbound",
        )

    def log_call_complete(
        self, from_service: str, to_service: str, method: str, success: bool, duration_ms: float, result_summary: str | None = None
    ):
        """Log the completion of a service call."""
        status = "✓" if success else "✗"
        self.logger.debug(
            f"{from_service} ← {to_service}.{method} {status} ({duration_ms:.1f}ms)" + (f" - {result_summary}" if result_summary else ""),
            from_service=from_service,
            to_service=to_service,
            method=method,
            success=success,
            duration_ms=duration_ms,
            result_summary=result_summary,
            call_direction="response",
        )


class ConfigurationLogger:
    """Logger for configuration and feature toggle decisions."""

    def __init__(self):
        self.logger = StructuredLogger("configuration")

    def log_runtime_config(self, tool_name: str, config: dict[str, Any]):
        """Log runtime configuration for a tool."""
        self.logger.info(f"Runtime Config for {tool_name}: {json.dumps(config, indent=2)}", tool_name=tool_name, configuration=config)

    def log_auto_configuration(self, decisions: dict[str, Any]):
        """Log auto-configuration decisions."""
        self.logger.info(f"Auto-configuration decisions: {json.dumps(decisions, indent=2)}", decisions=decisions)

    def log_feature_toggle(self, feature: str, enabled: bool, reason: str):
        """Log feature toggle decisions."""
        status = "ENABLED" if enabled else "DISABLED"
        self.logger.info(f"Feature '{feature}' {status} - {reason}", feature=feature, enabled=enabled, reason=reason)


def with_request_tracking(func):
    """Decorator to add request tracking to MCP tool functions."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Generate unique request ID
        req_id = str(uuid.uuid4())[:8]
        request_id.set(req_id)

        # Create logger for this function
        logger = StructuredLogger(func.__module__.split(".")[-1])

        # Log function start
        logger.info(f"Starting {func.__qualname__}", function=func.__qualname__, args_count=len(args), kwargs_keys=list(kwargs.keys()))

        start_time = time.time()

        try:
            result = await func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000

            # Log successful completion
            logger.info(
                f"Completed {func.__qualname__} in {duration_ms:.1f}ms", function=func.__qualname__, duration_ms=duration_ms, success=True
            )

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            # Log failure
            logger.error(
                f"Failed {func.__qualname__} after {duration_ms:.1f}ms: {str(e)}",
                function=func.__qualname__,
                duration_ms=duration_ms,
                success=False,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

    return wrapper


# Global instances for easy access
multi_modal_logger = MultiModalDebugLogger()
service_call_logger = ServiceCallLogger()
config_logger = ConfigurationLogger()


# Convenience function for quick debugging
def debug_log(component: str, message: str, **kwargs):
    """Quick debug logging with automatic request context."""
    logger = StructuredLogger(component)
    logger.debug(message, **kwargs)
