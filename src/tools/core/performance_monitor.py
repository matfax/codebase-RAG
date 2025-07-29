"""
Performance Monitoring for MCP Tools

This module provides comprehensive performance monitoring, timeout handling,
and performance optimization for all MCP tools to ensure <15 second responses.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for MCP tool operations."""

    tool_name: str
    start_time: float
    end_time: float | None = None
    execution_time_ms: float | None = None
    timeout_seconds: int = 15
    timed_out: bool = False
    memory_usage_mb: float | None = None
    cpu_usage_percent: float | None = None
    success: bool = True
    error_message: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    result_size: int = 0

    def finish(self, success: bool = True, error_message: str | None = None):
        """Mark the operation as finished and calculate metrics."""
        self.end_time = time.time()
        self.execution_time_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.error_message = error_message

        # Capture system metrics
        try:
            process = psutil.Process()
            self.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            self.cpu_usage_percent = process.cpu_percent()
        except Exception:
            pass

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "tool_name": self.tool_name,
            "execution_time_ms": self.execution_time_ms,
            "timeout_seconds": self.timeout_seconds,
            "timed_out": self.timed_out,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "success": self.success,
            "error_message": self.error_message,
            "parameters": self.parameters,
            "result_size": self.result_size,
        }


class PerformanceMonitor:
    """Comprehensive performance monitor for MCP tools."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_history: list[PerformanceMetrics] = []
        self.max_history_size = 1000

        # Performance thresholds
        self.warning_threshold_ms = 10000  # 10 seconds
        self.critical_threshold_ms = 15000  # 15 seconds

        # Active operations
        self.active_operations: dict[str, PerformanceMetrics] = {}

    def start_monitoring(self, tool_name: str, parameters: dict[str, Any] = None, timeout_seconds: int = 15) -> str:
        """Start monitoring a tool operation."""
        operation_id = f"{tool_name}_{time.time()}_{id(parameters)}"

        metrics = PerformanceMetrics(
            tool_name=tool_name, start_time=time.time(), timeout_seconds=timeout_seconds, parameters=parameters or {}
        )

        self.active_operations[operation_id] = metrics

        self.logger.debug(f"Started monitoring operation: {operation_id}")
        return operation_id

    def finish_monitoring(self, operation_id: str, success: bool = True, error_message: str | None = None, result_size: int = 0):
        """Finish monitoring a tool operation."""
        if operation_id not in self.active_operations:
            self.logger.warning(f"Unknown operation ID: {operation_id}")
            return

        metrics = self.active_operations.pop(operation_id)
        metrics.finish(success, error_message)
        metrics.result_size = result_size

        # Add to history
        self.metrics_history.append(metrics)

        # Trim history if needed
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size :]

        # Log performance warnings
        if metrics.execution_time_ms and metrics.execution_time_ms > self.critical_threshold_ms:
            self.logger.warning(
                f"CRITICAL: Tool {metrics.tool_name} took {metrics.execution_time_ms:.2f}ms " f"(>{self.critical_threshold_ms}ms threshold)"
            )
        elif metrics.execution_time_ms and metrics.execution_time_ms > self.warning_threshold_ms:
            self.logger.warning(
                f"WARNING: Tool {metrics.tool_name} took {metrics.execution_time_ms:.2f}ms " f"(>{self.warning_threshold_ms}ms threshold)"
            )

        self.logger.debug(f"Finished monitoring operation: {operation_id}")

    def get_performance_summary(self, tool_name: str | None = None) -> dict[str, Any]:
        """Get performance summary for tools."""
        filtered_metrics = self.metrics_history

        if tool_name:
            filtered_metrics = [m for m in self.metrics_history if m.tool_name == tool_name]

        if not filtered_metrics:
            return {"tool_name": tool_name or "all", "total_operations": 0, "summary": "No data available"}

        # Calculate statistics
        execution_times = [m.execution_time_ms for m in filtered_metrics if m.execution_time_ms is not None]
        success_count = sum(1 for m in filtered_metrics if m.success)
        timeout_count = sum(1 for m in filtered_metrics if m.timed_out)

        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
        max_time = max(execution_times) if execution_times else 0
        min_time = min(execution_times) if execution_times else 0

        # Performance percentiles
        if execution_times:
            sorted_times = sorted(execution_times)
            p95_time = sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 20 else max_time
            p99_time = sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 100 else max_time
        else:
            p95_time = p99_time = 0

        return {
            "tool_name": tool_name or "all",
            "total_operations": len(filtered_metrics),
            "successful_operations": success_count,
            "failed_operations": len(filtered_metrics) - success_count,
            "timeout_operations": timeout_count,
            "success_rate": (success_count / len(filtered_metrics)) * 100 if filtered_metrics else 0,
            "performance": {
                "average_time_ms": avg_time,
                "min_time_ms": min_time,
                "max_time_ms": max_time,
                "p95_time_ms": p95_time,
                "p99_time_ms": p99_time,
            },
            "compliance": {
                "under_10s": sum(1 for t in execution_times if t < 10000),
                "under_15s": sum(1 for t in execution_times if t < 15000),
                "over_15s": sum(1 for t in execution_times if t >= 15000),
                "compliance_rate": (sum(1 for t in execution_times if t < 15000) / len(execution_times)) * 100 if execution_times else 100,
            },
        }

    def get_active_operations(self) -> list[dict[str, Any]]:
        """Get currently active operations."""
        current_time = time.time()
        active_ops = []

        for operation_id, metrics in self.active_operations.items():
            elapsed_time = (current_time - metrics.start_time) * 1000
            remaining_time = max(0, (metrics.timeout_seconds * 1000) - elapsed_time)

            active_ops.append(
                {
                    "operation_id": operation_id,
                    "tool_name": metrics.tool_name,
                    "elapsed_time_ms": elapsed_time,
                    "remaining_time_ms": remaining_time,
                    "timeout_seconds": metrics.timeout_seconds,
                    "parameters": metrics.parameters,
                }
            )

        return active_ops

    def cleanup_stale_operations(self, max_age_seconds: int = 300):
        """Clean up stale operations that might have been orphaned."""
        current_time = time.time()
        stale_operations = []

        for operation_id, metrics in list(self.active_operations.items()):
            age_seconds = current_time - metrics.start_time
            if age_seconds > max_age_seconds:
                stale_operations.append(operation_id)
                self.finish_monitoring(operation_id, success=False, error_message="Operation timeout/cleanup")

        if stale_operations:
            self.logger.warning(f"Cleaned up {len(stale_operations)} stale operations")

        return len(stale_operations)


def with_performance_monitoring(timeout_seconds: int = 15, tool_name: str | None = None):
    """Decorator to add performance monitoring and timeout handling to MCP tools."""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get tool name
            actual_tool_name = tool_name or func.__name__

            # Get monitor instance
            monitor = get_performance_monitor()

            # Start monitoring
            operation_id = monitor.start_monitoring(tool_name=actual_tool_name, parameters=kwargs, timeout_seconds=timeout_seconds)

            try:
                # Execute with timeout
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)

                # Calculate result size
                result_size = 0
                if isinstance(result, dict):
                    result_size = len(str(result))
                elif isinstance(result, (list, tuple)):
                    result_size = len(result)

                # Finish monitoring successfully
                monitor.finish_monitoring(operation_id, success=True, result_size=result_size)

                # Add performance metadata to result
                if isinstance(result, dict):
                    operation_metrics = monitor.metrics_history[-1] if monitor.metrics_history else None
                    if operation_metrics:
                        result["_performance"] = {
                            "execution_time_ms": operation_metrics.execution_time_ms,
                            "within_timeout": operation_metrics.execution_time_ms < (timeout_seconds * 1000),
                            "memory_usage_mb": operation_metrics.memory_usage_mb,
                            "cpu_usage_percent": operation_metrics.cpu_usage_percent,
                        }

                return result

            except asyncio.TimeoutError:
                # Handle timeout
                monitor.finish_monitoring(operation_id, success=False, error_message=f"Operation timed out after {timeout_seconds} seconds")

                # Return timeout response
                return {
                    "error": f"Operation timed out after {timeout_seconds} seconds",
                    "error_type": "TimeoutError",
                    "tool_name": actual_tool_name,
                    "timeout_seconds": timeout_seconds,
                    "suggestion": "Try reducing query complexity or increasing timeout",
                    "_performance": {
                        "timed_out": True,
                        "timeout_seconds": timeout_seconds,
                    },
                }

            except Exception as e:
                # Handle other errors
                error_message = str(e)
                monitor.finish_monitoring(operation_id, success=False, error_message=error_message)

                # Re-raise the exception
                raise

        return wrapper

    return decorator


# Global instance
_performance_monitor = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create the performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


async def get_performance_dashboard() -> dict[str, Any]:
    """Get comprehensive performance dashboard data."""
    monitor = get_performance_monitor()

    # Cleanup stale operations
    stale_count = monitor.cleanup_stale_operations()

    # Get overall summary
    overall_summary = monitor.get_performance_summary()

    # Get per-tool summaries for key tools
    tool_summaries = {}
    key_tools = ["search", "index_directory", "multi_modal_search", "graph_analyze_structure"]

    for tool in key_tools:
        tool_summaries[tool] = monitor.get_performance_summary(tool)

    # Get active operations
    active_ops = monitor.get_active_operations()

    # System performance
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(".")

        system_metrics = {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_gb": memory.available / 1024 / 1024 / 1024,
            "disk_usage_percent": (disk.used / disk.total) * 100,
            "disk_free_gb": disk.free / 1024 / 1024 / 1024,
        }
    except Exception as e:
        system_metrics = {"error": f"Could not get system metrics: {e}"}

    return {
        "timestamp": time.time(),
        "overall_summary": overall_summary,
        "tool_summaries": tool_summaries,
        "active_operations": active_ops,
        "system_metrics": system_metrics,
        "maintenance": {
            "stale_operations_cleaned": stale_count,
            "history_size": len(monitor.metrics_history),
            "max_history_size": monitor.max_history_size,
        },
        "performance_targets": {
            "target_response_time_ms": 15000,
            "warning_threshold_ms": monitor.warning_threshold_ms,
            "critical_threshold_ms": monitor.critical_threshold_ms,
        },
    }
