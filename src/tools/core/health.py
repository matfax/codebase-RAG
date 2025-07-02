"""Health check tool implementation.

This module provides health check functionality for the MCP server.
"""

import logging
import os
import time
from datetime import datetime
from typing import Any

from tools.core.memory_utils import get_memory_stats, get_memory_usage_mb

logger = logging.getLogger(__name__)


async def health_check() -> dict[str, Any]:
    """
    Check the health of the MCP server and its dependencies.

    This is the async version for MCP tool registration.

    Returns:
        Dict[str, Any]: Health status information including:
            - status: Overall health status ("ok", "warning", "error")
            - message: Human-readable status message
            - services: Individual service health statuses
            - memory: Memory usage information
            - timestamp: ISO format timestamp
    """
    # Use the synchronous implementation
    return health_check_sync()


def health_check_sync() -> dict[str, Any]:
    """
    Synchronous health check of the MCP server and its dependencies.

    Checks:
        - Qdrant database connectivity and performance
        - Ollama service availability
        - Memory usage status
        - System resources

    Returns:
        Dict[str, Any]: Comprehensive health status information
    """
    services_status = {}
    overall_status = "ok"
    issues = []
    warnings = []

    start_time = time.time()

    # Check Qdrant connection
    try:
        from qdrant_client import QdrantClient

        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))

        # Create client with connection check
        try:
            client = QdrantClient(host=host, port=port)

            # Import database utils here to avoid circular imports
            from ..database.qdrant_utils import (
                check_qdrant_health,
                retry_qdrant_operation,
            )

            # Use retry wrapper for health check
            qdrant_status = retry_qdrant_operation(
                lambda: check_qdrant_health(client),
                "Qdrant health check",
                max_retries=1,  # Quick check, don't retry too much
            )

            services_status["qdrant"] = qdrant_status

            if not qdrant_status["healthy"]:
                overall_status = "error"
                issues.append("Qdrant database is not healthy")
            elif qdrant_status.get("response_time_ms", 0) > 500:
                warnings.append(f"Qdrant response time is slow: {qdrant_status['response_time_ms']:.0f}ms")
                if overall_status == "ok":
                    overall_status = "warning"

        except Exception as e:
            services_status["qdrant"] = {
                "healthy": False,
                "error": str(e),
                "host": f"{host}:{port}",
                "timestamp": datetime.now().isoformat(),
            }
            overall_status = "error"
            issues.append(f"Cannot connect to Qdrant at {host}:{port}")

    except ImportError:
        services_status["qdrant"] = {
            "healthy": False,
            "error": "Qdrant client not installed",
            "timestamp": datetime.now().isoformat(),
        }
        overall_status = "error"
        issues.append("Qdrant client library not installed")

    # Check Ollama connection
    try:
        import requests

        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        # Basic connectivity check with timeout
        try:
            response = requests.get(f"{ollama_host}/api/tags", timeout=5)
            response_time = response.elapsed.total_seconds() * 1000

            if response.status_code == 200:
                services_status["ollama"] = {
                    "healthy": True,
                    "status": "accessible",
                    "host": ollama_host,
                    "response_time_ms": response_time,
                    "timestamp": datetime.now().isoformat(),
                }

                # Check for slow response
                if response_time > 1000:
                    warnings.append(f"Ollama response time is slow: {response_time:.0f}ms")
                    if overall_status == "ok":
                        overall_status = "warning"
            else:
                services_status["ollama"] = {
                    "healthy": False,
                    "error": f"Unexpected status code: {response.status_code}",
                    "host": ollama_host,
                    "timestamp": datetime.now().isoformat(),
                }
                overall_status = "error"
                issues.append(f"Ollama returned status code {response.status_code}")

        except requests.RequestException as e:
            services_status["ollama"] = {
                "healthy": False,
                "error": f"Connection failed: {str(e)}",
                "host": ollama_host,
                "timestamp": datetime.now().isoformat(),
            }
            overall_status = "error"
            issues.append(f"Cannot connect to Ollama at {ollama_host}")

    except ImportError:
        services_status["ollama"] = {
            "healthy": False,
            "error": "requests library not installed",
            "timestamp": datetime.now().isoformat(),
        }
        overall_status = "error"
        issues.append("requests library not installed")

    # Check memory usage
    try:
        memory_stats = get_memory_stats()
        memory_mb = memory_stats["process_memory_mb"]
        memory_threshold = float(os.getenv("MEMORY_WARNING_THRESHOLD_MB", "1000"))

        memory_info = {
            "current_mb": memory_mb,
            "threshold_mb": memory_threshold,
            "system_memory": memory_stats["system_memory"],
            "healthy": memory_mb < memory_threshold,
        }

        services_status["memory"] = memory_info

        if memory_mb > memory_threshold:
            warnings.append(f"Memory usage is high: {memory_mb:.0f}MB (threshold: {memory_threshold:.0f}MB)")
            if overall_status == "ok":
                overall_status = "warning"

        # Check system memory availability
        system_mem = memory_stats["system_memory"]
        if system_mem.get("percent_used", 0) > 90:
            warnings.append(f"System memory usage is critical: {system_mem['percent_used']:.0f}%")
            if overall_status != "error":
                overall_status = "warning"

    except Exception as e:
        services_status["memory"] = {
            "healthy": True,  # Don't fail health check if we can't get memory stats
            "error": f"Could not get memory stats: {str(e)}",
            "timestamp": datetime.now().isoformat(),
        }

    # Calculate total check time
    total_check_time = (time.time() - start_time) * 1000

    # Build comprehensive message
    if overall_status == "ok":
        message = "All services are operational"
    elif overall_status == "warning":
        message = f"System operational with warnings: {'; '.join(warnings)}"
    else:
        message = f"System has critical issues: {'; '.join(issues)}"
        if warnings:
            message += f". Also: {'; '.join(warnings)}"

    return {
        "status": overall_status,
        "message": message,
        "services": services_status,
        "issues": issues if issues else None,
        "warnings": warnings if warnings else None,
        "dependencies_checked": ["qdrant", "ollama", "memory"],
        "check_duration_ms": total_check_time,
        "timestamp": datetime.now().isoformat(),
    }


def basic_health_check() -> dict[str, Any]:
    """
    Synchronous basic health check without external dependencies.

    This is a minimal health check that only verifies the MCP server
    itself is running, without checking external services.

    Returns:
        Dict[str, Any]: Basic health status
    """
    try:
        # Basic memory check
        memory_mb = get_memory_usage_mb()

        return {
            "status": "ok",
            "message": "MCP server is running",
            "memory_mb": memory_mb,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "status": "ok",  # Server is still running even if we can't get memory
            "message": "MCP server is running",
            "timestamp": datetime.now().isoformat(),
            "note": f"Could not get memory stats: {str(e)}",
        }
