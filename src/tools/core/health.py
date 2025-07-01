"""Health check tool implementation.

This module provides health check functionality for the MCP server.
"""

import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


async def health_check() -> Dict[str, Any]:
    """
    Check the health of the MCP server and its dependencies.
    
    Returns:
        Dict[str, Any]: Health status information
    """
    services_status = {}
    overall_healthy = True
    
    # Check Qdrant connection
    try:
        from ..database.qdrant_utils import check_qdrant_health
        from qdrant_client import QdrantClient
        
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        
        client = QdrantClient(host=host, port=port)
        qdrant_status = check_qdrant_health(client)
        services_status["qdrant"] = qdrant_status
        
        if not qdrant_status["healthy"]:
            overall_healthy = False
            
    except Exception as e:
        services_status["qdrant"] = {
            "healthy": False,
            "error": f"Failed to check Qdrant: {str(e)}"
        }
        overall_healthy = False
    
    # Check Ollama connection (basic connectivity)
    try:
        import requests
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            services_status["ollama"] = {
                "healthy": True,
                "status": "accessible",
                "host": ollama_host
            }
        else:
            services_status["ollama"] = {
                "healthy": False,
                "error": f"Unexpected status code: {response.status_code}"
            }
            overall_healthy = False
            
    except Exception as e:
        services_status["ollama"] = {
            "healthy": False,
            "error": f"Failed to check Ollama: {str(e)}"
        }
        overall_healthy = False
    
    return {
        "status": "healthy" if overall_healthy else "unhealthy",
        "message": "All services are operational" if overall_healthy else "Some services have issues",
        "services": services_status,
        "dependencies_checked": ["qdrant", "ollama"]
    }


def basic_health_check() -> Dict[str, Any]:
    """
    Synchronous basic health check without external dependencies.
    
    Returns:
        Dict[str, Any]: Basic health status
    """
    return {
        "status": "ok",
        "message": "MCP server is running",
        "timestamp": "system_operational"
    }