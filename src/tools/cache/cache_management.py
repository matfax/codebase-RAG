"""
Cache management tools for manual cache invalidation.

This module provides MCP tools for:
- Manual cache invalidation operations
- Cache inspection and debugging
- Cache statistics and monitoring
- Project-specific cache management
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

from ..core.error_utils import handle_tool_error, log_tool_usage
from ..core.errors import CacheError

logger = logging.getLogger(__name__)


async def manual_invalidate_file_cache(
    file_path: str,
    reason: str = "manual_invalidation",
    cascade: bool = True,
    use_partial: bool = True,
    old_content: str | None = None,
    new_content: str | None = None,
    project_name: str | None = None,
) -> dict[str, Any]:
    """
    Manually invalidate cache entries for a specific file.

    Args:
        file_path: Path to the file to invalidate
        reason: Reason for invalidation (for logging)
        cascade: Whether to cascade invalidation to dependent caches
        use_partial: Whether to use partial invalidation if content is provided
        old_content: Previous content of the file (for partial invalidation)
        new_content: New content of the file (for partial invalidation)
        project_name: Project name for scoped invalidation

    Returns:
        Dictionary with invalidation results and statistics
    """
    with log_tool_usage(
        "manual_invalidate_file_cache",
        {
            "file_path": file_path,
            "reason": reason,
            "cascade": cascade,
            "use_partial": use_partial,
            "has_old_content": old_content is not None,
            "has_new_content": new_content is not None,
            "project_name": project_name,
        },
    ):
        try:
            from ...services.cache_invalidation_service import (
                InvalidationReason,
                get_cache_invalidation_service,
            )

            # Validate file path
            abs_path = str(Path(file_path).resolve())
            if not Path(abs_path).exists():
                logger.warning(f"File does not exist: {abs_path}")
                # Continue with invalidation anyway - file might have been deleted

            # Get invalidation service
            invalidation_service = await get_cache_invalidation_service()

            # Map reason string to enum
            reason_mapping = {
                "manual_invalidation": InvalidationReason.MANUAL_INVALIDATION,
                "file_modified": InvalidationReason.FILE_MODIFIED,
                "file_deleted": InvalidationReason.FILE_DELETED,
                "file_added": InvalidationReason.FILE_ADDED,
                "content_changed": InvalidationReason.PARTIAL_CONTENT_CHANGE,
                "metadata_changed": InvalidationReason.METADATA_ONLY_CHANGE,
            }
            invalidation_reason = reason_mapping.get(reason, InvalidationReason.MANUAL_INVALIDATION)

            # Choose invalidation method
            if use_partial and (old_content is not None or new_content is not None):
                # Use partial invalidation for better optimization
                event = await invalidation_service.partial_invalidate_file_caches(abs_path, old_content, new_content, project_name, cascade)
                invalidation_type = "partial"
            else:
                # Use full invalidation
                event = await invalidation_service.invalidate_file_caches(abs_path, invalidation_reason, cascade)
                invalidation_type = "full"

            # Get statistics
            stats = invalidation_service.get_invalidation_stats()

            return {
                "success": True,
                "invalidation_type": invalidation_type,
                "event_id": event.event_id,
                "reason": event.reason.value,
                "affected_keys": len(event.affected_keys),
                "affected_files": event.affected_files,
                "timestamp": event.timestamp.isoformat(),
                "project_name": event.project_name,
                "optimization_info": (
                    {
                        "optimization_ratio": getattr(event.partial_result, "optimization_ratio", None),
                        "preserved_keys": len(getattr(event.partial_result, "preservation_keys", [])),
                        "invalidation_strategy": (
                            getattr(event.partial_result, "invalidation_type", {}).value
                            if hasattr(getattr(event.partial_result, "invalidation_type", None), "value")
                            else None
                        ),
                    }
                    if hasattr(event, "partial_result") and event.partial_result
                    else None
                ),
                "statistics": {
                    "total_invalidations": stats.total_invalidations,
                    "partial_invalidations": stats.partial_invalidations,
                    "keys_invalidated": stats.keys_invalidated,
                    "keys_preserved": stats.keys_preserved,
                    "avg_optimization_ratio": stats.avg_optimization_ratio,
                },
                "metadata": event.metadata,
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "manual_invalidate_file_cache",
                {
                    "file_path": file_path,
                    "reason": reason,
                },
            )


async def manual_invalidate_project_cache(
    project_name: str,
    reason: str = "manual_invalidation",
    invalidation_scope: str = "cascade",
    strategy: str = "immediate",
) -> dict[str, Any]:
    """
    Manually invalidate all cache entries for a project.

    Args:
        project_name: Name of the project to invalidate
        reason: Reason for invalidation (for logging)
        invalidation_scope: Scope of invalidation (file_only, project_wide, cascade, conservative, aggressive)
        strategy: Invalidation strategy (immediate, lazy, batch, scheduled)

    Returns:
        Dictionary with invalidation results and statistics
    """
    with log_tool_usage(
        "manual_invalidate_project_cache",
        {
            "project_name": project_name,
            "reason": reason,
            "invalidation_scope": invalidation_scope,
            "strategy": strategy,
        },
    ):
        try:
            from ...services.cache_invalidation_service import (
                InvalidationReason,
                InvalidationStrategy,
                ProjectInvalidationScope,
                ProjectInvalidationTrigger,
                get_cache_invalidation_service,
            )

            # Get invalidation service
            invalidation_service = await get_cache_invalidation_service()

            # Map reason string to enum
            reason_mapping = {
                "manual_invalidation": InvalidationReason.MANUAL_INVALIDATION,
                "project_changed": InvalidationReason.PROJECT_CHANGED,
                "dependency_changed": InvalidationReason.DEPENDENCY_CHANGED,
                "system_upgrade": InvalidationReason.SYSTEM_UPGRADE,
            }
            invalidation_reason = reason_mapping.get(reason, InvalidationReason.MANUAL_INVALIDATION)

            # Map scope string to enum
            scope_mapping = {
                "file_only": ProjectInvalidationScope.FILE_ONLY,
                "project_wide": ProjectInvalidationScope.PROJECT_WIDE,
                "cascade": ProjectInvalidationScope.CASCADE,
                "conservative": ProjectInvalidationScope.CONSERVATIVE,
                "aggressive": ProjectInvalidationScope.AGGRESSIVE,
            }
            scope_enum = scope_mapping.get(invalidation_scope, ProjectInvalidationScope.CASCADE)

            # Map strategy string to enum
            strategy_mapping = {
                "immediate": InvalidationStrategy.IMMEDIATE,
                "lazy": InvalidationStrategy.LAZY,
                "batch": InvalidationStrategy.BATCH,
                "scheduled": InvalidationStrategy.SCHEDULED,
            }
            strategy_enum = strategy_mapping.get(strategy, InvalidationStrategy.IMMEDIATE)

            # Create or update project policy if needed
            policy = invalidation_service.get_project_invalidation_policy(project_name)
            if policy.project_name == "__default__":
                # Create project-specific policy
                policy = invalidation_service.create_project_policy(
                    project_name=project_name,
                    scope=scope_enum,
                    strategy=strategy_enum,
                )

            # Perform invalidation
            if scope_enum != ProjectInvalidationScope.FILE_ONLY and strategy_enum == InvalidationStrategy.IMMEDIATE:
                # Use project-wide invalidation
                event = await invalidation_service.invalidate_project_with_policy(
                    project_name, invalidation_reason, ProjectInvalidationTrigger.MANUAL
                )
            else:
                # Use standard project invalidation
                event = await invalidation_service.invalidate_project_caches(project_name, invalidation_reason)

            # Get statistics
            stats = invalidation_service.get_invalidation_stats()

            return {
                "success": True,
                "project_name": project_name,
                "event_id": event.event_id,
                "reason": event.reason.value,
                "affected_keys": len(event.affected_keys),
                "timestamp": event.timestamp.isoformat(),
                "invalidation_scope": invalidation_scope,
                "strategy": strategy,
                "policy_applied": policy.project_name != "__default__",
                "statistics": {
                    "total_invalidations": stats.total_invalidations,
                    "partial_invalidations": stats.partial_invalidations,
                    "keys_invalidated": stats.keys_invalidated,
                    "keys_preserved": stats.keys_preserved,
                    "avg_optimization_ratio": stats.avg_optimization_ratio,
                },
                "metadata": event.metadata,
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "manual_invalidate_project_cache",
                {
                    "project_name": project_name,
                    "reason": reason,
                },
            )


async def manual_invalidate_cache_keys(
    cache_keys: list[str],
    reason: str = "manual_invalidation",
    cascade: bool = False,
) -> dict[str, Any]:
    """
    Manually invalidate specific cache keys.

    Args:
        cache_keys: List of cache keys to invalidate
        reason: Reason for invalidation (for logging)
        cascade: Whether to cascade invalidation to dependent caches

    Returns:
        Dictionary with invalidation results and statistics
    """
    with log_tool_usage(
        "manual_invalidate_cache_keys",
        {
            "cache_keys_count": len(cache_keys),
            "reason": reason,
            "cascade": cascade,
        },
    ):
        try:
            from ...services.cache_invalidation_service import (
                InvalidationReason,
                get_cache_invalidation_service,
            )

            if not cache_keys:
                return {
                    "success": False,
                    "error": "No cache keys provided",
                }

            # Get invalidation service
            invalidation_service = await get_cache_invalidation_service()

            # Map reason string to enum
            reason_mapping = {
                "manual_invalidation": InvalidationReason.MANUAL_INVALIDATION,
                "dependency_changed": InvalidationReason.DEPENDENCY_CHANGED,
                "cache_corruption": InvalidationReason.CACHE_CORRUPTION,
                "ttl_expired": InvalidationReason.TTL_EXPIRED,
            }
            invalidation_reason = reason_mapping.get(reason, InvalidationReason.MANUAL_INVALIDATION)

            # Perform invalidation
            event = await invalidation_service.invalidate_keys(cache_keys, invalidation_reason, cascade)

            # Get statistics
            stats = invalidation_service.get_invalidation_stats()

            return {
                "success": True,
                "event_id": event.event_id,
                "reason": event.reason.value,
                "requested_keys": len(cache_keys),
                "affected_keys": len(event.affected_keys),
                "cascade_applied": cascade,
                "timestamp": event.timestamp.isoformat(),
                "statistics": {
                    "total_invalidations": stats.total_invalidations,
                    "keys_invalidated": stats.keys_invalidated,
                    "avg_invalidation_time": stats.avg_invalidation_time,
                },
                "metadata": event.metadata,
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "manual_invalidate_cache_keys",
                {
                    "cache_keys_count": len(cache_keys) if cache_keys else 0,
                    "reason": reason,
                },
            )


async def manual_invalidate_cache_pattern(
    pattern: str,
    reason: str = "manual_invalidation",
) -> dict[str, Any]:
    """
    Manually invalidate cache keys matching a pattern.

    Args:
        pattern: Pattern to match cache keys (supports wildcards)
        reason: Reason for invalidation (for logging)

    Returns:
        Dictionary with invalidation results and statistics
    """
    with log_tool_usage(
        "manual_invalidate_cache_pattern",
        {
            "pattern": pattern,
            "reason": reason,
        },
    ):
        try:
            from ...services.cache_invalidation_service import (
                InvalidationReason,
                get_cache_invalidation_service,
            )

            if not pattern:
                return {
                    "success": False,
                    "error": "No pattern provided",
                }

            # Get invalidation service
            invalidation_service = await get_cache_invalidation_service()

            # Map reason string to enum
            reason_mapping = {
                "manual_invalidation": InvalidationReason.MANUAL_INVALIDATION,
                "dependency_changed": InvalidationReason.DEPENDENCY_CHANGED,
                "cache_corruption": InvalidationReason.CACHE_CORRUPTION,
                "system_upgrade": InvalidationReason.SYSTEM_UPGRADE,
            }
            invalidation_reason = reason_mapping.get(reason, InvalidationReason.MANUAL_INVALIDATION)

            # Perform pattern-based invalidation
            event = await invalidation_service.invalidate_pattern(pattern, invalidation_reason)

            # Get statistics
            stats = invalidation_service.get_invalidation_stats()

            return {
                "success": True,
                "pattern": pattern,
                "event_id": event.event_id,
                "reason": event.reason.value,
                "affected_keys": len(event.affected_keys),
                "timestamp": event.timestamp.isoformat(),
                "statistics": {
                    "total_invalidations": stats.total_invalidations,
                    "keys_invalidated": stats.keys_invalidated,
                    "avg_invalidation_time": stats.avg_invalidation_time,
                },
                "metadata": event.metadata,
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "manual_invalidate_cache_pattern",
                {
                    "pattern": pattern,
                    "reason": reason,
                },
            )


async def clear_all_caches(
    reason: str = "manual_invalidation",
    confirm: bool = False,
) -> dict[str, Any]:
    """
    Clear all caches across all services (DESTRUCTIVE OPERATION).

    Args:
        reason: Reason for clearing all caches
        confirm: Must be True to confirm this destructive operation

    Returns:
        Dictionary with clearing results and statistics
    """
    with log_tool_usage(
        "clear_all_caches",
        {
            "reason": reason,
            "confirm": confirm,
        },
    ):
        try:
            if not confirm:
                return {
                    "success": False,
                    "error": "Operation requires confirmation. Set confirm=True to proceed.",
                    "warning": "This operation will clear ALL cache data across all services.",
                }

            from ...services.cache_invalidation_service import (
                InvalidationReason,
                get_cache_invalidation_service,
            )

            # Get invalidation service
            invalidation_service = await get_cache_invalidation_service()

            # Map reason string to enum
            reason_mapping = {
                "manual_invalidation": InvalidationReason.MANUAL_INVALIDATION,
                "system_upgrade": InvalidationReason.SYSTEM_UPGRADE,
                "cache_corruption": InvalidationReason.CACHE_CORRUPTION,
            }
            invalidation_reason = reason_mapping.get(reason, InvalidationReason.MANUAL_INVALIDATION)

            # Perform complete cache clearing
            event = await invalidation_service.clear_all_caches(invalidation_reason)

            # Get statistics
            stats = invalidation_service.get_invalidation_stats()

            return {
                "success": True,
                "operation": "clear_all_caches",
                "event_id": event.event_id,
                "reason": event.reason.value,
                "timestamp": event.timestamp.isoformat(),
                "warning": "All cache data has been cleared",
                "statistics": {
                    "total_invalidations": stats.total_invalidations,
                    "avg_invalidation_time": stats.avg_invalidation_time,
                },
                "metadata": event.metadata,
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "clear_all_caches",
                {
                    "reason": reason,
                    "confirm": confirm,
                },
            )


# ===== CACHE INSPECTION AND DEBUGGING TOOLS =====


async def inspect_cache_state(
    cache_type: str = "all",
    include_content: bool = False,
    max_entries: int = 100,
) -> dict[str, Any]:
    """
    Inspect the current state of cache services with detailed debugging information.

    Args:
        cache_type: Type of cache to inspect (all, embedding, search, project, file, l1, l2)
        include_content: Whether to include actual cache content in response
        max_entries: Maximum number of cache entries to include

    Returns:
        Dictionary with detailed cache state information
    """
    with log_tool_usage(
        "inspect_cache_state",
        {
            "cache_type": cache_type,
            "include_content": include_content,
            "max_entries": max_entries,
        },
    ):
        try:
            from ...services.cache_service import get_cache_service

            # Get all cache services
            cache_services = {}

            if cache_type in ["all", "embedding"]:
                try:
                    from ...services.embedding_cache_service import get_embedding_cache_service

                    cache_services["embedding"] = await get_embedding_cache_service()
                except Exception as e:
                    logger.warning(f"Could not get embedding cache service: {e}")

            if cache_type in ["all", "search"]:
                try:
                    from ...services.search_cache_service import get_search_cache_service

                    cache_services["search"] = await get_search_cache_service()
                except Exception as e:
                    logger.warning(f"Could not get search cache service: {e}")

            if cache_type in ["all", "project"]:
                try:
                    from ...services.project_cache_service import get_project_cache_service

                    cache_services["project"] = await get_project_cache_service()
                except Exception as e:
                    logger.warning(f"Could not get project cache service: {e}")

            if cache_type in ["all", "file"]:
                try:
                    from ...services.file_cache_service import get_file_cache_service

                    cache_services["file"] = await get_file_cache_service()
                except Exception as e:
                    logger.warning(f"Could not get file cache service: {e}")

            # Get base cache service for L1/L2 inspection
            if cache_type in ["all", "l1", "l2"]:
                try:
                    cache_services["base"] = await get_cache_service()
                except Exception as e:
                    logger.warning(f"Could not get base cache service: {e}")

            # Inspect each cache service
            inspection_results = {}

            for service_name, service in cache_services.items():
                if not service:
                    continue

                try:
                    # Get service health status
                    health_status = await service.health_check() if hasattr(service, "health_check") else {"status": "unknown"}

                    # Get cache statistics
                    stats = service.get_stats() if hasattr(service, "get_stats") else {}

                    # Get cache keys (limited)
                    keys = []
                    content = {}

                    if hasattr(service, "get_all_keys"):
                        all_keys = await service.get_all_keys()
                        keys = list(all_keys)[:max_entries]

                        if include_content:
                            for key in keys[:10]:  # Limit content inspection to 10 keys
                                try:
                                    value = await service.get(key)
                                    content[key] = {
                                        "type": type(value).__name__,
                                        "size": len(str(value)) if value else 0,
                                        "content": str(value)[:500] if value else None,  # Truncate content
                                    }
                                except Exception as e:
                                    content[key] = {"error": str(e)}

                    # Get L1/L2 specific information
                    layer_info = {}
                    if service_name == "base" and hasattr(service, "_l1_cache"):
                        if cache_type in ["all", "l1"]:
                            layer_info["l1"] = {
                                "size": len(service._l1_cache) if service._l1_cache else 0,
                                "max_size": getattr(service._l1_cache, "maxsize", None) if service._l1_cache else None,
                                "keys": list(service._l1_cache.keys())[:max_entries] if service._l1_cache else [],
                            }
                        if cache_type in ["all", "l2"] and hasattr(service, "_redis_pool"):
                            try:
                                import redis.asyncio as redis

                                redis_client = redis.Redis(connection_pool=service._redis_pool)
                                redis_info = await redis_client.info()
                                layer_info["l2"] = {
                                    "connected_clients": redis_info.get("connected_clients", 0),
                                    "used_memory": redis_info.get("used_memory_human", "unknown"),
                                    "keyspace_hits": redis_info.get("keyspace_hits", 0),
                                    "keyspace_misses": redis_info.get("keyspace_misses", 0),
                                }
                            except Exception as e:
                                layer_info["l2"] = {"error": str(e)}

                    inspection_results[service_name] = {
                        "health": health_status,
                        "statistics": stats,
                        "keys_count": len(keys),
                        "sample_keys": keys[:20],  # Show first 20 keys
                        "layer_info": layer_info,
                        "content": content if include_content else None,
                    }

                except Exception as e:
                    inspection_results[service_name] = {
                        "error": str(e),
                        "health": {"status": "error", "error": str(e)},
                    }

            return {
                "success": True,
                "cache_type": cache_type,
                "inspection_results": inspection_results,
                "summary": {
                    "services_inspected": len(inspection_results),
                    "total_keys": sum(
                        result.get("keys_count", 0)
                        for result in inspection_results.values()
                        if isinstance(result, dict) and "keys_count" in result
                    ),
                    "healthy_services": sum(
                        1
                        for result in inspection_results.values()
                        if isinstance(result, dict) and result.get("health", {}).get("status") != "error"
                    ),
                },
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "inspect_cache_state",
                {
                    "cache_type": cache_type,
                    "include_content": include_content,
                },
            )


async def debug_cache_key(
    cache_key: str,
    cache_type: str = "all",
) -> dict[str, Any]:
    """
    Debug a specific cache key across all cache services.

    Args:
        cache_key: The cache key to debug
        cache_type: Type of cache to check (all, embedding, search, project, file)

    Returns:
        Dictionary with debugging information for the cache key
    """
    with log_tool_usage(
        "debug_cache_key",
        {
            "cache_key": cache_key,
            "cache_type": cache_type,
        },
    ):
        try:
            from ...services.cache_service import get_cache_service

            # Get all cache services
            cache_services = {}

            if cache_type in ["all", "embedding"]:
                try:
                    from ...services.embedding_cache_service import get_embedding_cache_service

                    cache_services["embedding"] = await get_embedding_cache_service()
                except Exception as e:
                    logger.warning(f"Could not get embedding cache service: {e}")

            if cache_type in ["all", "search"]:
                try:
                    from ...services.search_cache_service import get_search_cache_service

                    cache_services["search"] = await get_search_cache_service()
                except Exception as e:
                    logger.warning(f"Could not get search cache service: {e}")

            if cache_type in ["all", "project"]:
                try:
                    from ...services.project_cache_service import get_project_cache_service

                    cache_services["project"] = await get_project_cache_service()
                except Exception as e:
                    logger.warning(f"Could not get project cache service: {e}")

            if cache_type in ["all", "file"]:
                try:
                    from ...services.file_cache_service import get_file_cache_service

                    cache_services["file"] = await get_file_cache_service()
                except Exception as e:
                    logger.warning(f"Could not get file cache service: {e}")

            # Debug key in each service
            debug_results = {}

            for service_name, service in cache_services.items():
                if not service:
                    continue

                try:
                    # Check if key exists
                    exists = await service.exists(cache_key) if hasattr(service, "exists") else None

                    # Get value if exists
                    value = None
                    value_info = {}
                    if exists:
                        try:
                            value = await service.get(cache_key)
                            value_info = {
                                "type": type(value).__name__,
                                "size_bytes": len(str(value)) if value else 0,
                                "content_preview": str(value)[:200] if value else None,
                                "is_none": value is None,
                            }
                        except Exception as e:
                            value_info = {"error": str(e)}

                    # Get TTL if supported
                    ttl = None
                    if hasattr(service, "get_ttl"):
                        try:
                            ttl = await service.get_ttl(cache_key)
                        except Exception:
                            pass

                    # Get metadata if available
                    metadata = {}
                    if hasattr(service, "get_metadata"):
                        try:
                            metadata = await service.get_metadata(cache_key)
                        except Exception:
                            pass

                    debug_results[service_name] = {
                        "exists": exists,
                        "value_info": value_info,
                        "ttl_seconds": ttl,
                        "metadata": metadata,
                    }

                except Exception as e:
                    debug_results[service_name] = {
                        "error": str(e),
                    }

            return {
                "success": True,
                "cache_key": cache_key,
                "cache_type": cache_type,
                "debug_results": debug_results,
                "summary": {
                    "services_checked": len(debug_results),
                    "found_in_services": [
                        service for service, result in debug_results.items() if isinstance(result, dict) and result.get("exists")
                    ],
                    "errors": [service for service, result in debug_results.items() if isinstance(result, dict) and "error" in result],
                },
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "debug_cache_key",
                {
                    "cache_key": cache_key,
                    "cache_type": cache_type,
                },
            )


async def get_cache_invalidation_stats() -> dict[str, Any]:
    """
    Get comprehensive cache invalidation statistics and metrics.

    Returns:
        Dictionary with detailed invalidation statistics
    """
    with log_tool_usage("get_cache_invalidation_stats", {}):
        try:
            from ...services.cache_invalidation_service import get_cache_invalidation_service

            # Get invalidation service
            invalidation_service = await get_cache_invalidation_service()

            # Get current statistics
            stats = invalidation_service.get_invalidation_stats()

            # Get recent events
            recent_events = invalidation_service.get_recent_events(10)

            # Get monitored projects
            monitored_projects = invalidation_service.get_monitored_projects()

            return {
                "success": True,
                "statistics": {
                    "total_invalidations": stats.total_invalidations,
                    "file_based_invalidations": stats.file_based_invalidations,
                    "manual_invalidations": stats.manual_invalidations,
                    "cascade_invalidations": stats.cascade_invalidations,
                    "partial_invalidations": stats.partial_invalidations,
                    "keys_invalidated": stats.keys_invalidated,
                    "keys_preserved": stats.keys_preserved,
                    "avg_invalidation_time": stats.avg_invalidation_time,
                    "avg_optimization_ratio": stats.avg_optimization_ratio,
                    "last_invalidation": stats.last_invalidation.isoformat() if stats.last_invalidation else None,
                },
                "recent_events": [
                    {
                        "event_id": event.event_id,
                        "reason": event.reason.value,
                        "timestamp": event.timestamp.isoformat(),
                        "affected_keys_count": len(event.affected_keys),
                        "affected_files": event.affected_files,
                        "project_name": event.project_name,
                        "optimization_ratio": (
                            getattr(event.partial_result, "optimization_ratio", None)
                            if hasattr(event, "partial_result") and event.partial_result
                            else None
                        ),
                    }
                    for event in recent_events
                ],
                "monitoring": {
                    "monitored_projects": monitored_projects,
                    "projects_count": len(monitored_projects),
                },
            }

        except Exception as e:
            return handle_tool_error(e, "get_cache_invalidation_stats", {})


async def get_project_invalidation_policy(
    project_name: str,
) -> dict[str, Any]:
    """
    Get invalidation policy for a specific project.

    Args:
        project_name: Name of the project

    Returns:
        Dictionary with project invalidation policy details
    """
    with log_tool_usage(
        "get_project_invalidation_policy",
        {
            "project_name": project_name,
        },
    ):
        try:
            from ...services.cache_invalidation_service import get_cache_invalidation_service

            # Get invalidation service
            invalidation_service = await get_cache_invalidation_service()

            # Get policy summary
            policy_summary = invalidation_service.get_project_policy_summary(project_name)

            # Get project files if monitored
            project_files = invalidation_service.get_project_files(project_name)

            return {
                "success": True,
                "project_name": project_name,
                "policy": policy_summary,
                "monitoring": {
                    "is_monitored": bool(project_files),
                    "file_count": len(project_files),
                    "sample_files": list(project_files)[:10] if project_files else [],
                },
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "get_project_invalidation_policy",
                {
                    "project_name": project_name,
                },
            )


async def set_project_invalidation_policy(
    project_name: str,
    scope: str = "cascade",
    strategy: str = "immediate",
    batch_threshold: int = 5,
    delay_seconds: float = 0.0,
    file_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    invalidate_embeddings: bool = True,
    invalidate_search: bool = True,
    invalidate_project: bool = True,
    invalidate_file: bool = True,
    max_concurrent_invalidations: int = 10,
    cascade_depth_limit: int = 3,
) -> dict[str, Any]:
    """
    Set or update invalidation policy for a specific project.

    Args:
        project_name: Name of the project
        scope: Invalidation scope (file_only, project_wide, cascade, conservative, aggressive)
        strategy: Invalidation strategy (immediate, lazy, batch, scheduled)
        batch_threshold: Number of changes to trigger batch processing
        delay_seconds: Delay before processing invalidation
        file_patterns: File patterns to monitor
        exclude_patterns: Patterns to exclude from monitoring
        invalidate_embeddings: Whether to invalidate embedding caches
        invalidate_search: Whether to invalidate search caches
        invalidate_project: Whether to invalidate project caches
        invalidate_file: Whether to invalidate file caches
        max_concurrent_invalidations: Maximum concurrent invalidations
        cascade_depth_limit: Maximum cascade depth

    Returns:
        Dictionary with policy creation/update results
    """
    with log_tool_usage(
        "set_project_invalidation_policy",
        {
            "project_name": project_name,
            "scope": scope,
            "strategy": strategy,
        },
    ):
        try:
            from ...services.cache_invalidation_service import (
                InvalidationStrategy,
                ProjectInvalidationPolicy,
                ProjectInvalidationScope,
                get_cache_invalidation_service,
            )

            # Get invalidation service
            invalidation_service = await get_cache_invalidation_service()

            # Map scope string to enum
            scope_mapping = {
                "file_only": ProjectInvalidationScope.FILE_ONLY,
                "project_wide": ProjectInvalidationScope.PROJECT_WIDE,
                "cascade": ProjectInvalidationScope.CASCADE,
                "conservative": ProjectInvalidationScope.CONSERVATIVE,
                "aggressive": ProjectInvalidationScope.AGGRESSIVE,
            }
            scope_enum = scope_mapping.get(scope, ProjectInvalidationScope.CASCADE)

            # Map strategy string to enum
            strategy_mapping = {
                "immediate": InvalidationStrategy.IMMEDIATE,
                "lazy": InvalidationStrategy.LAZY,
                "batch": InvalidationStrategy.BATCH,
                "scheduled": InvalidationStrategy.SCHEDULED,
            }
            strategy_enum = strategy_mapping.get(strategy, InvalidationStrategy.IMMEDIATE)

            # Set default patterns if not provided
            if file_patterns is None:
                file_patterns = ["*.py", "*.js", "*.ts", "*.java", "*.cpp", "*.c", "*.h", "*.hpp"]

            if exclude_patterns is None:
                exclude_patterns = ["*.pyc", "*.log", "*.tmp", "__pycache__/*", ".git/*", "node_modules/*"]

            # Create project policy
            invalidation_service.create_project_policy(
                project_name=project_name,
                scope=scope_enum,
                strategy=strategy_enum,
                batch_threshold=batch_threshold,
                delay_seconds=delay_seconds,
                file_patterns=file_patterns,
                exclude_patterns=exclude_patterns,
                max_concurrent_invalidations=max_concurrent_invalidations,
                cascade_depth_limit=cascade_depth_limit,
                invalidate_embeddings=invalidate_embeddings,
                invalidate_search=invalidate_search,
                invalidate_project=invalidate_project,
                invalidate_file=invalidate_file,
            )

            # Get policy summary
            policy_summary = invalidation_service.get_project_policy_summary(project_name)

            return {
                "success": True,
                "project_name": project_name,
                "policy_created": True,
                "policy": policy_summary,
                "message": f"Invalidation policy set for project '{project_name}'",
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "set_project_invalidation_policy",
                {
                    "project_name": project_name,
                    "scope": scope,
                    "strategy": strategy,
                },
            )


# ===== CACHE WARMING AND PRELOADING TOOLS =====


async def warm_cache_for_project(
    project_name: str,
    cache_types: list[str] = None,
    warmup_strategy: str = "comprehensive",
    max_concurrent: int = 5,
) -> dict[str, Any]:
    """
    Warm up caches for a specific project with comprehensive preloading.

    Args:
        project_name: Name of the project to warm up
        cache_types: Types of caches to warm (embedding, search, project, file, all)
        warmup_strategy: Warmup strategy (comprehensive, selective, recent, critical)
        max_concurrent: Maximum concurrent warming operations

    Returns:
        Dictionary with cache warming results and statistics
    """
    with log_tool_usage(
        "warm_cache_for_project",
        {
            "project_name": project_name,
            "cache_types": cache_types,
            "warmup_strategy": warmup_strategy,
            "max_concurrent": max_concurrent,
        },
    ):
        try:
            from ...services.cache_warmup_service import get_cache_warmup_service

            # Get cache warmup service
            warmup_service = await get_cache_warmup_service()

            # Set default cache types
            if cache_types is None:
                cache_types = ["all"]

            # Map strategy string to enum if needed
            strategy_mapping = {
                "comprehensive": "COMPREHENSIVE",
                "selective": "SELECTIVE",
                "recent": "RECENT_FILES",
                "critical": "CRITICAL_PATHS",
            }

            # Start warming process
            warming_results = {}
            total_warmed = 0

            for cache_type in cache_types:
                try:
                    if cache_type == "all":
                        # Warm all cache types
                        result = await warmup_service.warm_project_caches(
                            project_name, strategy=strategy_mapping.get(warmup_strategy, "COMPREHENSIVE"), max_concurrent=max_concurrent
                        )
                    else:
                        # Warm specific cache type
                        result = await warmup_service.warm_cache_type(
                            project_name,
                            cache_type,
                            strategy=strategy_mapping.get(warmup_strategy, "COMPREHENSIVE"),
                            max_concurrent=max_concurrent,
                        )

                    warming_results[cache_type] = {
                        "success": True,
                        "warmed_entries": result.get("warmed_entries", 0),
                        "cache_hits": result.get("cache_hits", 0),
                        "duration_seconds": result.get("duration_seconds", 0),
                        "strategy_used": result.get("strategy_used", warmup_strategy),
                    }
                    total_warmed += result.get("warmed_entries", 0)

                except Exception as e:
                    warming_results[cache_type] = {
                        "success": False,
                        "error": str(e),
                    }

            return {
                "success": True,
                "project_name": project_name,
                "warmup_strategy": warmup_strategy,
                "cache_types": cache_types,
                "results": warming_results,
                "summary": {
                    "total_warmed_entries": total_warmed,
                    "successful_types": len([r for r in warming_results.values() if r.get("success")]),
                    "failed_types": len([r for r in warming_results.values() if not r.get("success")]),
                },
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "warm_cache_for_project",
                {
                    "project_name": project_name,
                    "cache_types": cache_types,
                    "warmup_strategy": warmup_strategy,
                },
            )


async def preload_embedding_cache(
    queries: list[str],
    project_name: str = None,
    model_name: str = None,
) -> dict[str, Any]:
    """
    Preload embedding cache with specific queries or content.

    Args:
        queries: List of queries/content to preload embeddings for
        project_name: Optional project name for scoped preloading
        model_name: Optional specific embedding model name

    Returns:
        Dictionary with preloading results and statistics
    """
    with log_tool_usage(
        "preload_embedding_cache",
        {
            "queries_count": len(queries),
            "project_name": project_name,
            "model_name": model_name,
        },
    ):
        try:
            from ...services.embedding_cache_service import get_embedding_cache_service
            from ...services.embedding_service import get_embedding_service

            # Get services
            cache_service = await get_embedding_cache_service()
            embedding_service = await get_embedding_service()

            if not queries:
                return {
                    "success": False,
                    "error": "No queries provided for preloading",
                }

            preloading_results = []
            cache_hits = 0
            cache_misses = 0

            for query in queries:
                try:
                    # Check if already cached
                    cached_embedding = await cache_service.get_cached_embedding(query, model_name)

                    if cached_embedding is not None:
                        cache_hits += 1
                        preloading_results.append(
                            {
                                "query": query[:100],  # Truncate for logging
                                "status": "cache_hit",
                                "already_cached": True,
                            }
                        )
                    else:
                        # Generate and cache embedding
                        embeddings = await embedding_service.generate_embeddings([query], model_name)
                        if embeddings and len(embeddings) > 0:
                            await cache_service.cache_embedding(query, embeddings[0], model_name, project_name)
                            cache_misses += 1
                            preloading_results.append(
                                {
                                    "query": query[:100],
                                    "status": "preloaded",
                                    "embedding_dimensions": len(embeddings[0]) if embeddings[0] else 0,
                                }
                            )
                        else:
                            preloading_results.append(
                                {
                                    "query": query[:100],
                                    "status": "failed",
                                    "error": "No embeddings generated",
                                }
                            )

                except Exception as e:
                    preloading_results.append(
                        {
                            "query": query[:100],
                            "status": "error",
                            "error": str(e),
                        }
                    )

            return {
                "success": True,
                "project_name": project_name,
                "model_name": model_name,
                "queries_processed": len(queries),
                "results": preloading_results,
                "summary": {
                    "cache_hits": cache_hits,
                    "newly_cached": cache_misses,
                    "total_processed": len(queries),
                    "success_rate": (cache_hits + cache_misses) / len(queries) if queries else 0,
                },
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "preload_embedding_cache",
                {
                    "queries_count": len(queries) if queries else 0,
                    "project_name": project_name,
                },
            )


async def preload_search_cache(
    search_queries: list[dict],
    project_name: str = None,
) -> dict[str, Any]:
    """
    Preload search result cache with specific search queries.

    Args:
        search_queries: List of search query dictionaries with parameters
        project_name: Optional project name for scoped preloading

    Returns:
        Dictionary with search cache preloading results
    """
    with log_tool_usage(
        "preload_search_cache",
        {
            "queries_count": len(search_queries),
            "project_name": project_name,
        },
    ):
        try:
            from ...services.search_cache_service import get_search_cache_service
            from ...tools.indexing.search_tools import search as perform_search

            # Get search cache service
            cache_service = await get_search_cache_service()

            if not search_queries:
                return {
                    "success": False,
                    "error": "No search queries provided for preloading",
                }

            preloading_results = []
            cache_hits = 0
            cache_misses = 0

            for query_params in search_queries:
                try:
                    # Extract query parameters
                    query = query_params.get("query", "")
                    n_results = query_params.get("n_results", 5)
                    search_mode = query_params.get("search_mode", "hybrid")

                    if not query:
                        preloading_results.append(
                            {
                                "query_params": query_params,
                                "status": "error",
                                "error": "No query provided",
                            }
                        )
                        continue

                    # Check if already cached
                    cached_results = await cache_service.get_cached_search_results(query, n_results, search_mode, project_name)

                    if cached_results is not None:
                        cache_hits += 1
                        preloading_results.append(
                            {
                                "query": query[:100],
                                "status": "cache_hit",
                                "already_cached": True,
                                "results_count": len(cached_results.get("results", [])),
                            }
                        )
                    else:
                        # Perform search and cache results
                        search_results = await perform_search(
                            query=query,
                            n_results=n_results,
                            search_mode=search_mode,
                            target_projects=[project_name] if project_name else None,
                        )

                        if search_results.get("success"):
                            await cache_service.cache_search_results(query, search_results, search_mode, project_name)
                            cache_misses += 1
                            preloading_results.append(
                                {
                                    "query": query[:100],
                                    "status": "preloaded",
                                    "results_count": len(search_results.get("results", [])),
                                }
                            )
                        else:
                            preloading_results.append(
                                {
                                    "query": query[:100],
                                    "status": "failed",
                                    "error": search_results.get("error", "Search failed"),
                                }
                            )

                except Exception as e:
                    preloading_results.append(
                        {
                            "query_params": query_params,
                            "status": "error",
                            "error": str(e),
                        }
                    )

            return {
                "success": True,
                "project_name": project_name,
                "queries_processed": len(search_queries),
                "results": preloading_results,
                "summary": {
                    "cache_hits": cache_hits,
                    "newly_cached": cache_misses,
                    "total_processed": len(search_queries),
                    "success_rate": (cache_hits + cache_misses) / len(search_queries) if search_queries else 0,
                },
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "preload_search_cache",
                {
                    "queries_count": len(search_queries) if search_queries else 0,
                    "project_name": project_name,
                },
            )


# ===== CACHE STATISTICS AND REPORTING TOOLS =====


async def get_comprehensive_cache_stats(
    project_name: str = None,
    include_historical: bool = False,
    time_range_hours: int = 24,
) -> dict[str, Any]:
    """
    Get comprehensive cache statistics across all cache services.

    Args:
        project_name: Optional project name for scoped statistics
        include_historical: Whether to include historical data
        time_range_hours: Time range for historical data (hours)

    Returns:
        Dictionary with comprehensive cache statistics
    """
    with log_tool_usage(
        "get_comprehensive_cache_stats",
        {
            "project_name": project_name,
            "include_historical": include_historical,
            "time_range_hours": time_range_hours,
        },
    ):
        try:
            # Get all cache services
            cache_stats = {}

            # Embedding cache statistics
            try:
                from ...services.embedding_cache_service import get_embedding_cache_service

                embedding_service = await get_embedding_cache_service()
                if hasattr(embedding_service, "get_cache_stats"):
                    cache_stats["embedding"] = await embedding_service.get_cache_stats(project_name)
                elif hasattr(embedding_service, "get_stats"):
                    cache_stats["embedding"] = embedding_service.get_stats()
            except Exception as e:
                cache_stats["embedding"] = {"error": str(e)}

            # Search cache statistics
            try:
                from ...services.search_cache_service import get_search_cache_service

                search_service = await get_search_cache_service()
                if hasattr(search_service, "get_cache_stats"):
                    cache_stats["search"] = await search_service.get_cache_stats(project_name)
                elif hasattr(search_service, "get_stats"):
                    cache_stats["search"] = search_service.get_stats()
            except Exception as e:
                cache_stats["search"] = {"error": str(e)}

            # Project cache statistics
            try:
                from ...services.project_cache_service import get_project_cache_service

                project_service = await get_project_cache_service()
                if hasattr(project_service, "get_cache_stats"):
                    cache_stats["project"] = await project_service.get_cache_stats(project_name)
                elif hasattr(project_service, "get_stats"):
                    cache_stats["project"] = project_service.get_stats()
            except Exception as e:
                cache_stats["project"] = {"error": str(e)}

            # File cache statistics
            try:
                from ...services.file_cache_service import get_file_cache_service

                file_service = await get_file_cache_service()
                if hasattr(file_service, "get_cache_stats"):
                    cache_stats["file"] = await file_service.get_cache_stats(project_name)
                elif hasattr(file_service, "get_stats"):
                    cache_stats["file"] = file_service.get_stats()
            except Exception as e:
                cache_stats["file"] = {"error": str(e)}

            # Cache invalidation statistics
            try:
                invalidation_stats = await get_cache_invalidation_stats()
                cache_stats["invalidation"] = invalidation_stats.get("statistics", {})
            except Exception as e:
                cache_stats["invalidation"] = {"error": str(e)}

            # Memory profiling statistics
            try:
                from ...services.cache_memory_profiler import get_cache_memory_profiler

                memory_profiler = await get_cache_memory_profiler()
                if hasattr(memory_profiler, "get_memory_stats"):
                    cache_stats["memory"] = await memory_profiler.get_memory_stats()
            except Exception as e:
                cache_stats["memory"] = {"error": str(e)}

            # Calculate aggregated statistics
            total_cache_hits = 0
            total_cache_misses = 0
            total_cache_size = 0
            total_keys = 0
            healthy_services = 0

            for service_name, stats in cache_stats.items():
                if isinstance(stats, dict) and "error" not in stats:
                    healthy_services += 1
                    total_cache_hits += stats.get("cache_hits", 0)
                    total_cache_misses += stats.get("cache_misses", 0)
                    total_cache_size += stats.get("cache_size_bytes", 0)
                    total_keys += stats.get("total_keys", 0)

            # Calculate derived metrics
            total_requests = total_cache_hits + total_cache_misses
            hit_rate = (total_cache_hits / total_requests) if total_requests > 0 else 0

            # Historical data if requested
            historical_data = {}
            if include_historical:
                try:
                    from ...utils.performance_monitor import get_performance_monitor

                    perf_monitor = get_performance_monitor()
                    historical_data = perf_monitor.get_cache_metrics_history(hours=time_range_hours, project_name=project_name)
                except Exception as e:
                    historical_data = {"error": str(e)}

            return {
                "success": True,
                "project_name": project_name,
                "time_range_hours": time_range_hours if include_historical else None,
                "cache_statistics": cache_stats,
                "aggregated_stats": {
                    "total_cache_hits": total_cache_hits,
                    "total_cache_misses": total_cache_misses,
                    "total_requests": total_requests,
                    "hit_rate": hit_rate,
                    "miss_rate": 1 - hit_rate,
                    "total_cache_size_bytes": total_cache_size,
                    "total_cache_size_mb": total_cache_size / (1024 * 1024),
                    "total_keys": total_keys,
                    "healthy_services": healthy_services,
                    "total_services": len(cache_stats),
                },
                "historical_data": historical_data if include_historical else None,
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "get_comprehensive_cache_stats",
                {
                    "project_name": project_name,
                    "include_historical": include_historical,
                },
            )


async def generate_cache_report(
    report_type: str = "comprehensive",
    project_name: str = None,
    export_format: str = "json",
) -> dict[str, Any]:
    """
    Generate a comprehensive cache performance report.

    Args:
        report_type: Type of report (comprehensive, performance, health, optimization)
        project_name: Optional project name for scoped reporting
        export_format: Format for export (json, markdown, csv)

    Returns:
        Dictionary with cache report data
    """
    with log_tool_usage(
        "generate_cache_report",
        {
            "report_type": report_type,
            "project_name": project_name,
            "export_format": export_format,
        },
    ):
        try:
            # Get comprehensive statistics
            stats_result = await get_comprehensive_cache_stats(project_name=project_name, include_historical=True, time_range_hours=24)

            if not stats_result.get("success"):
                return stats_result

            cache_stats = stats_result["cache_statistics"]
            aggregated_stats = stats_result["aggregated_stats"]

            # Generate report based on type
            report_data = {
                "report_metadata": {
                    "report_type": report_type,
                    "project_name": project_name,
                    "generated_at": time.time(),
                    "export_format": export_format,
                },
                "executive_summary": {
                    "overall_hit_rate": aggregated_stats["hit_rate"],
                    "total_cache_size_mb": aggregated_stats["total_cache_size_mb"],
                    "healthy_services": aggregated_stats["healthy_services"],
                    "total_services": aggregated_stats["total_services"],
                    "performance_score": min(100, aggregated_stats["hit_rate"] * 100),
                },
            }

            if report_type in ["comprehensive", "performance"]:
                report_data["performance_metrics"] = {
                    "cache_hit_rate_by_service": {
                        service: (stats.get("cache_hits", 0) / max(1, stats.get("cache_hits", 0) + stats.get("cache_misses", 0)))
                        for service, stats in cache_stats.items()
                        if isinstance(stats, dict) and "error" not in stats
                    },
                    "response_times": {
                        service: stats.get("avg_response_time_ms", 0)
                        for service, stats in cache_stats.items()
                        if isinstance(stats, dict) and "error" not in stats
                    },
                    "memory_usage": {
                        service: stats.get("memory_usage_bytes", 0)
                        for service, stats in cache_stats.items()
                        if isinstance(stats, dict) and "error" not in stats
                    },
                }

            if report_type in ["comprehensive", "health"]:
                report_data["health_status"] = {
                    "service_health": {
                        service: "healthy" if isinstance(stats, dict) and "error" not in stats else "error"
                        for service, stats in cache_stats.items()
                    },
                    "error_summary": {
                        service: stats.get("error", "No error")
                        for service, stats in cache_stats.items()
                        if isinstance(stats, dict) and "error" in stats
                    },
                }

            if report_type in ["comprehensive", "optimization"]:
                # Generate optimization recommendations
                recommendations = []

                if aggregated_stats["hit_rate"] < 0.7:
                    recommendations.append(
                        {
                            "priority": "high",
                            "category": "performance",
                            "recommendation": "Cache hit rate is below 70%. Consider cache warming or TTL optimization.",
                            "estimated_impact": "high",
                        }
                    )

                if aggregated_stats["total_cache_size_mb"] > 1000:
                    recommendations.append(
                        {
                            "priority": "medium",
                            "category": "memory",
                            "recommendation": "Cache size exceeds 1GB. Consider implementing cache eviction policies.",
                            "estimated_impact": "medium",
                        }
                    )

                if aggregated_stats["healthy_services"] < aggregated_stats["total_services"]:
                    recommendations.append(
                        {
                            "priority": "high",
                            "category": "reliability",
                            "recommendation": "Some cache services are unhealthy. Check error logs and connectivity.",
                            "estimated_impact": "high",
                        }
                    )

                report_data["optimization_recommendations"] = recommendations

            # Format report based on export format
            if export_format == "markdown":
                report_data["formatted_report"] = _format_report_as_markdown(report_data)
            elif export_format == "csv":
                report_data["csv_data"] = _format_report_as_csv(report_data)

            return {
                "success": True,
                "report": report_data,
                "summary": {
                    "report_type": report_type,
                    "services_analyzed": len(cache_stats),
                    "recommendations_count": len(report_data.get("optimization_recommendations", [])),
                    "overall_health": (
                        "healthy" if aggregated_stats["healthy_services"] == aggregated_stats["total_services"] else "degraded"
                    ),
                },
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "generate_cache_report",
                {
                    "report_type": report_type,
                    "project_name": project_name,
                },
            )


def _format_report_as_markdown(report_data: dict) -> str:
    """Format cache report as markdown."""
    md_lines = [
        "# Cache Performance Report",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Report Type:** {report_data['report_metadata']['report_type']}",
        "",
        "## Executive Summary",
        "",
        f"- **Overall Hit Rate:** {report_data['executive_summary']['overall_hit_rate']:.1%}",
        f"- **Total Cache Size:** {report_data['executive_summary']['total_cache_size_mb']:.1f} MB",
        f"- **Performance Score:** {report_data['executive_summary']['performance_score']:.1f}/100",
        "",
    ]

    if "optimization_recommendations" in report_data:
        md_lines.extend(
            [
                "## Optimization Recommendations",
                "",
            ]
        )
        for rec in report_data["optimization_recommendations"]:
            md_lines.extend(
                [
                    f"### {rec['priority'].title()} Priority - {rec['category'].title()}",
                    f"{rec['recommendation']}",
                    f"*Estimated Impact: {rec['estimated_impact']}*",
                    "",
                ]
            )

    return "\n".join(md_lines)


def _format_report_as_csv(report_data: dict) -> str:
    """Format cache report as CSV."""
    lines = [
        "metric,service,value,unit",
    ]

    if "performance_metrics" in report_data:
        for service, hit_rate in report_data["performance_metrics"]["cache_hit_rate_by_service"].items():
            lines.append(f"hit_rate,{service},{hit_rate:.3f},ratio")

        for service, response_time in report_data["performance_metrics"]["response_times"].items():
            lines.append(f"response_time,{service},{response_time},ms")

    return "\n".join(lines)


async def invalidate_chunks(
    file_path: str,
    chunk_ids: list[str],
    reason: str = "chunk_modified",
) -> dict[str, Any]:
    """
    Invalidate specific chunks within a file.

    Args:
        file_path: Path to the file containing the chunks
        chunk_ids: List of chunk IDs to invalidate
        reason: Reason for chunk invalidation

    Returns:
        Dictionary with chunk invalidation results
    """
    with log_tool_usage(
        "invalidate_chunks",
        {
            "file_path": file_path,
            "chunk_ids_count": len(chunk_ids),
            "reason": reason,
        },
    ):
        try:
            from ...services.cache_invalidation_service import (
                InvalidationReason,
                get_cache_invalidation_service,
            )

            if not chunk_ids:
                return {
                    "success": False,
                    "error": "No chunk IDs provided",
                }

            # Validate file path
            abs_path = str(Path(file_path).resolve())

            # Get invalidation service
            invalidation_service = await get_cache_invalidation_service()

            # Map reason string to enum
            reason_mapping = {
                "chunk_modified": InvalidationReason.CHUNK_MODIFIED,
                "manual_invalidation": InvalidationReason.MANUAL_INVALIDATION,
                "content_changed": InvalidationReason.PARTIAL_CONTENT_CHANGE,
            }
            invalidation_reason = reason_mapping.get(reason, InvalidationReason.CHUNK_MODIFIED)

            # Perform chunk invalidation
            event = await invalidation_service.invalidate_specific_chunks(abs_path, chunk_ids, invalidation_reason)

            # Get statistics
            stats = invalidation_service.get_invalidation_stats()

            return {
                "success": True,
                "file_path": abs_path,
                "chunk_ids": chunk_ids,
                "event_id": event.event_id,
                "reason": event.reason.value,
                "affected_keys": len(event.affected_keys),
                "timestamp": event.timestamp.isoformat(),
                "statistics": {
                    "total_invalidations": stats.total_invalidations,
                    "partial_invalidations": stats.partial_invalidations,
                    "keys_invalidated": stats.keys_invalidated,
                },
                "metadata": event.metadata,
            }

        except Exception as e:
            return handle_tool_error(
                e,
                "invalidate_chunks",
                {
                    "file_path": file_path,
                    "chunk_ids_count": len(chunk_ids) if chunk_ids else 0,
                    "reason": reason,
                },
            )
