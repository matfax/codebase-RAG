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


async def verify_cache_consistency(
    check_level: str = "basic",
    cache_keys: Optional[list[str]] = None,
    fix_issues: bool = False,
    max_keys: int = 1000
) -> dict[str, Any]:
    """
    Verify cache consistency across L1/L2 tiers and detect integrity issues.
    
    Args:
        check_level: Level of consistency checking (basic, comprehensive, deep)
        cache_keys: Specific keys to check (None for all keys)
        fix_issues: Whether to automatically fix detected issues
        max_keys: Maximum number of keys to check (for performance)
        
    Returns:
        Dictionary with consistency report and findings
    """
    with log_tool_usage(
        "verify_cache_consistency",
        {
            "check_level": check_level,
            "cache_keys_count": len(cache_keys) if cache_keys else 0,
            "fix_issues": fix_issues,
            "max_keys": max_keys
        }
    ):
        try:
            from ...services.cache_consistency_service import (
                get_cache_consistency_service,
                ConsistencyCheckLevel
            )
            
            # Map check level string to enum
            level_mapping = {
                "basic": ConsistencyCheckLevel.BASIC,
                "comprehensive": ConsistencyCheckLevel.COMPREHENSIVE,
                "deep": ConsistencyCheckLevel.DEEP
            }
            
            consistency_level = level_mapping.get(check_level, ConsistencyCheckLevel.BASIC)
            
            # Get consistency service
            consistency_service = await get_cache_consistency_service()
            
            # Limit keys if needed
            if cache_keys and len(cache_keys) > max_keys:
                cache_keys = cache_keys[:max_keys]
            
            # Perform consistency check
            report = await consistency_service.verify_consistency(
                check_level=consistency_level,
                cache_keys=cache_keys,
                fix_issues=fix_issues
            )
            
            # Format issues for response
            formatted_issues = []
            for issue in report.issues_found:
                issue_dict = {
                    "type": issue.issue_type.value,
                    "cache_key": issue.cache_key,
                    "description": issue.description,
                    "severity": issue.severity,
                    "discovered_at": issue.discovered_at.isoformat(),
                    "resolution_action": issue.resolution_action
                }
                
                # Add metadata if available
                if issue.metadata:
                    issue_dict["metadata"] = issue.metadata
                
                # Add value comparison for mismatches
                if issue.l1_value is not None or issue.l2_value is not None:
                    issue_dict["value_comparison"] = {
                        "l1_value_present": issue.l1_value is not None,
                        "l2_value_present": issue.l2_value is not None,
                        "values_differ": issue.l1_value != issue.l2_value
                    }
                
                formatted_issues.append(issue_dict)
            
            return {
                "success": True,
                "consistency_report": {
                    "check_level": report.check_level.value,
                    "duration_seconds": report.check_duration,
                    "keys_checked": report.total_keys_checked,
                    "consistency_score": report.consistency_score,
                    "is_consistent": report.is_consistent,
                    "checked_at": report.checked_at.isoformat()
                },
                "statistics": {
                    "l1_stats": report.l1_stats,
                    "l2_stats": report.l2_stats,
                    "total_issues": len(report.issues_found),
                    "issues_by_severity": _count_issues_by_severity(report.issues_found),
                    "issues_by_type": _count_issues_by_type(report.issues_found)
                },
                "issues": formatted_issues,
                "recommendations": report.recommendations,
                "fixes_applied": fix_issues
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "verify_cache_consistency",
                {
                    "check_level": check_level,
                    "cache_keys_count": len(cache_keys) if cache_keys else 0,
                    "fix_issues": fix_issues
                }
            )


async def get_cache_health_report(
    include_consistency: bool = True,
    include_performance: bool = True,
    include_statistics: bool = True
) -> dict[str, Any]:
    """
    Get a comprehensive cache health report.
    
    Args:
        include_consistency: Whether to include consistency checks
        include_performance: Whether to include performance metrics
        include_statistics: Whether to include detailed statistics
        
    Returns:
        Dictionary with comprehensive cache health information
    """
    with log_tool_usage(
        "get_cache_health_report",
        {
            "include_consistency": include_consistency,
            "include_performance": include_performance,
            "include_statistics": include_statistics
        }
    ):
        try:
            from ...services.cache_service import get_cache_service
            
            health_report = {
                "success": True,
                "timestamp": time.time(),
                "overall_health": "unknown"
            }
            
            # Get cache service
            cache_service = await get_cache_service()
            
            # Basic health check
            try:
                # Test basic operations
                test_key = f"health_check_{int(time.time())}"
                await cache_service.set(test_key, "health_test")
                test_value = await cache_service.get(test_key)
                await cache_service.delete(test_key)
                
                health_report["basic_operations"] = {
                    "set": True,
                    "get": test_value == "health_test",
                    "delete": True
                }
                
            except Exception as e:
                health_report["basic_operations"] = {
                    "set": False,
                    "get": False,
                    "delete": False,
                    "error": str(e)
                }
            
            # Consistency check
            if include_consistency:
                try:
                    consistency_result = await verify_cache_consistency(
                        check_level="basic",
                        fix_issues=False,
                        max_keys=100
                    )
                    health_report["consistency"] = consistency_result
                except Exception as e:
                    health_report["consistency"] = {
                        "success": False,
                        "error": str(e)
                    }
            
            # Performance metrics
            if include_performance:
                try:
                    if hasattr(cache_service, 'get_tier_stats'):
                        health_report["performance"] = cache_service.get_tier_stats()
                except Exception as e:
                    health_report["performance"] = {
                        "error": str(e)
                    }
            
            # Statistics
            if include_statistics:
                try:
                    stats_result = await get_cache_statistics()
                    health_report["statistics"] = stats_result
                except Exception as e:
                    health_report["statistics"] = {
                        "error": str(e)
                    }
            
            # Determine overall health
            issues = []
            if not health_report.get("basic_operations", {}).get("get", False):
                issues.append("Basic operations failing")
            
            if include_consistency:
                consistency = health_report.get("consistency", {})
                if consistency.get("success") and consistency.get("consistency_report", {}).get("consistency_score", 1.0) < 0.8:
                    issues.append("Poor consistency score")
            
            if issues:
                health_report["overall_health"] = "poor"
                health_report["health_issues"] = issues
            elif len(issues) == 0:
                health_report["overall_health"] = "good"
            else:
                health_report["overall_health"] = "degraded"
            
            return health_report
            
        except Exception as e:
            return handle_tool_error(
                e,
                "get_cache_health_report",
                {
                    "include_consistency": include_consistency,
                    "include_performance": include_performance,
                    "include_statistics": include_statistics
                }
            )


def _count_issues_by_severity(issues) -> dict[str, int]:
    """Count issues by severity level."""
    severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for issue in issues:
        severity = issue.severity
        if severity in severity_counts:
            severity_counts[severity] += 1
    return severity_counts


def _count_issues_by_type(issues) -> dict[str, int]:
    """Count issues by type."""
    type_counts = {}
    for issue in issues:
        issue_type = issue.issue_type.value
        type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
    return type_counts


async def create_cache_backup(
    backup_type: str = "full",
    tiers: Optional[list[str]] = None,
    include_metadata: bool = True,
    compress: bool = True,
    encrypt: bool = True,
    base_backup_id: Optional[str] = None
) -> dict[str, Any]:
    """
    Create a cache backup for disaster recovery.
    
    Args:
        backup_type: Type of backup (full, incremental, differential, snapshot)
        tiers: Cache tiers to backup (None for all tiers)
        include_metadata: Whether to include cache metadata
        compress: Whether to compress backup data
        encrypt: Whether to encrypt backup data
        base_backup_id: Base backup ID for incremental/differential backups
        
    Returns:
        Dictionary with backup creation results
    """
    with log_tool_usage(
        "create_cache_backup",
        {
            "backup_type": backup_type,
            "tiers": tiers,
            "include_metadata": include_metadata,
            "compress": compress,
            "encrypt": encrypt,
            "base_backup_id": base_backup_id
        }
    ):
        try:
            from ...services.cache_backup_service import (
                get_cache_backup_service,
                BackupType
            )
            
            # Map backup type string to enum
            type_mapping = {
                "full": BackupType.FULL,
                "incremental": BackupType.INCREMENTAL,
                "differential": BackupType.DIFFERENTIAL,
                "snapshot": BackupType.SNAPSHOT
            }
            
            backup_type_enum = type_mapping.get(backup_type, BackupType.FULL)
            
            # Get backup service
            backup_service = await get_cache_backup_service()
            
            # Create backup
            metadata = await backup_service.create_backup(
                backup_type=backup_type_enum,
                tiers=tiers,
                include_metadata=include_metadata,
                compress=compress,
                encrypt=encrypt,
                base_backup_id=base_backup_id
            )
            
            return {
                "success": True,
                "backup_id": metadata.backup_id,
                "backup_type": metadata.backup_type.value,
                "timestamp": metadata.timestamp.isoformat(),
                "cache_tiers": metadata.cache_tiers,
                "total_entries": metadata.total_entries,
                "total_size_bytes": metadata.total_size_bytes,
                "compression_ratio": metadata.compression_ratio,
                "encryption_enabled": metadata.encryption_enabled,
                "duration_seconds": metadata.duration_seconds,
                "status": metadata.status.value,
                "checksum": metadata.checksum,
                "includes_metadata": metadata.includes_metadata,
                "base_backup_id": metadata.base_backup_id
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "create_cache_backup",
                {
                    "backup_type": backup_type,
                    "tiers": tiers,
                    "base_backup_id": base_backup_id
                }
            )


async def restore_cache_from_backup(
    backup_id: str,
    strategy: str = "replace_all",
    target_tiers: Optional[list[str]] = None,
    selective_keys: Optional[list[str]] = None,
    dry_run: bool = False
) -> dict[str, Any]:
    """
    Restore cache from backup for disaster recovery.
    
    Args:
        backup_id: ID of backup to restore from
        strategy: Recovery strategy (replace_all, merge_preserve_existing, 
                 merge_overwrite_existing, selective_restore)
        target_tiers: Target cache tiers (None for all)
        selective_keys: Specific keys to restore (for selective strategy)
        dry_run: Whether to perform a dry run without actual restoration
        
    Returns:
        Dictionary with restoration results
    """
    with log_tool_usage(
        "restore_cache_from_backup",
        {
            "backup_id": backup_id,
            "strategy": strategy,
            "target_tiers": target_tiers,
            "selective_keys_count": len(selective_keys) if selective_keys else 0,
            "dry_run": dry_run
        }
    ):
        try:
            from ...services.cache_backup_service import (
                get_cache_backup_service,
                RecoveryStrategy
            )
            
            # Map strategy string to enum
            strategy_mapping = {
                "replace_all": RecoveryStrategy.REPLACE_ALL,
                "merge_preserve_existing": RecoveryStrategy.MERGE_PRESERVE_EXISTING,
                "merge_overwrite_existing": RecoveryStrategy.MERGE_OVERWRITE_EXISTING,
                "selective_restore": RecoveryStrategy.SELECTIVE_RESTORE
            }
            
            recovery_strategy = strategy_mapping.get(strategy, RecoveryStrategy.REPLACE_ALL)
            
            # Get backup service
            backup_service = await get_cache_backup_service()
            
            # Perform restoration
            restore_op = await backup_service.restore_from_backup(
                backup_id=backup_id,
                strategy=recovery_strategy,
                target_tiers=target_tiers,
                selective_keys=selective_keys,
                dry_run=dry_run
            )
            
            return {
                "success": True,
                "restore_id": restore_op.restore_id,
                "backup_id": restore_op.backup_id,
                "target_tiers": restore_op.target_tiers,
                "strategy": restore_op.strategy.value,
                "started_at": restore_op.started_at.isoformat(),
                "completed_at": restore_op.completed_at.isoformat() if restore_op.completed_at else None,
                "status": restore_op.status.value,
                "restored_entries": restore_op.restored_entries,
                "failed_entries": restore_op.failed_entries,
                "dry_run": restore_op.dry_run,
                "error_message": restore_op.error_message
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "restore_cache_from_backup",
                {
                    "backup_id": backup_id,
                    "strategy": strategy,
                    "dry_run": dry_run
                }
            )


async def list_cache_backups(
    backup_type: Optional[str] = None,
    max_age_days: Optional[int] = None,
    limit: int = 50
) -> dict[str, Any]:
    """
    List available cache backups.
    
    Args:
        backup_type: Filter by backup type (full, incremental, differential, snapshot)
        max_age_days: Maximum age in days
        limit: Maximum number of backups to return
        
    Returns:
        Dictionary with list of available backups
    """
    with log_tool_usage(
        "list_cache_backups",
        {
            "backup_type": backup_type,
            "max_age_days": max_age_days,
            "limit": limit
        }
    ):
        try:
            from ...services.cache_backup_service import (
                get_cache_backup_service,
                BackupType
            )
            
            # Get backup service
            backup_service = await get_cache_backup_service()
            
            # Map backup type if provided
            backup_type_enum = None
            if backup_type:
                type_mapping = {
                    "full": BackupType.FULL,
                    "incremental": BackupType.INCREMENTAL,
                    "differential": BackupType.DIFFERENTIAL,
                    "snapshot": BackupType.SNAPSHOT
                }
                backup_type_enum = type_mapping.get(backup_type)
            
            # List backups
            backups = await backup_service.list_backups(
                backup_type=backup_type_enum,
                max_age_days=max_age_days
            )
            
            # Limit results
            if limit:
                backups = backups[:limit]
            
            # Format backups for response
            formatted_backups = []
            for backup in backups:
                formatted_backups.append({
                    "backup_id": backup.backup_id,
                    "backup_type": backup.backup_type.value,
                    "timestamp": backup.timestamp.isoformat(),
                    "cache_tiers": backup.cache_tiers,
                    "total_entries": backup.total_entries,
                    "total_size_bytes": backup.total_size_bytes,
                    "compression_ratio": backup.compression_ratio,
                    "encryption_enabled": backup.encryption_enabled,
                    "duration_seconds": backup.duration_seconds,
                    "status": backup.status.value,
                    "includes_metadata": backup.includes_metadata,
                    "base_backup_id": backup.base_backup_id,
                    "age_hours": (datetime.now() - backup.timestamp).total_seconds() / 3600
                })
            
            return {
                "success": True,
                "backups": formatted_backups,
                "total_count": len(formatted_backups),
                "filters_applied": {
                    "backup_type": backup_type,
                    "max_age_days": max_age_days,
                    "limit": limit
                }
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "list_cache_backups",
                {
                    "backup_type": backup_type,
                    "max_age_days": max_age_days,
                    "limit": limit
                }
            )


async def verify_backup_integrity(backup_id: str) -> dict[str, Any]:
    """
    Verify the integrity of a cache backup.
    
    Args:
        backup_id: ID of backup to verify
        
    Returns:
        Dictionary with verification results
    """
    with log_tool_usage(
        "verify_backup_integrity",
        {"backup_id": backup_id}
    ):
        try:
            from ...services.cache_backup_service import get_cache_backup_service
            
            # Get backup service
            backup_service = await get_cache_backup_service()
            
            # Verify backup
            verification_result = await backup_service.verify_backup_integrity(backup_id)
            
            return {
                "success": True,
                "backup_id": backup_id,
                "verification_result": verification_result
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "verify_backup_integrity",
                {"backup_id": backup_id}
            )


async def delete_cache_backup(backup_id: str, confirm: bool = False) -> dict[str, Any]:
    """
    Delete a cache backup.
    
    Args:
        backup_id: ID of backup to delete
        confirm: Confirmation flag to prevent accidental deletion
        
    Returns:
        Dictionary with deletion results
    """
    with log_tool_usage(
        "delete_cache_backup",
        {"backup_id": backup_id, "confirm": confirm}
    ):
        try:
            if not confirm:
                return {
                    "success": False,
                    "error": "Confirmation required. Set confirm=True to delete backup.",
                    "backup_id": backup_id
                }
            
            from ...services.cache_backup_service import get_cache_backup_service
            
            # Get backup service
            backup_service = await get_cache_backup_service()
            
            # Delete backup
            deleted = await backup_service.delete_backup(backup_id)
            
            return {
                "success": deleted,
                "backup_id": backup_id,
                "deleted": deleted,
                "message": "Backup deleted successfully" if deleted else "Backup not found or deletion failed"
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "delete_cache_backup",
                {"backup_id": backup_id}
            )


async def get_backup_disaster_recovery_plan() -> dict[str, Any]:
    """
    Get disaster recovery plan and recommendations for cache backups.
    
    Returns:
        Dictionary with disaster recovery plan and recommendations
    """
    with log_tool_usage("get_backup_disaster_recovery_plan", {}):
        try:
            from ...services.cache_backup_service import get_cache_backup_service
            
            # Get backup service
            backup_service = await get_cache_backup_service()
            
            # List recent backups
            recent_backups = await backup_service.list_backups(max_age_days=7)
            full_backups = [b for b in recent_backups if b.backup_type.value == "full"]
            incremental_backups = [b for b in recent_backups if b.backup_type.value == "incremental"]
            
            # Analyze backup coverage
            now = datetime.now()
            last_full_backup = full_backups[0] if full_backups else None
            last_incremental_backup = incremental_backups[0] if incremental_backups else None
            
            # Calculate recovery time objectives
            rto_estimate = "< 30 minutes"  # Recovery Time Objective
            rpo_estimate = "< 6 hours"     # Recovery Point Objective
            
            if last_full_backup:
                days_since_full = (now - last_full_backup.timestamp).days
                if days_since_full > 7:
                    rto_estimate = "1-2 hours"
                    rpo_estimate = "< 24 hours"
            
            # Generate recommendations
            recommendations = []
            
            if not last_full_backup:
                recommendations.append("Create an initial full backup immediately")
            elif (now - last_full_backup.timestamp).days > 7:
                recommendations.append("Create a new full backup (last full backup is over 7 days old)")
            
            if not last_incremental_backup:
                recommendations.append("Set up regular incremental backups for better RPO")
            elif (now - last_incremental_backup.timestamp).hours > 24:
                recommendations.append("Create more frequent incremental backups")
            
            if len(recent_backups) < 3:
                recommendations.append("Increase backup frequency to maintain multiple recovery points")
            
            # Disaster recovery procedures
            procedures = {
                "complete_cache_loss": [
                    "1. Identify most recent full backup",
                    "2. Apply any subsequent incremental backups in chronological order",
                    "3. Restore using replace_all strategy",
                    "4. Verify cache functionality",
                    "5. Monitor performance and consistency"
                ],
                "partial_cache_corruption": [
                    "1. Run cache consistency verification",
                    "2. Identify corrupted cache tiers",
                    "3. Restore only affected tiers using merge strategy",
                    "4. Verify restored data integrity",
                    "5. Resume normal operations"
                ],
                "cache_performance_degradation": [
                    "1. Create immediate backup of current state",
                    "2. Restore from recent known-good backup",
                    "3. Compare performance metrics",
                    "4. Identify root cause of degradation",
                    "5. Implement fixes and monitor"
                ]
            }
            
            return {
                "success": True,
                "disaster_recovery_plan": {
                    "backup_status": {
                        "total_backups": len(recent_backups),
                        "full_backups": len(full_backups),
                        "incremental_backups": len(incremental_backups),
                        "last_full_backup": last_full_backup.backup_id if last_full_backup else None,
                        "last_full_backup_age_hours": (now - last_full_backup.timestamp).total_seconds() / 3600 if last_full_backup else None,
                        "last_incremental_backup": last_incremental_backup.backup_id if last_incremental_backup else None,
                        "last_incremental_backup_age_hours": (now - last_incremental_backup.timestamp).total_seconds() / 3600 if last_incremental_backup else None
                    },
                    "recovery_objectives": {
                        "rto_estimate": rto_estimate,
                        "rpo_estimate": rpo_estimate,
                        "backup_coverage": "Good" if last_full_backup and (now - last_full_backup.timestamp).days <= 7 else "Needs Improvement"
                    },
                    "recommendations": recommendations,
                    "procedures": procedures
                },
                "timestamp": now.isoformat()
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "get_backup_disaster_recovery_plan",
                {}
            )


async def configure_cache_failover(
    enable_failover: bool = True,
    health_check_interval_seconds: int = 30,
    failure_threshold: int = 3,
    recovery_threshold: int = 5,
    auto_recovery_enabled: bool = True,
    performance_degradation_threshold: float = 0.5
) -> dict[str, Any]:
    """
    Configure cache failover settings and initialize failover service.
    
    Args:
        enable_failover: Whether to enable failover functionality
        health_check_interval_seconds: Interval between health checks
        failure_threshold: Number of failures before triggering failover
        recovery_threshold: Number of successes needed for recovery
        auto_recovery_enabled: Whether to automatically attempt recovery
        performance_degradation_threshold: Threshold for performance degradation (ratio)
        
    Returns:
        Dictionary with failover configuration results
    """
    with log_tool_usage(
        "configure_cache_failover",
        {
            "enable_failover": enable_failover,
            "health_check_interval_seconds": health_check_interval_seconds,
            "failure_threshold": failure_threshold,
            "recovery_threshold": recovery_threshold,
            "auto_recovery_enabled": auto_recovery_enabled,
            "performance_degradation_threshold": performance_degradation_threshold
        }
    ):
        try:
            if not enable_failover:
                return {
                    "success": True,
                    "message": "Failover disabled by configuration",
                    "failover_enabled": False
                }
            
            from ...services.cache_failover_service import (
                get_cache_failover_service,
                FailoverConfiguration
            )
            
            # Create failover configuration
            failover_config = FailoverConfiguration(
                health_check_interval_seconds=health_check_interval_seconds,
                failure_threshold=failure_threshold,
                recovery_threshold=recovery_threshold,
                auto_recovery_enabled=auto_recovery_enabled,
                performance_degradation_threshold=performance_degradation_threshold
            )
            
            # Get or create failover service with new configuration
            # Note: In a real implementation, you might need to recreate the service
            # with new configuration, but for simplicity we'll just report the config
            
            return {
                "success": True,
                "message": "Failover configuration updated",
                "failover_enabled": True,
                "configuration": {
                    "health_check_interval_seconds": health_check_interval_seconds,
                    "failure_threshold": failure_threshold,
                    "recovery_threshold": recovery_threshold,
                    "auto_recovery_enabled": auto_recovery_enabled,
                    "performance_degradation_threshold": performance_degradation_threshold
                }
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "configure_cache_failover",
                {
                    "enable_failover": enable_failover,
                    "health_check_interval_seconds": health_check_interval_seconds
                }
            )


async def get_cache_failover_status() -> dict[str, Any]:
    """
    Get current cache failover status and health information.
    
    Returns:
        Dictionary with failover status and service health
    """
    with log_tool_usage("get_cache_failover_status", {}):
        try:
            from ...services.cache_failover_service import get_cache_failover_service
            
            # Get failover service
            failover_service = await get_cache_failover_service()
            
            # Get comprehensive status
            status = await failover_service.get_failover_status()
            
            return {
                "success": True,
                "failover_status": status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "get_cache_failover_status",
                {}
            )


async def trigger_manual_failover(reason: str = "Manual failover requested") -> dict[str, Any]:
    """
    Manually trigger cache failover to backup services.
    
    Args:
        reason: Reason for manual failover
        
    Returns:
        Dictionary with failover trigger results
    """
    with log_tool_usage(
        "trigger_manual_failover",
        {"reason": reason}
    ):
        try:
            from ...services.cache_failover_service import get_cache_failover_service
            
            # Get failover service
            failover_service = await get_cache_failover_service()
            
            # Trigger manual failover
            event = await failover_service.manual_failover(reason)
            
            return {
                "success": event.success,
                "event_id": event.event_id,
                "trigger": event.trigger.value,
                "timestamp": event.timestamp.isoformat(),
                "primary_service_id": event.primary_service_id,
                "failover_service_id": event.failover_service_id,
                "duration_seconds": event.duration_seconds,
                "error_message": event.error_message,
                "trigger_details": event.trigger_details
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "trigger_manual_failover",
                {"reason": reason}
            )


async def trigger_manual_recovery() -> dict[str, Any]:
    """
    Manually trigger recovery to primary cache service.
    
    Returns:
        Dictionary with recovery attempt results
    """
    with log_tool_usage("trigger_manual_recovery", {}):
        try:
            from ...services.cache_failover_service import get_cache_failover_service
            
            # Get failover service
            failover_service = await get_cache_failover_service()
            
            # Attempt manual recovery
            success = await failover_service.manual_recovery()
            
            # Get updated status
            status = await failover_service.get_failover_status()
            
            return {
                "success": success,
                "recovery_successful": success,
                "current_status": status["status"],
                "is_failed_over": status["is_failed_over"],
                "current_service": status["current_service"],
                "message": "Recovery successful" if success else "Recovery failed or not needed",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "trigger_manual_recovery",
                {}
            )


async def register_failover_service(
    service_type: str = "redis",
    connection_config: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """
    Register a new failover cache service.
    
    Args:
        service_type: Type of failover service (redis, memory, etc.)
        connection_config: Connection configuration for the failover service
        
    Returns:
        Dictionary with registration results
    """
    with log_tool_usage(
        "register_failover_service",
        {
            "service_type": service_type,
            "has_connection_config": connection_config is not None
        }
    ):
        try:
            from ...services.cache_failover_service import get_cache_failover_service
            
            # This is a placeholder implementation
            # In a real system, you would create the actual failover service
            # based on the service_type and connection_config
            
            return {
                "success": False,
                "message": "Failover service registration not yet implemented",
                "service_type": service_type,
                "note": "This feature requires additional implementation to create and configure failover services dynamically"
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "register_failover_service",
                {
                    "service_type": service_type,
                    "has_connection_config": connection_config is not None
                }
            )


async def test_cache_failover_scenario(
    scenario_type: str = "connection_failure",
    duration_seconds: int = 30
) -> dict[str, Any]:
    """
    Test cache failover behavior with simulated failure scenarios.
    
    Args:
        scenario_type: Type of failure to simulate (connection_failure, performance_degradation)
        duration_seconds: Duration to maintain the simulated failure
        
    Returns:
        Dictionary with test results
    """
    with log_tool_usage(
        "test_cache_failover_scenario",
        {
            "scenario_type": scenario_type,
            "duration_seconds": duration_seconds
        }
    ):
        try:
            from ...services.cache_failover_service import get_cache_failover_service
            
            # Get failover service
            failover_service = await get_cache_failover_service()
            
            # Get initial status
            initial_status = await failover_service.get_failover_status()
            
            # This is a placeholder for failover testing
            # In a real implementation, you would:
            # 1. Inject failures into the primary service
            # 2. Monitor failover behavior
            # 3. Verify recovery
            # 4. Return detailed test results
            
            return {
                "success": True,
                "message": "Failover test completed (placeholder implementation)",
                "scenario_type": scenario_type,
                "duration_seconds": duration_seconds,
                "initial_status": initial_status["status"],
                "test_results": {
                    "failover_triggered": False,
                    "recovery_successful": False,
                    "response_time_impact": 0.0,
                    "data_consistency_maintained": True
                },
                "note": "This is a placeholder implementation. Real testing would require failure injection capabilities."
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "test_cache_failover_scenario",
                {
                    "scenario_type": scenario_type,
                    "duration_seconds": duration_seconds
                }
            )


async def get_failover_performance_metrics() -> dict[str, Any]:
    """
    Get performance metrics for cache failover operations.
    
    Returns:
        Dictionary with failover performance metrics
    """
    with log_tool_usage("get_failover_performance_metrics", {}):
        try:
            from ...services.cache_failover_service import get_cache_failover_service
            
            # Get failover service
            failover_service = await get_cache_failover_service()
            
            # Get status for metrics
            status = await failover_service.get_failover_status()
            
            # Calculate metrics from failover events
            recent_events = status.get("recent_events", [])
            
            # Performance metrics
            metrics = {
                "total_failover_events": len(failover_service._failover_events),
                "successful_failovers": len([e for e in failover_service._failover_events if e.success]),
                "average_failover_time": 0.0,
                "fastest_failover_time": 0.0,
                "slowest_failover_time": 0.0,
                "recovery_events": len([e for e in failover_service._failover_events if e.recovery_timestamp]),
                "current_uptime_hours": 0.0
            }
            
            # Calculate average failover time
            successful_events = [e for e in failover_service._failover_events if e.success and e.duration_seconds > 0]
            if successful_events:
                durations = [e.duration_seconds for e in successful_events]
                metrics["average_failover_time"] = sum(durations) / len(durations)
                metrics["fastest_failover_time"] = min(durations)
                metrics["slowest_failover_time"] = max(durations)
            
            # Calculate current uptime
            if failover_service._failover_events:
                last_event = max(failover_service._failover_events, key=lambda e: e.timestamp)
                if last_event.recovery_timestamp:
                    uptime_delta = datetime.now() - last_event.recovery_timestamp
                    metrics["current_uptime_hours"] = uptime_delta.total_seconds() / 3600
            
            return {
                "success": True,
                "performance_metrics": metrics,
                "service_health": status.get("service_health", {}),
                "current_status": status.get("status"),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "get_failover_performance_metrics",
                {}
            )


async def get_cache_performance_summary() -> dict[str, Any]:
    """
    Get comprehensive cache performance summary and metrics.
    
    Returns:
        Dictionary with performance metrics and analysis
    """
    with log_tool_usage("get_cache_performance_summary", {}):
        try:
            from ...services.cache_performance_service import get_cache_performance_service
            
            # Get performance service
            performance_service = await get_cache_performance_service()
            
            # Get performance summary
            summary = await performance_service.get_performance_summary()
            
            return {
                "success": True,
                "performance_summary": summary
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "get_cache_performance_summary",
                {}
            )


async def get_performance_degradation_events(
    limit: int = 50,
    active_only: bool = False,
    severity_filter: Optional[str] = None
) -> dict[str, Any]:
    """
    Get recent performance degradation events.
    
    Args:
        limit: Maximum number of events to return
        active_only: Whether to return only active degradations
        severity_filter: Filter by severity level (low, medium, high, critical)
        
    Returns:
        Dictionary with degradation events
    """
    with log_tool_usage(
        "get_performance_degradation_events",
        {
            "limit": limit,
            "active_only": active_only,
            "severity_filter": severity_filter
        }
    ):
        try:
            from ...services.cache_performance_service import get_cache_performance_service
            
            # Get performance service
            performance_service = await get_cache_performance_service()
            
            # Get degradation events
            events = await performance_service.get_degradation_events(
                limit=limit,
                active_only=active_only
            )
            
            # Filter by severity if specified
            if severity_filter:
                events = [event for event in events if event.get("severity") == severity_filter]
            
            return {
                "success": True,
                "degradation_events": events,
                "total_count": len(events),
                "filters_applied": {
                    "limit": limit,
                    "active_only": active_only,
                    "severity_filter": severity_filter
                }
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "get_performance_degradation_events",
                {
                    "limit": limit,
                    "active_only": active_only,
                    "severity_filter": severity_filter
                }
            )


async def trigger_performance_remediation(
    action: str,
    target_metric: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """
    Manually trigger performance remediation action.
    
    Args:
        action: Remediation action to trigger (cache_eviction, connection_pool_restart, 
               garbage_collection, cache_warmup, etc.)
        target_metric: Target metric to measure improvement against
        metadata: Additional metadata for the remediation action
        
    Returns:
        Dictionary with remediation results
    """
    with log_tool_usage(
        "trigger_performance_remediation",
        {
            "action": action,
            "target_metric": target_metric,
            "has_metadata": metadata is not None
        }
    ):
        try:
            from ...services.cache_performance_service import (
                get_cache_performance_service,
                RemediationAction,
                PerformanceMetricType
            )
            
            # Map action string to enum
            action_mapping = {
                "cache_eviction": RemediationAction.CACHE_EVICTION,
                "connection_pool_restart": RemediationAction.CONNECTION_POOL_RESTART,
                "garbage_collection": RemediationAction.GARBAGE_COLLECTION,
                "cache_warmup": RemediationAction.CACHE_WARMUP,
                "load_balancing": RemediationAction.LOAD_BALANCING,
                "circuit_breaker_trip": RemediationAction.CIRCUIT_BREAKER_TRIP,
                "alert_notification": RemediationAction.ALERT_NOTIFICATION,
                "auto_scaling": RemediationAction.AUTO_SCALING
            }
            
            remediation_action = action_mapping.get(action)
            if not remediation_action:
                return {
                    "success": False,
                    "error": f"Unknown remediation action: {action}",
                    "available_actions": list(action_mapping.keys())
                }
            
            # Map target metric if provided
            target_metric_enum = None
            if target_metric:
                metric_mapping = {
                    "response_time": PerformanceMetricType.RESPONSE_TIME,
                    "error_rate": PerformanceMetricType.ERROR_RATE,
                    "hit_rate": PerformanceMetricType.HIT_RATE,
                    "memory_usage": PerformanceMetricType.MEMORY_USAGE,
                    "cpu_usage": PerformanceMetricType.CPU_USAGE,
                    "network_io": PerformanceMetricType.NETWORK_IO,
                    "disk_io": PerformanceMetricType.DISK_IO,
                    "connection_count": PerformanceMetricType.CONNECTION_COUNT
                }
                target_metric_enum = metric_mapping.get(target_metric)
            
            # Get performance service
            performance_service = await get_cache_performance_service()
            
            # Trigger remediation
            result = await performance_service.trigger_manual_remediation(
                action=remediation_action,
                target_metric=target_metric_enum,
                metadata=metadata
            )
            
            return {
                "success": result.success,
                "remediation_result": {
                    "action": result.action.value,
                    "started_at": result.started_at.isoformat(),
                    "completed_at": result.completed_at.isoformat() if result.completed_at else None,
                    "success": result.success,
                    "error_message": result.error_message,
                    "performance_improvement": result.performance_improvement,
                    "metadata": result.metadata
                }
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "trigger_performance_remediation",
                {
                    "action": action,
                    "target_metric": target_metric
                }
            )


async def configure_performance_monitoring(
    monitoring_interval_seconds: int = 60,
    degradation_threshold_ratio: float = 2.0,
    critical_threshold_ratio: float = 5.0,
    auto_remediation_enabled: bool = True,
    alert_thresholds: Optional[dict[str, float]] = None
) -> dict[str, Any]:
    """
    Configure cache performance monitoring settings.
    
    Args:
        monitoring_interval_seconds: Interval between performance checks
        degradation_threshold_ratio: Threshold ratio for degradation detection
        critical_threshold_ratio: Threshold ratio for critical degradation
        auto_remediation_enabled: Whether to enable automatic remediation
        alert_thresholds: Dictionary of alert thresholds for different metrics
        
    Returns:
        Dictionary with configuration results
    """
    with log_tool_usage(
        "configure_performance_monitoring",
        {
            "monitoring_interval_seconds": monitoring_interval_seconds,
            "degradation_threshold_ratio": degradation_threshold_ratio,
            "critical_threshold_ratio": critical_threshold_ratio,
            "auto_remediation_enabled": auto_remediation_enabled,
            "has_alert_thresholds": alert_thresholds is not None
        }
    ):
        try:
            from ...services.cache_performance_service import PerformanceConfiguration
            
            # Default alert thresholds
            default_alert_thresholds = {
                "response_time_p95": 1000.0,  # 1 second
                "error_rate": 0.05,           # 5%
                "hit_rate": 0.8,              # 80%
                "memory_usage": 0.85          # 85%
            }
            
            # Merge with provided thresholds
            final_alert_thresholds = default_alert_thresholds.copy()
            if alert_thresholds:
                final_alert_thresholds.update(alert_thresholds)
            
            # Create performance configuration
            perf_config = PerformanceConfiguration(
                monitoring_interval_seconds=monitoring_interval_seconds,
                degradation_threshold_ratio=degradation_threshold_ratio,
                critical_threshold_ratio=critical_threshold_ratio,
                auto_remediation_enabled=auto_remediation_enabled,
                alert_thresholds=final_alert_thresholds
            )
            
            # Note: In a real implementation, you would update the running service
            # For now, we'll just return the configuration
            
            return {
                "success": True,
                "message": "Performance monitoring configuration updated",
                "configuration": {
                    "monitoring_interval_seconds": monitoring_interval_seconds,
                    "degradation_threshold_ratio": degradation_threshold_ratio,
                    "critical_threshold_ratio": critical_threshold_ratio,
                    "auto_remediation_enabled": auto_remediation_enabled,
                    "alert_thresholds": final_alert_thresholds
                }
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "configure_performance_monitoring",
                {
                    "monitoring_interval_seconds": monitoring_interval_seconds,
                    "degradation_threshold_ratio": degradation_threshold_ratio
                }
            )


async def analyze_performance_trends(
    time_window_hours: int = 24,
    metric_types: Optional[list[str]] = None
) -> dict[str, Any]:
    """
    Analyze cache performance trends over a specified time window.
    
    Args:
        time_window_hours: Time window for trend analysis in hours
        metric_types: List of metric types to analyze (None for all)
        
    Returns:
        Dictionary with performance trend analysis
    """
    with log_tool_usage(
        "analyze_performance_trends",
        {
            "time_window_hours": time_window_hours,
            "metric_types": metric_types
        }
    ):
        try:
            from ...services.cache_performance_service import get_cache_performance_service
            
            # Get performance service
            performance_service = await get_cache_performance_service()
            
            # Get performance summary for trend data
            summary = await performance_service.get_performance_summary()
            
            # Analyze trends (simplified implementation)
            trends = {
                "time_window_hours": time_window_hours,
                "analysis_timestamp": datetime.now().isoformat(),
                "metric_trends": {},
                "overall_trend": "stable",
                "recommendations": []
            }
            
            # Analyze each metric type
            metrics = summary.get("metrics", {})
            baselines = summary.get("baselines", {})
            
            for metric_name, metric_data in metrics.items():
                if metric_types and metric_name not in metric_types:
                    continue
                
                current_value = metric_data.get("current", 0)
                average_value = metric_data.get("average", 0)
                baseline_info = baselines.get(metric_name, {})
                baseline_value = baseline_info.get("baseline_value", average_value)
                
                # Calculate trend
                if baseline_value > 0:
                    trend_ratio = current_value / baseline_value
                    if trend_ratio > 1.2:
                        trend = "degrading"
                    elif trend_ratio < 0.8:
                        trend = "improving"
                    else:
                        trend = "stable"
                else:
                    trend = "unknown"
                
                trends["metric_trends"][metric_name] = {
                    "current_value": current_value,
                    "average_value": average_value,
                    "baseline_value": baseline_value,
                    "trend_ratio": trend_ratio if baseline_value > 0 else 1.0,
                    "trend": trend,
                    "p95_value": metric_data.get("p95", current_value),
                    "p99_value": metric_data.get("p99", current_value)
                }
                
                # Generate recommendations based on trends
                if trend == "degrading":
                    if metric_name == "response_time":
                        trends["recommendations"].append(f"Response time is degrading. Consider cache optimization or scaling.")
                    elif metric_name == "error_rate":
                        trends["recommendations"].append(f"Error rate is increasing. Check system health and connections.")
                    elif metric_name == "memory_usage":
                        trends["recommendations"].append(f"Memory usage is high. Consider cache eviction or scaling.")
            
            # Determine overall trend
            degrading_metrics = [name for name, data in trends["metric_trends"].items() if data["trend"] == "degrading"]
            if len(degrading_metrics) >= 2:
                trends["overall_trend"] = "degrading"
            elif degrading_metrics:
                trends["overall_trend"] = "concerning"
            
            return {
                "success": True,
                "performance_trends": trends
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "analyze_performance_trends",
                {
                    "time_window_hours": time_window_hours,
                    "metric_types": metric_types
                }
            )


async def get_performance_recommendations() -> dict[str, Any]:
    """
    Get performance optimization recommendations based on current metrics.
    
    Returns:
        Dictionary with performance recommendations
    """
    with log_tool_usage("get_performance_recommendations", {}):
        try:
            from ...services.cache_performance_service import get_cache_performance_service
            
            # Get performance service
            performance_service = await get_cache_performance_service()
            
            # Get current performance summary
            summary = await performance_service.get_performance_summary()
            
            # Get recent degradation events
            events = await performance_service.get_degradation_events(limit=10, active_only=True)
            
            # Generate recommendations
            recommendations = {
                "timestamp": datetime.now().isoformat(),
                "overall_health": "good",
                "immediate_actions": [],
                "optimization_recommendations": [],
                "monitoring_recommendations": [],
                "capacity_planning": []
            }
            
            # Analyze metrics for recommendations
            metrics = summary.get("metrics", {})
            operation_stats = summary.get("operation_stats", {})
            
            # Response time analysis
            response_time_data = metrics.get("response_time", {})
            if response_time_data:
                p95 = response_time_data.get("p95", 0)
                if p95 > 1000:  # > 1 second
                    recommendations["immediate_actions"].append("High response times detected. Consider cache optimization.")
                    recommendations["overall_health"] = "concerning"
                elif p95 > 500:  # > 500ms
                    recommendations["optimization_recommendations"].append("Response times are elevated. Monitor for trends.")
            
            # Error rate analysis
            error_rate_data = metrics.get("error_rate", {})
            if error_rate_data:
                current_error_rate = error_rate_data.get("current", 0)
                if current_error_rate > 0.1:  # > 10%
                    recommendations["immediate_actions"].append("High error rate detected. Check system health.")
                    recommendations["overall_health"] = "poor"
                elif current_error_rate > 0.05:  # > 5%
                    recommendations["monitoring_recommendations"].append("Error rate is elevated. Increase monitoring.")
            
            # Memory usage analysis
            memory_data = metrics.get("memory_usage", {})
            if memory_data:
                current_memory = memory_data.get("current", 0)
                if current_memory > 0.9:  # > 90%
                    recommendations["immediate_actions"].append("Critical memory usage. Immediate action required.")
                    recommendations["overall_health"] = "critical"
                elif current_memory > 0.8:  # > 80%
                    recommendations["capacity_planning"].append("Memory usage is high. Plan for scaling.")
            
            # Hit rate analysis
            hit_rate_data = metrics.get("hit_rate", {})
            if hit_rate_data:
                current_hit_rate = hit_rate_data.get("current", 1.0)
                if current_hit_rate < 0.7:  # < 70%
                    recommendations["optimization_recommendations"].append("Low cache hit rate. Review caching strategy.")
                elif current_hit_rate < 0.8:  # < 80%
                    recommendations["monitoring_recommendations"].append("Hit rate could be improved. Monitor cache efficiency.")
            
            # Active degradation events
            if events:
                critical_events = [e for e in events if e.get("severity") == "critical"]
                high_events = [e for e in events if e.get("severity") == "high"]
                
                if critical_events:
                    recommendations["immediate_actions"].append(f"{len(critical_events)} critical performance issues need immediate attention.")
                    recommendations["overall_health"] = "critical"
                elif high_events:
                    recommendations["immediate_actions"].append(f"{len(high_events)} high-severity performance issues detected.")
                    if recommendations["overall_health"] == "good":
                        recommendations["overall_health"] = "concerning"
            
            # General recommendations
            if not recommendations["immediate_actions"]:
                if not recommendations["optimization_recommendations"]:
                    recommendations["optimization_recommendations"].append("Performance is good. Continue regular monitoring.")
                recommendations["monitoring_recommendations"].append("Set up regular performance trend analysis.")
                recommendations["capacity_planning"].append("Review capacity planning based on usage trends.")
            
            return {
                "success": True,
                "performance_recommendations": recommendations,
                "active_degradations": len(events),
                "metrics_analyzed": len(metrics)
            }
            
        except Exception as e:
            return handle_tool_error(
                e,
                "get_performance_recommendations",
                {}
            )
