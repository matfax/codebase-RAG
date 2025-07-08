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
