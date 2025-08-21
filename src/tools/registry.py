"""MCP Tools Registry

This module manages the registration of all MCP tools.
"""

import logging
import os  # Add this import for environment variables
from typing import Any, Optional, Union

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_tools(mcp_app: FastMCP) -> None:
    """
    Register all MCP tools with the FastMCP application.

    Args:
        mcp_app: The FastMCP application instance
    """
    # Check environment variable to determine registration mode
    env = os.getenv("MCP_ENV", "development").lower()
    logger.info(f"Registering tools in {env} mode")

    # Register MCP Prompts system first
    try:
        from src.prompts.registry import register_prompts

        register_prompts(mcp_app)
        logger.info("MCP Prompts registered successfully")
    except ImportError as e:
        logger.warning(f"Skipping MCP Prompts registration due to import issues: {e}")
    except Exception as e:
        logger.error(f"Failed to register MCP Prompts: {e}")

    logger.info("Registering MCP Tools...")

    # Register core tools (always registered)
    from .core.health import health_check

    @mcp_app.tool()
    async def health_check_tool():
        """Check MCP server health: Qdrant, Ollama, memory usage, system resources."""
        return await health_check()

    # Register auto-configuration tool
    from .core.auto_configuration import get_recommended_configuration

    @mcp_app.tool()
    async def get_auto_configuration(
        directory: str = ".",
        usage_pattern: str = "balanced",
    ):
        """Get optimal MCP configuration recommendations based on system and project analysis.

        Key params: directory(.), usage_pattern(balanced)
        """
        return await get_recommended_configuration(directory, usage_pattern)

    # Register compatibility check tool
    from .core.compatibility_check import run_compatibility_check

    @mcp_app.tool()
    async def check_tool_compatibility():
        """Check backward compatibility of all MCP tools after Wave 7.0 enhancements.

        This tool verifies that existing tool interfaces continue to work correctly
        and that new features don't break existing workflows.

        Returns:
            Dictionary containing comprehensive compatibility report with test results
        """
        return await run_compatibility_check()

    # Register performance monitoring tools
    from .core.performance_monitor import get_performance_dashboard

    @mcp_app.tool()
    async def get_performance_dashboard_tool():
        """Get comprehensive performance dashboard for all MCP tools.

        This tool provides real-time performance monitoring, timeout tracking,
        and system metrics to ensure optimal performance and <15 second responses.

        Returns:
            Dictionary containing performance metrics, active operations, and system status
        """
        return await get_performance_dashboard()

    # Register service health monitoring tools
    from .core.graceful_degradation import get_service_health_status

    @mcp_app.tool()
    async def get_service_health_status_tool():
        """Get comprehensive service health status and error tracking.

        This tool provides information about service degradation levels,
        error patterns, and recovery status for all MCP tools.

        Returns:
            Dictionary containing service health metrics and error summaries
        """
        return await get_service_health_status()

    # Register indexing tools
    from .indexing.index_tools import index_directory as index_directory_impl

    @mcp_app.tool()
    async def index_directory(
        directory: str = ".",
        patterns: list[str] | None = None,
        recursive: bool = True,
        clear_existing: bool = False,
        incremental: bool = False,
        project_name: str | None = None,
    ):
        """Index files in directory with smart detection. Supports incremental indexing.

        Key params: directory(.), recursive(True), clear_existing(False), incremental(False)
        """
        return await index_directory_impl(directory, patterns, recursive, clear_existing, incremental, project_name)

    # Register search tools
    from .indexing.search_tools import (
        analyze_repository_tool as analyze_repository_impl,
    )
    from .indexing.search_tools import (
        check_index_status_tool as check_index_status_impl,
    )
    from .indexing.search_tools import (
        get_file_filtering_stats_tool as get_file_filtering_stats_impl,
    )
    from .indexing.search_tools import search as search_impl

    @mcp_app.tool()
    async def search(
        query: str,
        n_results: int = 5,
        cross_project: bool = False,
        search_mode: str = "hybrid",
        include_context: bool = True,
        context_chunks: int = 1,
        target_projects: list[str] | None = None,
        collection_types: list[str] | None = None,
        multi_modal_mode: str | None = None,
        include_query_analysis: bool = False,
        performance_timeout_seconds: int = 15,
        minimal_output: bool = False,
    ):
        """Search indexed content with semantic, keyword, and hybrid modes.
        English queries recommended for optimal performance.

        Args:
            query: Search query (English preferred)
            n_results: Results to return (default: 5)
            cross_project: Search across all projects (default: False)
            search_mode: "semantic", "keyword", or "hybrid" (default: "hybrid")
            collection_types: Filter by ["code"], ["config"], ["documentation"]
            target_projects: Specific project names to search
            include_context: Include surrounding code context (default: True)
            context_chunks: Context chunks before/after results (0-5, default: 1)
        """
        return await search_impl(
            query,
            n_results,
            cross_project,
            search_mode,
            include_context,
            context_chunks,
            target_projects,
            collection_types,
            multi_modal_mode,
            multi_modal_mode is not None,  # enable_multi_modal
            False,  # enable_manual_mode_selection (deprecated)
            include_query_analysis,
            performance_timeout_seconds,
            minimal_output,
        )

    @mcp_app.tool()
    async def analyze_repository_tool(directory: str = "."):
        """Analyze repository structure: file counts, language breakdown, complexity, indexing recommendations."""
        return await analyze_repository_impl(directory)

    @mcp_app.tool()
    async def get_file_filtering_stats_tool(directory: str = "."):
        """Get detailed statistics about file filtering for debugging
        and optimization.

        This tool shows how many files are excluded by different criteria,
        helping users understand and optimize their .ragignore patterns.

        Args:
            directory: Path to the directory to analyze
                       (default: current directory)

        Returns:
            Detailed breakdown of file filtering statistics including
            exclusion reasons, configuration settings, and recommendations.
        """
        return await get_file_filtering_stats_impl(directory)

    @mcp_app.tool()
    async def check_index_status(directory: str = "."):
        """Check indexing status and provide reindexing recommendations."""
        return await check_index_status_impl(directory)

    # Register multi-modal search tools
    try:
        from .indexing.multi_modal_search_tools import (
            analyze_query_features as analyze_query_features_impl,
        )
        from .indexing.multi_modal_search_tools import (
            get_retrieval_mode_performance as get_retrieval_mode_performance_impl,
        )
        from .indexing.multi_modal_search_tools import (
            multi_modal_search as multi_modal_search_impl,
        )

        @mcp_app.tool()
        async def multi_modal_search(
            query: str,
            n_results: int = 10,
            mode: str | None = None,
            target_projects: list[str] | None = None,
            cross_project: bool = False,
            include_analysis: bool = True,
            include_performance_metrics: bool = False,
            minimal_output: bool = True,  # Added this line
        ):
            """Advanced multi-modal search with LightRAG modes. Provides intelligent
            mode selection and query analysis. English queries strongly recommended.

            Modes:
                - local: Entity-focused retrieval
                - global: Relationship-focused retrieval
                - hybrid: Balanced approach
                - mix: Automatic mode selection

            Args:
                query: Search query (English strongly recommended)
                mode: Retrieval mode or None for auto-selection
                n_results: Results to return (default: 10)
                cross_project: Search across projects (default: False)
                target_projects: Specific project names to search
                include_analysis: Include query analysis (default: True)
                minimal_output: Return minimal output (default: True)
            """
            return await multi_modal_search_impl(
                query=query,
                n_results=n_results,
                mode=mode,
                target_projects=target_projects,
                cross_project=cross_project,
                enable_manual_mode_selection=(mode is not None),
                include_analysis=include_analysis,
                include_performance_metrics=include_performance_metrics,
                minimal_output=minimal_output,  # Pass the new parameter
            )

        @mcp_app.tool()
        async def analyze_query_features(query: str):
            """Analyze query features and recommend optimal retrieval mode."""
            return await analyze_query_features_impl(query)

        @mcp_app.tool()
        async def get_retrieval_mode_performance(
            mode: str | None = None,
            include_comparison: bool = True,
            include_alerts: bool = True,
            include_history: bool = False,
            history_limit: int = 50,
        ):
            """Get performance metrics and analytics for retrieval modes.

            This tool provides comprehensive performance monitoring data
            for the multi-modal retrieval system.

            Args:
                mode: Specific mode to get metrics for ('local', 'global', 'hybrid', 'mix')
                include_comparison: Whether to include mode comparison
                include_alerts: Whether to include active alerts
                include_history: Whether to include query history
                history_limit: Limit for query history (default: 50)

            Returns:
                Dictionary containing performance metrics and analytics
            """
            return await get_retrieval_mode_performance_impl(
                mode,
                include_comparison,
                include_alerts,
                include_history,
                history_limit,
            )

        logger.info("Multi-modal search tools registered successfully")

    except ImportError as e:
        logger.warning(f"Multi-modal search tools not available: {e}")
    except Exception as e:
        logger.error(f"Failed to register multi-modal search tools: {e}")

    # Register project tools (core)
    from .project.project_tools import register_project_tools

    register_project_tools(mcp_app)

    # Register file tools (core)
    from .project.file_tools import register_file_tools

    register_file_tools(mcp_app)

    # Register specific cache tools for production
    from .cache.cache_control import get_cache_health_status
    from .cache.cache_management import (
        clear_all_caches,
        warm_cache_for_project,
    )
    from .cache.cache_optimization import optimize_cache_performance

    @mcp_app.tool()
    async def clear_all_caches_tool(
        reason: str = "manual_invalidation",
        confirm: bool = False,
    ):
        """Clear all caches (DESTRUCTIVE). Requires confirm=True."""
        return await clear_all_caches(reason, confirm)

    @mcp_app.tool()
    async def warm_cache_for_project_tool(
        project_name: str,
        cache_types: list[str] | None = None,
        warmup_strategy: str = "comprehensive",
        max_concurrent: int = 5,
    ):
        """Warm up caches for a specific project with comprehensive
        preloading.

        Args:
            project_name: Name of the project to warm up
            cache_types: Types of caches to warm
                         (embedding, search, project, file, all)
            warmup_strategy: Warmup strategy (comprehensive, selective,
                             recent, critical)
            max_concurrent: Maximum concurrent warming operations

        Returns:
            Dictionary with cache warming results and statistics
        """
        return await warm_cache_for_project(project_name, cache_types, warmup_strategy, max_concurrent)

    @mcp_app.tool()
    async def get_cache_health_status_tool(
        include_detailed_checks: bool = True,
        check_connectivity: bool = True,
        check_performance: bool = True,
    ):
        """Get comprehensive cache health status across all services.

        Args:
            include_detailed_checks: Whether to include detailed health checks
            check_connectivity: Whether to check cache connectivity
            check_performance: Whether to check performance metrics

        Returns:
            Dictionary with comprehensive health status and alerts
        """
        return await get_cache_health_status(include_detailed_checks, check_connectivity, check_performance)

    @mcp_app.tool()
    async def optimize_cache_performance_tool(
        optimization_type: str = "comprehensive",
        apply_changes: bool = False,
        project_name: str | None = None,
    ):
        """Analyze cache performance and provide optimization
        recommendations.

        Args:
            optimization_type: Type of optimization (comprehensive, memory,
                               ttl, connections, hit_rate)
            apply_changes: Whether to automatically apply safe optimizations
            project_name: Optional project name for scoped optimization

        Returns:
            Dictionary with optimization analysis and recommendations
        """
        return await optimize_cache_performance(optimization_type, apply_changes, project_name)

    if env == "development":
        # Register additional tools for development mode
        # Register chunking tools
        from .indexing.chunking_tools import register_chunking_tools

        register_chunking_tools(mcp_app)

        # Register parser tools
        from .indexing.parser_tools import register_parser_tools

        register_parser_tools(mcp_app)

        # Register progress tools
        from .indexing.progress_tools import register_progress_tools

        register_progress_tools(mcp_app)

        # Register all cache management tools
        from .cache.cache_management import (
            debug_cache_key,
            generate_cache_report,
            get_cache_invalidation_stats,
            get_comprehensive_cache_stats,
            get_project_invalidation_policy,
            inspect_cache_state,
            invalidate_chunks,
            manual_invalidate_cache_keys,
            manual_invalidate_cache_pattern,
            manual_invalidate_file_cache,
            manual_invalidate_project_cache,
            preload_embedding_cache,
            preload_search_cache,
            set_project_invalidation_policy,
        )

        @mcp_app.tool()
        async def manual_invalidate_file_cache_tool(
            file_path: str,
            reason: str = "manual_invalidation",
            cascade: bool = True,
            use_partial: bool = True,
            old_content: str | None = None,
            new_content: str | None = None,
            project_name: str | None = None,
        ):
            """Manually invalidate cache entries for a specific file.

            Args:
                file_path: Path to the file to invalidate
                reason: Reason for invalidation
                       (manual_invalidation, file_modified, file_deleted,
                       content_changed, metadata_changed)
                cascade: Whether to cascade invalidation to dependent caches
                use_partial: Whether to use partial invalidation if content
                            is provided
                old_content: Previous content of the file
                            (for partial invalidation)
                new_content: New content of the file
                            (for partial invalidation)
                project_name: Project name for scoped invalidation

            Returns:
                Dictionary with invalidation results and optimization
                statistics
            """
            return await manual_invalidate_file_cache(file_path, reason, cascade, use_partial, old_content, new_content, project_name)

        @mcp_app.tool()
        async def manual_invalidate_project_cache_tool(
            project_name: str,
            reason: str = "manual_invalidation",
            invalidation_scope: str = "cascade",
            strategy: str = "immediate",
        ):
            """Manually invalidate all cache entries for a project.

            Args:
                project_name: Name of the project to invalidate
                reason: Reason for invalidation
                       (manual_invalidation, project_changed,
                       dependency_changed, system_upgrade)
                invalidation_scope: Scope of invalidation
                                   (file_only, project_wide, cascade,
                                   conservative, aggressive)
                strategy: Invalidation strategy (immediate, lazy, batch,
                          scheduled)

            Returns:
                Dictionary with invalidation results and statistics
            """
            return await manual_invalidate_project_cache(project_name, reason, invalidation_scope, strategy)

        @mcp_app.tool()
        async def manual_invalidate_cache_keys_tool(
            cache_keys: list[str],
            reason: str = "manual_invalidation",
            cascade: bool = False,
        ):
            """Manually invalidate specific cache keys.

            Args:
                cache_keys: List of cache keys to invalidate
                reason: Reason for invalidation
                       (manual_invalidation, dependency_changed,
                       cache_corruption, ttl_expired)
                cascade: Whether to cascade invalidation to dependent caches

            Returns:
                Dictionary with invalidation results and statistics
            """
            return await manual_invalidate_cache_keys(cache_keys, reason, cascade)

        @mcp_app.tool()
        async def manual_invalidate_cache_pattern_tool(
            pattern: str,
            reason: str = "manual_invalidation",
        ):
            """Manually invalidate cache keys matching a pattern.

            Args:
                pattern: Pattern to match cache keys (supports wildcards)
                reason: Reason for invalidation
                       (manual_invalidation, dependency_changed,
                       cache_corruption, system_upgrade)

            Returns:
                Dictionary with invalidation results and statistics
            """
            return await manual_invalidate_cache_pattern(pattern, reason)

        @mcp_app.tool()
        async def get_cache_invalidation_stats_tool():
            """Get comprehensive cache invalidation statistics and metrics.

            Returns:
                Dictionary with detailed invalidation statistics, recent
                events, and monitoring info
            """
            return await get_cache_invalidation_stats()

        @mcp_app.tool()
        async def get_project_invalidation_policy_tool(project_name: str):
            """Get invalidation policy for a specific project.

            Args:
                project_name: Name of the project

            Returns:
                Dictionary with project invalidation policy details and
                monitoring status
            """
            return await get_project_invalidation_policy(project_name)

        @mcp_app.tool()
        async def set_project_invalidation_policy_tool(
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
        ):
            """Set or update invalidation policy for a specific project.

            Args:
                project_name: Name of the project
                scope: Invalidation scope (file_only, project_wide, cascade,
                       conservative, aggressive)
                strategy: Invalidation strategy (immediate, lazy, batch,
                          scheduled)
                batch_threshold: Number of changes to trigger batch processing
                delay_seconds: Delay before processing invalidation
                file_patterns: File patterns to monitor
                               (default: common code files)
                exclude_patterns: Patterns to exclude from monitoring
                                  (default: temp/cache files)
                invalidate_embeddings: Whether to invalidate embedding caches
                invalidate_search: Whether to invalidate search caches
                invalidate_project: Whether to invalidate project caches
                invalidate_file: Whether to invalidate file caches
                max_concurrent_invalidations: Maximum concurrent invalidations
                cascade_depth_limit: Maximum cascade depth

            Returns:
                Dictionary with policy creation/update results
            """
            return await set_project_invalidation_policy(
                project_name,
                scope,
                strategy,
                batch_threshold,
                delay_seconds,
                file_patterns,
                exclude_patterns,
                invalidate_embeddings,
                invalidate_search,
                invalidate_project,
                invalidate_file,
                max_concurrent_invalidations,
                cascade_depth_limit,
            )

        @mcp_app.tool()
        async def invalidate_chunks_tool(
            file_path: str,
            chunk_ids: list[str],
            reason: str = "chunk_modified",
        ):
            """Invalidate specific chunks within a file.

            Args:
                file_path: Path to the file containing the chunks
                chunk_ids: List of chunk IDs to invalidate
                reason: Reason for chunk invalidation (chunk_modified,
                       manual_invalidation, content_changed)

            Returns:
                Dictionary with chunk invalidation results and statistics
            """
            return await invalidate_chunks(file_path, chunk_ids, reason)

        @mcp_app.tool()
        async def inspect_cache_state_tool(
            cache_type: str = "all",
            include_content: bool = False,
            max_entries: int = 100,
        ):
            """Inspect the current state of cache services with detailed
            debugging information.

            Args:
                cache_type: Type of cache to inspect (all, embedding, search,
                            project, file, l1, l2)
                include_content: Whether to include actual cache content in
                                 response
                max_entries: Maximum number of cache entries to include

            Returns:
                Dictionary with detailed cache state information and service
                health
            """
            return await inspect_cache_state(cache_type, include_content, max_entries)

        @mcp_app.tool()
        async def debug_cache_key_tool(
            cache_key: str,
            cache_type: str = "all",
        ):
            """Debug a specific cache key across all cache services.

            Args:
                cache_key: The cache key to debug
                cache_type: Type of cache to check (all, embedding, search,
                            project, file)

            Returns:
                Dictionary with debugging information for the cache key
            """
            return await debug_cache_key(cache_key, cache_type)

        @mcp_app.tool()
        async def preload_embedding_cache_tool(
            queries: list[str],
            project_name: str | None = None,
            model_name: str | None = None,
        ):
            """Preload embedding cache with specific queries or content.

            Args:
                queries: List of queries/content to preload embeddings for
                project_name: Optional project name for scoped preloading
                model_name: Optional specific embedding model name

            Returns:
                Dictionary with preloading results and statistics
            """
            return await preload_embedding_cache(queries, project_name, model_name)

        @mcp_app.tool()
        async def preload_search_cache_tool(
            search_queries: list[dict],
            project_name: str | None = None,
        ):
            """Preload search result cache with specific search queries.

            Args:
                search_queries: List of search query dictionaries with
                               parameters
                project_name: Optional project name for scoped preloading

            Returns:
                Dictionary with search cache preloading results
            """
            return await preload_search_cache(search_queries, project_name)

        @mcp_app.tool()
        async def get_comprehensive_cache_stats_tool(
            project_name: str | None = None,
            include_historical: bool = False,
            time_range_hours: int = 24,
        ):
            """Get comprehensive cache statistics across all cache services.

            Args:
                project_name: Optional project name for scoped statistics
                include_historical: Whether to include historical data
                time_range_hours: Time range for historical data (hours)

            Returns:
                Dictionary with comprehensive cache statistics and aggregated
                metrics
            """
            return await get_comprehensive_cache_stats(project_name, include_historical, time_range_hours)

        @mcp_app.tool()
        async def generate_cache_report_tool(
            report_type: str = "comprehensive",
            project_name: str | None = None,
            export_format: str = "json",
        ):
            """Generate a comprehensive cache performance report.

            Args:
                report_type: Type of report (comprehensive, performance,
                             health, optimization)
                project_name: Optional project name for scoped reporting
                export_format: Format for export (json, markdown, csv)

            Returns:
                Dictionary with cache report data and optimization
                recommendations
            """
            return await generate_cache_report(report_type, project_name, export_format)

        # Register cache control interface tools
        from .cache.cache_control import (
            configure_cache_alerts,
            export_cache_configuration,
            get_cache_alerts,
            get_cache_configuration,
            import_cache_configuration,
            update_cache_configuration,
        )

        @mcp_app.tool()
        async def get_cache_configuration_tool(
            config_type: str = "all",
            export_format: str = "json",
        ):
            """Get current cache configuration across all services.

            Args:
                config_type: Type of configuration to retrieve (all, redis,
                             memory, ttl, limits, security)
                export_format: Format for export (json, yaml, env)

            Returns:
                Dictionary with cache configuration details and formatted
                output
            """
            return await get_cache_configuration(config_type, export_format)

        @mcp_app.tool()
        async def update_cache_configuration_tool(
            config_updates: dict[str, Any],
            validate_only: bool = False,
            restart_services: bool = False,
        ):
            """Update cache configuration settings.

            Args:
                config_updates: Dictionary of configuration updates to apply
                validate_only: Only validate changes without applying them
                restart_services: Whether to restart cache services after
                                  updates

            Returns:
                Dictionary with update results and validation status
            """
            return await update_cache_configuration(config_updates, validate_only, restart_services)

        @mcp_app.tool()
        async def export_cache_configuration_tool(
            export_path: str,
            config_type: str = "all",
            include_sensitive: bool = False,
        ):
            """Export cache configuration to file.

            Args:
                export_path: Path to export configuration file
                config_type: Type of configuration to export (all, redis,
                             memory, ttl, limits)
                include_sensitive: Whether to include sensitive information
                                   (passwords, keys)

            Returns:
                Dictionary with export results and file information
            """
            return await export_cache_configuration(export_path, config_type, include_sensitive)

        @mcp_app.tool()
        async def import_cache_configuration_tool(
            import_path: str,
            config_type: str = "all",
            validate_only: bool = True,
            backup_current: bool = True,
        ):
            """Import cache configuration from file.

            Args:
                import_path: Path to configuration file to import
                config_type: Type of configuration to import (all, redis,
                             memory, ttl, limits)
                validate_only: Only validate without applying changes
                backup_current: Create backup of current configuration

            Returns:
                Dictionary with import results and validation status
            """
            return await import_cache_configuration(import_path, config_type, validate_only, backup_current)

        @mcp_app.tool()
        async def configure_cache_alerts_tool(
            alert_config: dict[str, Any],
            enable_alerts: bool = True,
        ):
            """Configure cache monitoring alerts and thresholds.

            Args:
                alert_config: Alert configuration including thresholds and
                              notification settings
                enable_alerts: Whether to enable alert monitoring

            Returns:
                Dictionary with alert configuration results
            """
            return await configure_cache_alerts(alert_config, enable_alerts)

        @mcp_app.tool()
        async def get_cache_alerts_tool(
            severity_filter: str = "all",
            time_range_hours: int = 24,
            service_filter: str = "all",
        ):
            """Get recent cache alerts and notifications.

            Args:
                severity_filter: Filter by severity (all, error, warning, info)
                time_range_hours: Time range for alert history (hours)
                service_filter: Filter by service (all, embedding, search,
                                project, file)

            Returns:
                Dictionary with recent alerts and statistics
            """
            return await get_cache_alerts(severity_filter, time_range_hours, service_filter)

        # Register cache optimization tools
        from .cache.cache_optimization import (
            backup_cache_data,
            get_migration_status,
            migrate_cache_data,
            restore_cache_data,
        )

        @mcp_app.tool()
        async def backup_cache_data_tool(
            backup_path: str,
            backup_type: str = "incremental",
            include_services: list[str] = None,
            compression: bool = True,
        ):
            """Create a backup of cache data and configuration.

            Args:
                backup_path: Path where backup will be created
                backup_type: Type of backup (full, incremental,
                             configuration_only)
                include_services: List of services to backup (default: all)
                compression: Whether to compress backup data

            Returns:
                Dictionary with backup operation results
            """
            return await backup_cache_data(backup_path, backup_type, include_services, compression)

        @mcp_app.tool()
        async def restore_cache_data_tool(
            backup_path: str,
            restore_type: str = "full",
            target_services: list[str] = None,
            validate_only: bool = False,
        ):
            """Restore cache data from backup.

            Args:
                backup_path: Path to backup file or directory
                restore_type: Type of restore (full, configuration_only,
                              data_only)
                target_services: List of services to restore (default: all
                                 from backup)
                validate_only: Only validate backup without restoring

            Returns:
                Dictionary with restore operation results
            """
            return await restore_cache_data(backup_path, restore_type, target_services, validate_only)

        @mcp_app.tool()
        async def migrate_cache_data_tool(
            migration_type: str,
            source_config: dict[str, Any] = None,
            target_config: dict[str, Any] = None,
            dry_run: bool = True,
        ):
            """Migrate cache data between different configurations or
            versions.

            Args:
                migration_type: Type of migration (redis_upgrade,
                                schema_migration, configuration_migration)
                source_config: Source configuration for migration
                target_config: Target configuration for migration
                dry_run: Whether to perform a dry run without making changes

            Returns:
                Dictionary with migration operation results
            """
            return await migrate_cache_data(migration_type, source_config, target_config, dry_run)

        @mcp_app.tool()
        async def get_migration_status_tool(
            migration_id: str = None,
        ):
            """Get status of ongoing or completed migrations.

            Args:
                migration_id: Optional specific migration ID to check

            Returns:
                Dictionary with migration status information
            """
            return await get_migration_status(migration_id)

        # Register file monitoring tools
        from .cache.file_monitoring_tools import register_file_monitoring_tools

        register_file_monitoring_tools(mcp_app)

        # Register cascade invalidation tools
        from .cache.cascade_invalidation_tools import register_cascade_invalidation_tools

        register_cascade_invalidation_tools(mcp_app)

    logger.info("All MCP Tools registered successfully")
