"""MCP Tools Registry

This module manages the registration of all MCP tools.
"""

import logging

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)


def register_tools(mcp_app: FastMCP) -> None:
    """
    Register all MCP tools with the FastMCP application.

    Args:
        mcp_app: The FastMCP application instance
    """
    # Register MCP Prompts system first
    try:
        from prompts import register_prompts

        register_prompts(mcp_app)
        logger.info("MCP Prompts system registered successfully")
    except Exception as e:
        logger.error(f"Failed to register MCP Prompts system: {e}")
        # Continue without prompts if there's an error

    logger.info("Registering MCP Tools...")

    # Register core tools
    from .core.health import health_check

    @mcp_app.tool()
    async def health_check_tool():
        """Check the health of the MCP server and its dependencies.

        Checks the status of:
        - Qdrant database connectivity
        - Ollama service availability
        - Memory usage and system resources

        Returns detailed health information with warnings and issues.
        """
        return await health_check()

    # Register indexing tools
    from .indexing.index_tools import index_directory as index_directory_impl

    @mcp_app.tool()
    async def index_directory(
        directory: str = ".",
        patterns: list[str] = None,
        recursive: bool = True,
        clear_existing: bool = False,
        incremental: bool = False,
        project_name: str = None,
    ):
        """Index files in a directory with smart existing data detection and time estimation.

        Args:
            directory: Directory to index (default: current directory)
            patterns: File patterns to include (default: common code file types)
            recursive: Whether to index subdirectories (default: True)
            clear_existing: Whether to clear existing indexed data (default: False)
                          If False and existing data is found, returns recommendations instead of indexing
            incremental: Whether to use incremental indexing (only process changed files) (default: False)
            project_name: Optional custom project name for collections (default: auto-detect)

        Returns:
            Dictionary with indexing results, time estimates, or recommendations for existing data
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
        target_projects: list[str] = None,
    ):
        """Search indexed content using natural language queries.

        This tool provides function-level precision search with intelligent chunking,
        supporting multiple search modes and context expansion for better code understanding.

        Args:
            query: Natural language search query
            n_results: Number of results to return (1-100, default: 5)
            cross_project: Whether to search across all projects (default: False - current project only)
            search_mode: Search strategy - "semantic", "keyword", or "hybrid" (default: "hybrid")
            include_context: Whether to include surrounding code context (default: True)
            context_chunks: Number of context chunks to include before/after results (0-5, default: 1)
            target_projects: List of specific project names to search in (optional)

        Returns:
            Dictionary containing search results with metadata, scores, and context
        """
        return await search_impl(
            query,
            n_results,
            cross_project,
            search_mode,
            include_context,
            context_chunks,
            target_projects,
        )

    @mcp_app.tool()
    async def analyze_repository_tool(directory: str = "."):
        """Analyze repository structure and provide detailed statistics for indexing planning.

        This tool helps assess repository complexity, file distribution, and provides
        recommendations for optimal indexing strategies.

        Args:
            directory: Path to the directory to analyze (default: current directory)

        Returns:
            Detailed analysis including file counts, size distribution, language breakdown,
            complexity assessment, and indexing recommendations.
        """
        return await analyze_repository_impl(directory)

    @mcp_app.tool()
    async def get_file_filtering_stats_tool(directory: str = "."):
        """Get detailed statistics about file filtering for debugging and optimization.

        This tool shows how many files are excluded by different criteria,
        helping users understand and optimize their .ragignore patterns.

        Args:
            directory: Path to the directory to analyze (default: current directory)

        Returns:
            Detailed breakdown of file filtering statistics including exclusion reasons,
            configuration settings, and recommendations.
        """
        return await get_file_filtering_stats_impl(directory)

    @mcp_app.tool()
    async def check_index_status(directory: str = "."):
        """Check if a directory already has indexed data and provide recommendations.

        This tool helps users understand the current indexing state and make informed
        decisions about whether to reindex or use existing data.

        Args:
            directory: Path to the directory to check (default: current directory)

        Returns:
            Status information and recommendations for the indexed data
        """
        return await check_index_status_impl(directory)

    # Register chunking tools
    from .indexing.chunking_tools import register_chunking_tools

    register_chunking_tools(mcp_app)

    # Register parser tools
    from .indexing.parser_tools import register_parser_tools

    register_parser_tools(mcp_app)

    # Register progress tools
    from .indexing.progress_tools import register_progress_tools

    register_progress_tools(mcp_app)

    # Register project tools
    from .project.project_tools import register_project_tools

    register_project_tools(mcp_app)

    # Register file tools
    from .project.file_tools import register_file_tools

    register_file_tools(mcp_app)

    # Register cache management tools
    from .cache.cache_management import (
        clear_all_caches,
        get_cache_invalidation_stats,
        get_project_invalidation_policy,
        invalidate_chunks,
        manual_invalidate_cache_keys,
        manual_invalidate_cache_pattern,
        manual_invalidate_file_cache,
        manual_invalidate_project_cache,
        set_project_invalidation_policy,
    )

    @mcp_app.tool()
    async def manual_invalidate_file_cache_tool(
        file_path: str,
        reason: str = "manual_invalidation",
        cascade: bool = True,
        use_partial: bool = True,
        old_content: str = None,
        new_content: str = None,
        project_name: str = None,
    ):
        """Manually invalidate cache entries for a specific file.

        Args:
            file_path: Path to the file to invalidate
            reason: Reason for invalidation (manual_invalidation, file_modified, file_deleted, content_changed, metadata_changed)
            cascade: Whether to cascade invalidation to dependent caches
            use_partial: Whether to use partial invalidation if content is provided
            old_content: Previous content of the file (for partial invalidation)
            new_content: New content of the file (for partial invalidation)
            project_name: Project name for scoped invalidation

        Returns:
            Dictionary with invalidation results and optimization statistics
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
            reason: Reason for invalidation (manual_invalidation, project_changed, dependency_changed, system_upgrade)
            invalidation_scope: Scope of invalidation (file_only, project_wide, cascade, conservative, aggressive)
            strategy: Invalidation strategy (immediate, lazy, batch, scheduled)

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
            reason: Reason for invalidation (manual_invalidation, dependency_changed, cache_corruption, ttl_expired)
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
            reason: Reason for invalidation (manual_invalidation, dependency_changed, cache_corruption, system_upgrade)

        Returns:
            Dictionary with invalidation results and statistics
        """
        return await manual_invalidate_cache_pattern(pattern, reason)

    @mcp_app.tool()
    async def clear_all_caches_tool(
        reason: str = "manual_invalidation",
        confirm: bool = False,
    ):
        """Clear all caches across all services (DESTRUCTIVE OPERATION).

        Args:
            reason: Reason for clearing all caches (manual_invalidation, system_upgrade, cache_corruption)
            confirm: Must be True to confirm this destructive operation

        Returns:
            Dictionary with clearing results and statistics
        """
        return await clear_all_caches(reason, confirm)

    @mcp_app.tool()
    async def get_cache_invalidation_stats_tool():
        """Get comprehensive cache invalidation statistics and metrics.

        Returns:
            Dictionary with detailed invalidation statistics, recent events, and monitoring info
        """
        return await get_cache_invalidation_stats()

    @mcp_app.tool()
    async def get_project_invalidation_policy_tool(project_name: str):
        """Get invalidation policy for a specific project.

        Args:
            project_name: Name of the project

        Returns:
            Dictionary with project invalidation policy details and monitoring status
        """
        return await get_project_invalidation_policy(project_name)

    @mcp_app.tool()
    async def set_project_invalidation_policy_tool(
        project_name: str,
        scope: str = "cascade",
        strategy: str = "immediate",
        batch_threshold: int = 5,
        delay_seconds: float = 0.0,
        file_patterns: list[str] = None,
        exclude_patterns: list[str] = None,
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
            scope: Invalidation scope (file_only, project_wide, cascade, conservative, aggressive)
            strategy: Invalidation strategy (immediate, lazy, batch, scheduled)
            batch_threshold: Number of changes to trigger batch processing
            delay_seconds: Delay before processing invalidation
            file_patterns: File patterns to monitor (default: common code files)
            exclude_patterns: Patterns to exclude from monitoring (default: temp/cache files)
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
            reason: Reason for chunk invalidation (chunk_modified, manual_invalidation, content_changed)

        Returns:
            Dictionary with chunk invalidation results and statistics
        """
        return await invalidate_chunks(file_path, chunk_ids, reason)

    # Register file monitoring tools
    from .cache.file_monitoring_tools import register_file_monitoring_tools

    register_file_monitoring_tools(mcp_app)

    # Register cascade invalidation tools
    from .cache.cascade_invalidation_tools import register_cascade_invalidation_tools

    register_cascade_invalidation_tools(mcp_app)

    logger.info("All MCP Tools registered successfully")
