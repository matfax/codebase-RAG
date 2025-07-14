"""Parser diagnostic tools for Tree-sitter health monitoring.

This module provides tools for diagnosing and monitoring the health of Tree-sitter parsers
used in intelligent code chunking.
"""

import logging
import traceback
from datetime import datetime
from typing import Any

from mcp.server.fastmcp import FastMCP
from src.tools.core.error_utils import handle_tool_error, log_tool_usage
from src.tools.core.errors import ParserError

# Configure logging
logger = logging.getLogger(__name__)


def diagnose_parser_health(comprehensive: bool = False, language: str | None = None) -> dict[str, Any]:
    """
    Diagnose Tree-sitter parser health and functionality.

    This tool verifies parser installations, tests parsing functionality,
    and identifies potential issues with the intelligent chunking system.

    Args:
        comprehensive: Run comprehensive diagnostics (slower but more detailed)
        language: Test specific language only (e.g., "python", "javascript")

    Returns:
        Dictionary with diagnostic results and health recommendations
    """
    with log_tool_usage("diagnose_parser_health", {"comprehensive": comprehensive, "language": language}):
        try:
            from src.utils.parser_diagnostics import parser_diagnostics

            if language:
                # Test specific language
                logger.info(f"Running parser diagnostics for {language}")
                test_result = parser_diagnostics.test_specific_language(language)

                result = {
                    "test_type": "specific_language",
                    "language": language,
                    "success": test_result.success,
                    "parse_time_ms": test_result.parse_time_ms,
                    "node_count": test_result.node_count,
                    "error_count": test_result.error_count,
                    "timestamp": datetime.now().isoformat(),
                }

                if test_result.error_message:
                    result["error_message"] = test_result.error_message

                if test_result.parsed_content:
                    result["sample_parsed_content"] = test_result.parsed_content

                logger.info(f"Language test completed: {language} - " f"{'Success' if test_result.success else 'Failed'}")

                return result

            elif comprehensive:
                # Run comprehensive diagnostics
                logger.info("Running comprehensive parser diagnostics")
                health_report = parser_diagnostics.run_comprehensive_diagnostics()

                # Generate human-readable report
                diagnostic_report = parser_diagnostics.generate_diagnostic_report(health_report)

                result = {
                    "test_type": "comprehensive",
                    "health_score": health_report.overall_health_score(),
                    "health_status": health_report.health_status(),
                    "tree_sitter_available": health_report.tree_sitter_available,
                    "tree_sitter_version": health_report.tree_sitter_version,
                    "total_languages": health_report.total_languages,
                    "installed_languages": health_report.installed_languages,
                    "failed_languages": health_report.failed_languages,
                    "parsing_tests": {
                        lang: {
                            "success": test.success,
                            "parse_time_ms": test.parse_time_ms,
                            "node_count": test.node_count,
                            "error_count": test.error_count,
                            "error_message": test.error_message,
                        }
                        for lang, test in health_report.parsing_tests.items()
                    },
                    "performance_metrics": {
                        "average_parse_time_ms": health_report.average_parse_time_ms,
                        "fastest_language": health_report.fastest_language,
                        "slowest_language": health_report.slowest_language,
                    },
                    "critical_issues": health_report.critical_issues,
                    "warnings": health_report.warnings,
                    "recommendations": health_report.recommendations,
                    "test_duration_seconds": health_report.test_duration_seconds,
                    "diagnostic_report": diagnostic_report,
                    "timestamp": datetime.now().isoformat(),
                }

                logger.info(
                    f"Comprehensive diagnostics completed - "
                    f"Health Score: {health_report.overall_health_score():.1f}/100 "
                    f"({health_report.health_status()})"
                )

                return result

            else:
                # Quick health check
                logger.info("Running quick parser health check")
                quick_result = parser_diagnostics.run_quick_health_check()

                quick_result["test_type"] = "quick_check"

                logger.info(f"Quick health check completed: {quick_result.get('status', 'unknown')}")

                return quick_result

        except Exception as e:
            error_msg = f"Parser diagnostics failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ParserError(error_msg) from e


def diagnose_parser_health_sync(comprehensive: bool = False, language: str | None = None) -> dict[str, Any]:
    """Synchronous wrapper for diagnose_parser_health."""
    return handle_tool_error(diagnose_parser_health, comprehensive=comprehensive, language=language)


def register_parser_tools(mcp_app: FastMCP):
    """Register parser diagnostic MCP tools."""

    @mcp_app.tool()
    def diagnose_parser_health_tool(comprehensive: bool = False, language: str | None = None) -> dict[str, Any]:
        """
        Diagnose Tree-sitter parser health and functionality.

        This tool verifies parser installations, tests parsing functionality,
        and identifies potential issues with the intelligent chunking system.

        Args:
            comprehensive: Run comprehensive diagnostics (slower but more detailed)
            language: Test specific language only (e.g., "python", "javascript")

        Returns:
            Dictionary with diagnostic results and health recommendations
        """
        return diagnose_parser_health_sync(comprehensive, language)
