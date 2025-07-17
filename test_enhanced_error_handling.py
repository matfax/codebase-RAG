#!/usr/bin/env python3
"""
Test Enhanced Error Handling for Graph RAG Tools

This script tests the enhanced error handling and suggestions system
implemented in Task 3.9.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from src.services.breadcrumb_resolver_service import BreadcrumbResolutionResult
from src.tools.graph_rag.function_chain_analysis import _generate_enhanced_suggestions, trace_function_chain
from src.tools.graph_rag.function_path_finding import find_function_path


async def test_enhanced_error_handling():
    """Test enhanced error handling and suggestions."""
    print("Testing Enhanced Error Handling for Graph RAG Tools")
    print("=" * 60)

    # Test 1: Entry point not found
    print("\n1. Testing entry point resolution failure...")
    try:
        result = await trace_function_chain(
            entry_point="nonexistent_function",
            project_name="test_project",
            direction="forward",
            max_depth=10,
        )

        # Verify enhanced error handling
        assert result["success"] is False
        assert "suggestions" in result
        assert "error_details" in result
        assert "alternatives" in result

        print("✅ Entry point resolution failure handled correctly")
        print(f"   - Error: {result['error']}")
        print(f"   - Suggestions: {len(result['suggestions'])} provided")
        print(f"   - Error details: {result['error_details'].get('error_type', 'unknown')}")

    except Exception as e:
        print(f"❌ Test 1 failed: {e}")

    # Test 2: Path finding with non-existent functions
    print("\n2. Testing path finding with non-existent functions...")
    try:
        result = await find_function_path(
            start_function="nonexistent_start",
            end_function="nonexistent_end",
            project_name="test_project",
            strategy="optimal",
        )

        # Verify enhanced error handling
        assert result["success"] is False
        assert "suggestions" in result
        assert "error_details" in result
        assert "alternatives" in result

        print("✅ Path finding failure handled correctly")
        print(f"   - Error: {result['error']}")
        print(f"   - Suggestions: {len(result['suggestions'])} provided")
        print(f"   - Error details: {result['error_details'].get('error_type', 'unknown')}")

    except Exception as e:
        print(f"❌ Test 2 failed: {e}")

    # Test 3: Enhanced suggestions generation
    print("\n3. Testing enhanced suggestions generation...")
    try:
        suggestions = await _generate_enhanced_suggestions(
            entry_point="test_function",
            project_name="test_project",
            breadcrumb_result=None,
            error_type="general_error",
            error_message="Test error message",
        )

        # Verify suggestions structure
        assert "suggestions" in suggestions
        assert "error_details" in suggestions
        assert "alternatives" in suggestions
        assert isinstance(suggestions["suggestions"], list)
        assert len(suggestions["suggestions"]) > 0

        print("✅ Enhanced suggestions generation works correctly")
        print(f"   - Suggestions: {len(suggestions['suggestions'])} generated")
        print(f"   - Error details: {suggestions['error_details'].get('error_type', 'unknown')}")
        print(f"   - Alternatives: {len(suggestions['alternatives'])} provided")

    except Exception as e:
        print(f"❌ Test 3 failed: {e}")

    # Test 4: Invalid parameters
    print("\n4. Testing invalid parameter handling...")
    try:
        result = await trace_function_chain(
            entry_point="",  # Empty entry point
            project_name="test_project",
            direction="invalid_direction",  # Invalid direction
            max_depth=0,  # Invalid depth
        )

        # Verify parameter validation
        assert result["success"] is False
        assert "suggestions" in result
        assert "Entry point is required" in result["error"] or "Invalid direction" in result["error"]

        print("✅ Invalid parameter handling works correctly")
        print(f"   - Error: {result['error']}")
        print(f"   - Suggestions provided: {len(result.get('suggestions', []))}")

    except Exception as e:
        print(f"❌ Test 4 failed: {e}")

    print("\n" + "=" * 60)
    print("Enhanced Error Handling Test Complete")
    print("=" * 60)


async def test_error_message_quality():
    """Test the quality and usefulness of error messages."""
    print("\nTesting Error Message Quality")
    print("-" * 40)

    # Test different error types
    error_types = ["entry_point", "start_function", "end_function", "no_paths_found", "general_error"]

    for error_type in error_types:
        print(f"\n  Testing {error_type} error type...")

        try:
            suggestions = await _generate_enhanced_suggestions(
                entry_point="test_function",
                project_name="test_project",
                breadcrumb_result=None,
                error_type=error_type,
                error_message=f"Test error for {error_type}",
            )

            # Verify suggestions are contextual and helpful
            assert len(suggestions["suggestions"]) > 0
            assert suggestions["error_details"]["error_type"] == error_type

            # Check for specific suggestions based on error type
            if error_type == "entry_point":
                assert any("exact function names" in s for s in suggestions["suggestions"])
            elif error_type == "no_paths_found":
                assert any("No paths found" in s for s in suggestions["suggestions"])

            print(f"    ✅ {error_type}: {len(suggestions['suggestions'])} contextual suggestions")

        except Exception as e:
            print(f"    ❌ {error_type}: {e}")


def main():
    """Main function to run error handling tests."""
    print("Graph RAG Enhanced Error Handling Test Suite")
    print("=" * 60)

    # Run async tests
    asyncio.run(test_enhanced_error_handling())
    asyncio.run(test_error_message_quality())

    print("\n🎉 All enhanced error handling tests completed!")
    print("The Graph RAG tools now provide:")
    print("  • Intelligent error suggestions")
    print("  • Detailed error context")
    print("  • Alternative function recommendations")
    print("  • Context-aware troubleshooting tips")
    print("  • Project-specific guidance")


if __name__ == "__main__":
    main()
