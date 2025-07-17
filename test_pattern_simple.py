#!/usr/bin/env python3
"""
Simplified test for pattern matching functionality.
"""

import re
from typing import Any, Dict, List


# Copy the pattern matching functions for testing
def _matches_scope_pattern(breadcrumb: str, pattern: str) -> bool:
    """Check if a breadcrumb matches the scope pattern."""
    if not breadcrumb or not pattern:
        return False

    # "*" matches everything
    if pattern == "*":
        return True

    # Exact match
    if pattern == breadcrumb:
        return True

    # Convert pattern to regex for advanced matching
    regex_pattern = _convert_pattern_to_regex(pattern)

    try:
        return bool(re.match(regex_pattern, breadcrumb))
    except Exception:
        # Fallback to simple matching if regex fails
        return _simple_pattern_match(breadcrumb, pattern)


def _convert_pattern_to_regex(pattern: str) -> str:
    """Convert a breadcrumb pattern to a regex pattern."""
    # Escape special regex characters except * and ?
    escaped = re.escape(pattern)

    # Replace escaped wildcards with regex equivalents
    regex_pattern = escaped.replace(r"\*", ".*")  # * matches any sequence
    regex_pattern = regex_pattern.replace(r"\?", ".")  # ? matches any single character

    # Add anchors to match the full string
    regex_pattern = f"^{regex_pattern}$"

    return regex_pattern


def _simple_pattern_match(breadcrumb: str, pattern: str) -> bool:
    """Simple pattern matching fallback implementation."""
    # Split pattern by wildcards
    parts = pattern.split("*")

    if len(parts) == 1:
        # No wildcards, exact match
        return breadcrumb == pattern

    # Check if breadcrumb matches all parts in sequence
    current_pos = 0

    for i, part in enumerate(parts):
        if not part:  # Empty part (consecutive wildcards)
            continue

        if i == 0:
            # First part - must match at the beginning
            if not breadcrumb.startswith(part):
                return False
            current_pos = len(part)
        elif i == len(parts) - 1:
            # Last part - must match at the end
            if not breadcrumb.endswith(part):
                return False
            # Check if there's enough space for this part
            if current_pos > len(breadcrumb) - len(part):
                return False
        else:
            # Middle part - must be found after current position
            pos = breadcrumb.find(part, current_pos)
            if pos == -1:
                return False
            current_pos = pos + len(part)

    return True


def test_pattern_matching():
    """Test pattern matching functionality."""
    print("🧪 Testing Pattern Matching for Wave 4.0 Subtask 4.2")
    print("=" * 60)

    test_cases = [
        # (breadcrumb, pattern, expected_result, description)
        ("api.user.get_user", "*", True, "Universal wildcard"),
        ("api.user.get_user", "api.*", True, "Prefix matching"),
        ("api.user.get_user", "core.*", False, "Prefix mismatch"),
        ("api.user.get_user", "*.get_user", True, "Suffix matching"),
        ("api.user.get_user", "*.get_product", False, "Suffix mismatch"),
        ("api.user.get_user", "api.*.get_user", True, "Middle wildcard"),
        ("api.user.get_user", "api.*.get_product", False, "Middle wildcard mismatch"),
        ("MyClass.method1", "MyClass.*", True, "Class method matching"),
        ("MyClass.method1", "YourClass.*", False, "Class method mismatch"),
        ("user.test_create", "*.test_*", True, "Multiple wildcards"),
        ("user.create", "*.test_*", False, "Multiple wildcards mismatch"),
        ("core.user.database.save", "core.*.database.*", True, "Complex pattern"),
        ("core.user.service.save", "core.*.database.*", False, "Complex pattern mismatch"),
        ("api.user.get_user", "api.user.get_user", True, "Exact match"),
        ("api.user.get_user", "api.user.get_product", False, "Exact mismatch"),
    ]

    passed = 0
    failed = 0

    for breadcrumb, pattern, expected, description in test_cases:
        result = _matches_scope_pattern(breadcrumb, pattern)
        if result == expected:
            print(f"✅ PASS: {description}")
            print(f"   '{breadcrumb}' matches '{pattern}' -> {result}")
            passed += 1
        else:
            print(f"❌ FAIL: {description}")
            print(f"   '{breadcrumb}' matches '{pattern}' -> {result} (expected {expected})")
            failed += 1

    print(f"\nPattern matching tests: {passed} passed, {failed} failed")

    # Test regex conversion
    print("\nTesting regex conversion...")
    regex_tests = [
        ("*", "^.*$"),
        ("api.*", "^api\\..*$"),
        ("*.handler", "^.*\\.handler$"),
        ("api.*.get_*", "^api\\..*\\.get_.*$"),
    ]

    for pattern, expected_regex in regex_tests:
        result = _convert_pattern_to_regex(pattern)
        if result == expected_regex:
            print(f"✅ PASS: '{pattern}' -> '{result}'")
        else:
            print(f"❌ FAIL: '{pattern}' -> '{result}' (expected '{expected_regex}')")

    print("\n" + "=" * 60)
    if failed == 0:
        print("🎉 All tests passed! Pattern matching functionality is working correctly.")
        print("\nSupported patterns:")
        print("- '*' - matches all functions")
        print("- 'api.*' - matches functions starting with 'api.'")
        print("- '*.handler' - matches functions ending with '.handler'")
        print("- 'api.*.get_*' - matches API getter functions")
        print("- 'MyClass.*' - matches all methods in MyClass")
        print("- 'core.*.database.*' - matches database functions in core modules")
        return True
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = test_pattern_matching()
    exit(0 if success else 1)
