#!/usr/bin/env python3
"""
Test script for pattern matching functionality in project chain analysis.
"""

import sys

sys.path.append("/Users/jeff/Documents/personal/Agentic-RAG/trees/function-chain-mcp-tools-wave")

from src.tools.graph_rag.project_chain_analysis import (
    _convert_pattern_to_regex,
    _get_advanced_pattern_examples,
    _matches_scope_pattern,
    _simple_pattern_match,
    _validate_scope_pattern,
)


def test_basic_patterns():
    """Test basic pattern matching functionality."""
    print("Testing basic pattern matching...")

    test_cases = [
        # (breadcrumb, pattern, expected_result)
        ("api.user.get_user", "*", True),
        ("api.user.get_user", "api.*", True),
        ("api.user.get_user", "core.*", False),
        ("api.user.get_user", "*.get_user", True),
        ("api.user.get_user", "*.get_product", False),
        ("api.user.get_user", "api.*.get_user", True),
        ("api.user.get_user", "api.*.get_product", False),
        ("MyClass.method1", "MyClass.*", True),
        ("MyClass.method1", "YourClass.*", False),
        ("user.test_create", "*.test_*", True),
        ("user.create", "*.test_*", False),
        ("core.user.database.save", "core.*.database.*", True),
        ("core.user.service.save", "core.*.database.*", False),
        ("api.user.get_user", "api.user.get_user", True),
        ("api.user.get_user", "api.user.get_product", False),
    ]

    passed = 0
    failed = 0

    for breadcrumb, pattern, expected in test_cases:
        result = _matches_scope_pattern(breadcrumb, pattern)
        if result == expected:
            print(f"✅ PASS: '{breadcrumb}' matches '{pattern}' -> {result}")
            passed += 1
        else:
            print(f"❌ FAIL: '{breadcrumb}' matches '{pattern}' -> {result} (expected {expected})")
            failed += 1

    print(f"\nBasic pattern tests: {passed} passed, {failed} failed")
    return failed == 0


def test_pattern_validation():
    """Test pattern validation functionality."""
    print("\nTesting pattern validation...")

    valid_patterns = ["*", "api.*", "*.handler", "MyClass.*", "api.*.get_*", "core.*.database.*"]

    invalid_patterns = ["", "api.|user", "api.&user", "api.(user)", "api.[user]", "api.{user}", "api.^user", "api.$user", "api.+user"]

    passed = 0
    failed = 0

    for pattern in valid_patterns:
        result = _validate_scope_pattern(pattern)
        if result["valid"]:
            print(f"✅ PASS: '{pattern}' is valid")
            passed += 1
        else:
            print(f"❌ FAIL: '{pattern}' should be valid but got: {result['error']}")
            failed += 1

    for pattern in invalid_patterns:
        result = _validate_scope_pattern(pattern)
        if not result["valid"]:
            print(f"✅ PASS: '{pattern}' is invalid - {result['error']}")
            passed += 1
        else:
            print(f"❌ FAIL: '{pattern}' should be invalid but was accepted")
            failed += 1

    print(f"\nPattern validation tests: {passed} passed, {failed} failed")
    return failed == 0


def test_regex_conversion():
    """Test regex conversion functionality."""
    print("\nTesting regex conversion...")

    test_cases = [
        ("*", "^.*$"),
        ("api.*", "^api\\..*$"),
        ("*.handler", "^.*\\.handler$"),
        ("api.*.get_*", "^api\\..*\\.get_.*$"),
        ("MyClass.*", "^MyClass\\..*$"),
    ]

    passed = 0
    failed = 0

    for pattern, expected_regex in test_cases:
        result = _convert_pattern_to_regex(pattern)
        if result == expected_regex:
            print(f"✅ PASS: '{pattern}' -> '{result}'")
            passed += 1
        else:
            print(f"❌ FAIL: '{pattern}' -> '{result}' (expected '{expected_regex}')")
            failed += 1

    print(f"\nRegex conversion tests: {passed} passed, {failed} failed")
    return failed == 0


def test_simple_pattern_match():
    """Test simple pattern matching fallback."""
    print("\nTesting simple pattern matching fallback...")

    test_cases = [
        ("api.user.get_user", "api.*", True),
        ("api.user.get_user", "core.*", False),
        ("api.user.get_user", "api.*.get_user", True),
        ("api.user.get_user", "api.*.get_product", False),
        ("complex.path.with.many.parts", "complex.*.many.*", True),
        ("complex.path.with.many.parts", "complex.*.few.*", False),
    ]

    passed = 0
    failed = 0

    for breadcrumb, pattern, expected in test_cases:
        result = _simple_pattern_match(breadcrumb, pattern)
        if result == expected:
            print(f"✅ PASS: '{breadcrumb}' matches '{pattern}' -> {result}")
            passed += 1
        else:
            print(f"❌ FAIL: '{breadcrumb}' matches '{pattern}' -> {result} (expected {expected})")
            failed += 1

    print(f"\nSimple pattern matching tests: {passed} passed, {failed} failed")
    return failed == 0


def test_advanced_examples():
    """Test advanced pattern examples."""
    print("\nTesting advanced pattern examples...")

    examples = _get_advanced_pattern_examples()

    if len(examples) >= 7:
        print(f"✅ PASS: Got {len(examples)} advanced pattern examples")
        for example in examples[:3]:  # Show first 3
            print(f"  - {example['pattern']}: {example['description']}")
        return True
    else:
        print(f"❌ FAIL: Expected at least 7 examples, got {len(examples)}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Pattern Matching Functionality for Wave 4.0 Subtask 4.2")
    print("=" * 60)

    all_passed = True

    all_passed &= test_basic_patterns()
    all_passed &= test_pattern_validation()
    all_passed &= test_regex_conversion()
    all_passed &= test_simple_pattern_match()
    all_passed &= test_advanced_examples()

    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! Pattern matching functionality is working correctly.")
    else:
        print("❌ Some tests failed. Please review the implementation.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
