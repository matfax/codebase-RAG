"""Tests for point_id_validator module."""

import uuid

import pytest

from src.utils.point_id_validator import InvalidPointIDError, detect_invalid_hex_format, is_valid_point_id, validate_point_id


class TestIsValidPointId:
    """Test Point ID validation function."""

    def test_valid_uuids(self):
        """Test valid UUID Point IDs."""
        valid_uuids = [
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            str(uuid.uuid4()),
            str(uuid.uuid5(uuid.NAMESPACE_DNS, "test")),
        ]

        for valid_uuid in valid_uuids:
            assert is_valid_point_id(valid_uuid), f"Should be valid: {valid_uuid}"

    def test_valid_unsigned_integers(self):
        """Test valid unsigned integer Point IDs."""
        valid_integers = [
            "0",
            "1",
            "12345",
            "999999999999999999",
        ]

        for valid_int in valid_integers:
            assert is_valid_point_id(valid_int), f"Should be valid: {valid_int}"

    def test_invalid_formats(self):
        """Test invalid Point ID formats."""
        invalid_formats = [
            "fedfda3b08808e5f",  # Old hex format
            "not-a-uuid",
            "",
            None,
            "-1",  # Negative integer
            "12.34",  # Float
            "550e8400-e29b-41d4-a716",  # Incomplete UUID
            "hello123",
        ]

        for invalid_format in invalid_formats:
            assert not is_valid_point_id(invalid_format), f"Should be invalid: {invalid_format}"

    def test_non_string_types(self):
        """Test non-string inputs."""
        non_strings = [
            None,
            123,
            12.34,
            [],
            {},
            True,
        ]

        for non_string in non_strings:
            assert not is_valid_point_id(non_string), f"Should be invalid: {non_string}"


class TestValidatePointId:
    """Test Point ID validation function that raises exceptions."""

    def test_valid_point_ids_pass(self):
        """Test that valid Point IDs pass validation."""
        valid_ids = [
            "550e8400-e29b-41d4-a716-446655440000",
            "12345",
            str(uuid.uuid4()),
        ]

        for valid_id in valid_ids:
            # Should not raise any exception
            validate_point_id(valid_id)

    def test_empty_point_id_raises_error(self):
        """Test that empty Point ID raises error."""
        with pytest.raises(InvalidPointIDError, match="Point ID cannot be empty"):
            validate_point_id("")

    def test_non_string_raises_error(self):
        """Test that non-string Point ID raises error."""
        with pytest.raises(InvalidPointIDError, match="Point ID must be a string"):
            validate_point_id(123)

    def test_hex_format_raises_specific_error(self):
        """Test that old hex format raises specific error."""
        hex_id = "fedfda3b08808e5f"

        with pytest.raises(InvalidPointIDError) as exc_info:
            validate_point_id(hex_id)

        error_msg = str(exc_info.value)
        assert "16-character hex string" in error_msg
        assert "Qdrant requires valid UUIDs" in error_msg

    def test_invalid_format_raises_error(self):
        """Test that invalid format raises error."""
        with pytest.raises(InvalidPointIDError) as exc_info:
            validate_point_id("invalid-format")

        error_msg = str(exc_info.value)
        assert "Invalid Point ID" in error_msg
        assert "Qdrant requires valid UUIDs" in error_msg

    def test_context_included_in_error(self):
        """Test that context is included in error messages."""
        context = "test file: /path/to/file.py"

        with pytest.raises(InvalidPointIDError) as exc_info:
            validate_point_id("invalid", context=context)

        error_msg = str(exc_info.value)
        assert context in error_msg

    def test_negative_integer_raises_error(self):
        """Test that negative integers raise error."""
        with pytest.raises(InvalidPointIDError):
            validate_point_id("-1")


class TestDetectInvalidHexFormat:
    """Test detection of old invalid hex format."""

    def test_detects_hex_format(self):
        """Test detection of 16-character hex strings."""
        hex_formats = [
            "fedfda3b08808e5f",
            "1234567890abcdef",
            "ABCDEF1234567890",
        ]

        for hex_format in hex_formats:
            assert detect_invalid_hex_format(hex_format), f"Should detect: {hex_format}"

    def test_ignores_valid_formats(self):
        """Test that valid formats are not detected as hex."""
        valid_formats = [
            "550e8400-e29b-41d4-a716-446655440000",  # Valid UUID
            "12345",  # Valid integer
            "abc",  # Too short
            "fedfda3b08808e5f123",  # Too long
            "not-hex-at-all",
        ]

        for valid_format in valid_formats:
            assert not detect_invalid_hex_format(valid_format), f"Should not detect: {valid_format}"

    def test_case_insensitive_detection(self):
        """Test that detection works regardless of case."""
        hex_variants = [
            "fedfda3b08808e5f",
            "FEDFDA3B08808E5F",
            "FedFda3B08808e5F",
        ]

        for hex_variant in hex_variants:
            assert detect_invalid_hex_format(hex_variant), f"Should detect: {hex_variant}"

    def test_non_string_input(self):
        """Test handling of non-string inputs."""
        non_strings = [None, 123, [], {}]

        for non_string in non_strings:
            assert not detect_invalid_hex_format(non_string), f"Should not detect: {non_string}"
