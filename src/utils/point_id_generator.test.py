"""Tests for point_id_generator module."""

import uuid
from pathlib import Path

import pytest

from src.utils.point_id_generator import POINT_ID_NAMESPACE, generate_deterministic_uuid, is_valid_uuid_format


class TestGenerateDeterministicUuid:
    """Test deterministic UUID generation."""

    def test_generates_valid_uuid(self):
        """Test that generated UUIDs are valid."""
        file_path = "/path/to/test/file.py"
        result = generate_deterministic_uuid(file_path)

        # Should be a valid UUID string
        assert is_valid_uuid_format(result)
        assert isinstance(result, str)
        assert len(result) == 36  # Standard UUID string length
        assert result.count("-") == 4  # Standard UUID format

    def test_deterministic_behavior(self):
        """Test that same input always produces same output."""
        file_path = "/path/to/test/file.py"

        # Generate UUID multiple times
        uuid1 = generate_deterministic_uuid(file_path)
        uuid2 = generate_deterministic_uuid(file_path)
        uuid3 = generate_deterministic_uuid(file_path)

        # All should be identical
        assert uuid1 == uuid2 == uuid3

    def test_different_paths_generate_different_uuids(self):
        """Test that different paths generate different UUIDs."""
        path1 = "/path/to/file1.py"
        path2 = "/path/to/file2.py"
        path3 = "/different/path/file.py"

        uuid1 = generate_deterministic_uuid(path1)
        uuid2 = generate_deterministic_uuid(path2)
        uuid3 = generate_deterministic_uuid(path3)

        # All should be different
        assert uuid1 != uuid2
        assert uuid1 != uuid3
        assert uuid2 != uuid3

    def test_handles_path_objects(self):
        """Test that Path objects work correctly."""
        path_str = "/path/to/test/file.py"
        path_obj = Path(path_str)

        uuid_from_str = generate_deterministic_uuid(path_str)
        uuid_from_path = generate_deterministic_uuid(str(path_obj))

        # Should generate same UUID
        assert uuid_from_str == uuid_from_path

    def test_consistent_namespace_usage(self):
        """Test that the namespace UUID is used correctly."""
        file_path = "/test/file.py"
        result = generate_deterministic_uuid(file_path)

        # Manually generate using the same method
        expected = str(uuid.uuid5(POINT_ID_NAMESPACE, file_path))

        assert result == expected

    def test_handles_special_characters(self):
        """Test paths with special characters."""
        special_paths = [
            "/path with spaces/file.py",
            "/path/with-dashes/file.py",
            "/path_with_underscores/file.py",
            "/path/with.dots/file.py",
            "/path/with中文/file.py",
        ]

        # All should generate valid UUIDs
        uuids = []
        for path in special_paths:
            result = generate_deterministic_uuid(path)
            assert is_valid_uuid_format(result)
            uuids.append(result)

        # All should be different
        assert len(set(uuids)) == len(uuids)


class TestIsValidUuidFormat:
    """Test UUID format validation."""

    def test_valid_uuids(self):
        """Test valid UUID formats."""
        valid_uuids = [
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
            "123e4567-e89b-12d3-a456-426614174000",
            str(uuid.uuid4()),
            str(uuid.uuid5(POINT_ID_NAMESPACE, "test")),
        ]

        for valid_uuid in valid_uuids:
            assert is_valid_uuid_format(valid_uuid), f"Should be valid: {valid_uuid}"

    def test_invalid_formats(self):
        """Test invalid formats."""
        invalid_formats = [
            "fedfda3b08808e5f",  # Old hex format
            "not-a-uuid",
            "123456789",
            "",
            None,
            "550e8400-e29b-41d4-a716",  # Too short
            "550e8400-e29b-41d4-a716-446655440000-extra",  # Too long
            "550e8400e29b41d4a716446655440000",  # No dashes
        ]

        for invalid_format in invalid_formats:
            assert not is_valid_uuid_format(invalid_format), f"Should be invalid: {invalid_format}"

    def test_case_insensitive(self):
        """Test that validation is case insensitive."""
        uuid_lower = "550e8400-e29b-41d4-a716-446655440000"
        uuid_upper = "550E8400-E29B-41D4-A716-446655440000"
        uuid_mixed = "550e8400-E29B-41d4-A716-446655440000"

        assert is_valid_uuid_format(uuid_lower)
        assert is_valid_uuid_format(uuid_upper)
        assert is_valid_uuid_format(uuid_mixed)
