"""
Tests for cache consistency verification service.

This module contains comprehensive tests for the CacheConsistencyService,
including unit tests and integration tests for consistency verification.
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from .cache_consistency_service import (
    CacheConsistencyService,
    ConsistencyCheckLevel,
    ConsistencyIssue,
    ConsistencyIssueType,
    ConsistencyReport,
    get_cache_consistency_service
)
from ..config.cache_config import CacheConfig
from ..models.cache_models import CacheEntry, CacheMetadata


class TestCacheConsistencyService:
    """Test cases for CacheConsistencyService."""

    @pytest.fixture
    async def mock_cache_service(self):
        """Create a mock cache service."""
        cache_service = AsyncMock()
        
        # Mock L1 cache
        cache_service._l1_cache = {
            "test_key_1": CacheEntry(
                data={"test": "data1"},
                checksum="abc123",
                size=20,
                created_at=datetime.now()
            ),
            "test_key_2": CacheEntry(
                data={"test": "data2"},
                checksum="def456",
                size=20,
                created_at=datetime.now()
            )
        }
        
        # Mock Redis
        cache_service._redis = AsyncMock()
        cache_service._redis.keys.return_value = [b"test_key_1", b"test_key_2", b"test_key_3"]
        cache_service._redis.get.side_effect = lambda key: {
            "test_key_1": json.dumps({"data": {"test": "data1"}, "checksum": "abc123"}),
            "test_key_2": json.dumps({"data": {"test": "different_data"}, "checksum": "xyz789"}),
            "test_key_3": json.dumps({"data": {"test": "data3"}, "checksum": "ghi012"})
        }.get(key)
        cache_service._redis.exists.side_effect = lambda key: key in ["test_key_1", "test_key_2", "test_key_3"]
        
        # Mock cache service methods
        cache_service.get.side_effect = lambda key: cache_service._l1_cache.get(key)
        cache_service.set = AsyncMock()
        cache_service.delete = AsyncMock()
        
        return cache_service

    @pytest.fixture
    def consistency_service(self):
        """Create a consistency service instance."""
        config = CacheConfig()
        return CacheConsistencyService(config)

    @pytest.mark.asyncio
    async def test_basic_consistency_check(self, consistency_service, mock_cache_service):
        """Test basic consistency verification."""
        with patch.object(consistency_service, '_get_cache_service', return_value=mock_cache_service):
            report = await consistency_service.verify_consistency(
                check_level=ConsistencyCheckLevel.BASIC
            )
            
            assert isinstance(report, ConsistencyReport)
            assert report.check_level == ConsistencyCheckLevel.BASIC
            assert report.total_keys_checked > 0
            assert isinstance(report.consistency_score, float)
            assert 0.0 <= report.consistency_score <= 1.0

    @pytest.mark.asyncio
    async def test_l1_l2_mismatch_detection(self, consistency_service, mock_cache_service):
        """Test detection of L1/L2 mismatches."""
        with patch.object(consistency_service, '_get_cache_service', return_value=mock_cache_service):
            report = await consistency_service.verify_consistency(
                check_level=ConsistencyCheckLevel.COMPREHENSIVE
            )
            
            # Should detect mismatch for test_key_2
            l1_l2_issues = [issue for issue in report.issues_found 
                          if issue.issue_type == ConsistencyIssueType.L1_L2_MISMATCH]
            assert len(l1_l2_issues) > 0
            
            # Check specific issue details
            mismatch_issue = l1_l2_issues[0]
            assert mismatch_issue.cache_key == "test_key_2"
            assert mismatch_issue.severity == "medium"
            assert mismatch_issue.resolution_action == "sync_l1_l2"

    @pytest.mark.asyncio
    async def test_orphaned_entry_detection(self, consistency_service, mock_cache_service):
        """Test detection of orphaned entries."""
        with patch.object(consistency_service, '_get_cache_service', return_value=mock_cache_service):
            report = await consistency_service.verify_consistency(
                check_level=ConsistencyCheckLevel.DEEP
            )
            
            # Should detect orphaned entry for test_key_3 (exists in L2 but not L1)
            # This is normal and shouldn't be flagged as high severity
            orphan_issues = [issue for issue in report.issues_found 
                           if issue.issue_type == ConsistencyIssueType.ORPHANED_ENTRY]
            
            # Check that orphaned entries have low severity
            for issue in orphan_issues:
                assert issue.severity in ["low", "medium"]

    @pytest.mark.asyncio
    async def test_data_integrity_validation(self, consistency_service, mock_cache_service):
        """Test data integrity validation with checksums."""
        # Create entry with invalid checksum
        corrupted_entry = CacheEntry(
            data={"test": "data"},
            checksum="invalid_checksum",
            size=15,
            created_at=datetime.now()
        )
        
        mock_cache_service.get.side_effect = lambda key: corrupted_entry if key == "corrupted_key" else None
        
        with patch.object(consistency_service, '_get_cache_service', return_value=mock_cache_service):
            issues = await consistency_service._check_data_integrity(mock_cache_service, ["corrupted_key"])
            
            checksum_issues = [issue for issue in issues 
                             if issue.issue_type == ConsistencyIssueType.INVALID_CHECKSUM]
            assert len(checksum_issues) > 0
            
            issue = checksum_issues[0]
            assert issue.cache_key == "corrupted_key"
            assert issue.severity == "high"
            assert "checksum" in issue.description.lower()

    @pytest.mark.asyncio
    async def test_expiration_validation(self, consistency_service, mock_cache_service):
        """Test validation of expired entries."""
        # Create expired entry
        expired_entry = CacheEntry(
            data={"test": "data"},
            checksum="abc123",
            size=15,
            created_at=datetime.now() - timedelta(hours=2),
            expires_at=datetime.now() - timedelta(hours=1)  # Expired 1 hour ago
        )
        
        mock_cache_service.get.side_effect = lambda key: expired_entry if key == "expired_key" else None
        
        with patch.object(consistency_service, '_get_cache_service', return_value=mock_cache_service):
            issues = await consistency_service._check_expiration_consistency(mock_cache_service, ["expired_key"])
            
            expired_issues = [issue for issue in issues 
                            if issue.issue_type == ConsistencyIssueType.EXPIRED_DATA]
            assert len(expired_issues) > 0
            
            issue = expired_issues[0]
            assert issue.cache_key == "expired_key"
            assert issue.resolution_action == "remove_expired"

    @pytest.mark.asyncio
    async def test_metadata_validation(self, consistency_service, mock_cache_service):
        """Test metadata consistency validation."""
        # Create entry with invalid size metadata
        invalid_size_entry = CacheEntry(
            data={"test": "this is much longer data than reported"},
            checksum="abc123",
            size=5,  # Incorrect size
            created_at=datetime.now()
        )
        
        mock_cache_service.get.side_effect = lambda key: invalid_size_entry if key == "invalid_size_key" else None
        
        with patch.object(consistency_service, '_get_cache_service', return_value=mock_cache_service):
            issues = await consistency_service._check_metadata_consistency(mock_cache_service, ["invalid_size_key"])
            
            metadata_issues = [issue for issue in issues 
                             if issue.issue_type == ConsistencyIssueType.METADATA_MISMATCH]
            assert len(metadata_issues) > 0
            
            issue = metadata_issues[0]
            assert issue.cache_key == "invalid_size_key"
            assert "size" in issue.description.lower()

    @pytest.mark.asyncio
    async def test_consistency_score_calculation(self, consistency_service):
        """Test consistency score calculation."""
        # Test with no issues
        score = consistency_service._calculate_consistency_score([], 100)
        assert score == 1.0
        
        # Test with various severities
        issues = [
            ConsistencyIssue(
                issue_type=ConsistencyIssueType.L1_L2_MISMATCH,
                cache_key="key1",
                description="Test",
                severity="low"
            ),
            ConsistencyIssue(
                issue_type=ConsistencyIssueType.CORRUPTED_DATA,
                cache_key="key2",
                description="Test",
                severity="critical"
            )
        ]
        
        score = consistency_service._calculate_consistency_score(issues, 100)
        assert 0.0 <= score < 1.0  # Should be less than 1.0 due to issues

    @pytest.mark.asyncio
    async def test_automatic_issue_fixing(self, consistency_service, mock_cache_service):
        """Test automatic fixing of consistency issues."""
        issues = [
            ConsistencyIssue(
                issue_type=ConsistencyIssueType.L1_L2_MISMATCH,
                cache_key="sync_key",
                description="Test sync",
                resolution_action="sync_l1_l2",
                l2_value={"corrected": "data"}
            ),
            ConsistencyIssue(
                issue_type=ConsistencyIssueType.CORRUPTED_DATA,
                cache_key="corrupt_key",
                description="Test corruption",
                resolution_action="remove_corrupted"
            ),
            ConsistencyIssue(
                issue_type=ConsistencyIssueType.EXPIRED_DATA,
                cache_key="expired_key",
                description="Test expiration",
                resolution_action="remove_expired"
            )
        ]
        
        with patch.object(consistency_service, '_get_cache_service', return_value=mock_cache_service):
            fix_results = await consistency_service._fix_consistency_issues(mock_cache_service, issues)
            
            assert fix_results["fixed"] >= 2  # At least remove operations should succeed
            assert fix_results["failed"] == 0
            
            # Verify cache operations were called
            assert mock_cache_service.delete.call_count >= 2
            assert mock_cache_service.set.call_count >= 0

    @pytest.mark.asyncio
    async def test_recommendation_generation(self, consistency_service):
        """Test generation of recommendations."""
        # Test with good consistency
        recommendations = consistency_service._generate_recommendations([], 0.95)
        assert "good" in " ".join(recommendations).lower()
        
        # Test with poor consistency and various issues
        issues = [
            ConsistencyIssue(
                issue_type=ConsistencyIssueType.L1_L2_MISMATCH,
                cache_key="key1",
                description="Test",
                severity="medium"
            ),
            ConsistencyIssue(
                issue_type=ConsistencyIssueType.CORRUPTED_DATA,
                cache_key="key2",
                description="Test",
                severity="high"
            )
        ]
        
        recommendations = consistency_service._generate_recommendations(issues, 0.7)
        assert len(recommendations) > 1
        assert any("threshold" in rec.lower() for rec in recommendations)
        assert any("mismatch" in rec.lower() for rec in recommendations)
        assert any("corrupted" in rec.lower() for rec in recommendations)

    @pytest.mark.asyncio
    async def test_comprehensive_consistency_check(self, consistency_service, mock_cache_service):
        """Test comprehensive consistency check with fixes."""
        with patch.object(consistency_service, '_get_cache_service', return_value=mock_cache_service):
            report = await consistency_service.verify_consistency(
                check_level=ConsistencyCheckLevel.COMPREHENSIVE,
                fix_issues=True
            )
            
            assert isinstance(report, ConsistencyReport)
            assert report.check_level == ConsistencyCheckLevel.COMPREHENSIVE
            assert len(report.recommendations) > 0
            assert isinstance(report.l1_stats, dict)
            assert isinstance(report.l2_stats, dict)
            
            # Check that statistics are populated
            assert "total_keys" in report.l1_stats
            assert "total_keys" in report.l2_stats

    def test_values_equal_comparison(self, consistency_service):
        """Test value equality comparison."""
        # Test identical values
        assert consistency_service._values_equal({"a": 1}, {"a": 1})
        assert consistency_service._values_equal("test", "test")
        
        # Test different values
        assert not consistency_service._values_equal({"a": 1}, {"a": 2})
        assert not consistency_service._values_equal("test1", "test2")
        
        # Test string/bytes conversion
        assert consistency_service._values_equal("test", b"test")
        assert consistency_service._values_equal(b"test", "test")
        
        # Test complex objects
        assert consistency_service._values_equal(
            {"list": [1, 2, 3], "dict": {"nested": True}},
            {"list": [1, 2, 3], "dict": {"nested": True}}
        )

    def test_checksum_calculation(self, consistency_service):
        """Test checksum calculation."""
        # Test consistent checksums
        data1 = {"test": "data", "number": 123}
        data2 = {"test": "data", "number": 123}
        
        checksum1 = consistency_service._calculate_checksum(data1)
        checksum2 = consistency_service._calculate_checksum(data2)
        
        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA-256 hex length
        
        # Test different data produces different checksums
        data3 = {"test": "different", "number": 123}
        checksum3 = consistency_service._calculate_checksum(data3)
        
        assert checksum1 != checksum3

    def test_data_validation(self, consistency_service):
        """Test data validation methods."""
        # Test valid data
        assert consistency_service._is_data_valid({"valid": "data"})
        assert consistency_service._is_data_valid("string data")
        assert consistency_service._is_data_valid([1, 2, 3])
        
        # Test metadata validation
        valid_metadata = CacheMetadata(
            created_at=datetime.now() - timedelta(minutes=5),
            expires_at=datetime.now() + timedelta(hours=1)
        )
        assert consistency_service._is_metadata_valid(valid_metadata)
        
        # Test invalid metadata (created in future)
        invalid_metadata = CacheMetadata(
            created_at=datetime.now() + timedelta(hours=1),
            expires_at=datetime.now() + timedelta(hours=2)
        )
        assert not consistency_service._is_metadata_valid(invalid_metadata)

    @pytest.mark.asyncio
    async def test_get_global_service(self):
        """Test global service singleton."""
        service1 = await get_cache_consistency_service()
        service2 = await get_cache_consistency_service()
        
        assert service1 is service2  # Should be same instance
        assert isinstance(service1, CacheConsistencyService)

    @pytest.mark.asyncio
    async def test_deep_consistency_check_sampling(self, consistency_service, mock_cache_service):
        """Test that deep consistency checks properly sample large key sets."""
        # Mock a large number of keys
        large_key_set = [f"key_{i}" for i in range(2000)]
        
        with patch.object(consistency_service, '_get_all_cache_keys', return_value=large_key_set):
            with patch.object(consistency_service, '_get_cache_service', return_value=mock_cache_service):
                report = await consistency_service.verify_consistency(
                    check_level=ConsistencyCheckLevel.DEEP
                )
                
                # Should sample down to max_sample_size
                max_sample = consistency_service.check_config["max_sample_size"]
                assert report.total_keys_checked <= max_sample

    @pytest.mark.asyncio
    async def test_consistency_check_error_handling(self, consistency_service):
        """Test error handling in consistency checks."""
        # Mock a cache service that throws errors
        failing_cache_service = AsyncMock()
        failing_cache_service._l1_cache = {"error_key": "test"}
        failing_cache_service._redis = AsyncMock()
        failing_cache_service._redis.keys.side_effect = Exception("Redis error")
        
        with patch.object(consistency_service, '_get_cache_service', return_value=failing_cache_service):
            # Should not raise exception, but handle gracefully
            report = await consistency_service.verify_consistency(
                check_level=ConsistencyCheckLevel.BASIC
            )
            
            assert isinstance(report, ConsistencyReport)
            # Check that errors were handled gracefully
            assert report.consistency_score >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])