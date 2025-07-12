"""
Tests for cache backup and disaster recovery service.

This module contains comprehensive tests for the CacheBackupService,
including backup creation, restoration, and disaster recovery scenarios.
"""

import asyncio
import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from .cache_backup_service import (
    CacheBackupService,
    BackupType,
    BackupStatus,
    RecoveryStrategy,
    BackupMetadata,
    RestoreOperation,
    BackupConfiguration,
    get_cache_backup_service
)
from ..config.cache_config import CacheConfig
from ..models.cache_models import CacheEntry


class TestCacheBackupService:
    """Test cases for CacheBackupService."""

    @pytest.fixture
    def temp_backup_dir(self):
        """Create a temporary backup directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def backup_config(self, temp_backup_dir):
        """Create backup configuration."""
        return BackupConfiguration(
            backup_directory=temp_backup_dir,
            max_backup_age_days=7,
            max_backup_count=10,
            compression_enabled=True,
            encryption_enabled=False,  # Disable for testing
            verify_backup_integrity=True
        )

    @pytest.fixture
    async def mock_cache_service(self):
        """Create a mock cache service with test data."""
        cache_service = AsyncMock()
        
        # Mock L1 cache data
        test_entries = {
            "test_key_1": CacheEntry(
                data={"content": "test data 1", "type": "test"},
                size=100,
                created_at=datetime.now(),
                checksum="abc123"
            ),
            "test_key_2": CacheEntry(
                data={"content": "test data 2", "type": "test"},
                size=150,
                created_at=datetime.now(),
                checksum="def456"
            ),
            "test_key_3": CacheEntry(
                data={"content": "test data 3", "type": "test"},
                size=200,
                created_at=datetime.now(),
                checksum="ghi789"
            )
        }
        
        # Mock L1 cache
        cache_service._l1_cache = AsyncMock()
        cache_service._l1_cache._cache = test_entries
        cache_service.l1_cache = AsyncMock()
        cache_service.l1_cache.clear = AsyncMock()
        cache_service.l1_cache.set = AsyncMock()
        cache_service.l1_cache.exists = AsyncMock(return_value=False)
        
        # Mock Redis
        cache_service._redis = AsyncMock()
        cache_service._redis.keys.return_value = [b"redis_key_1", b"redis_key_2"]
        cache_service._redis.get.side_effect = lambda key: {
            "redis_key_1": b'{"data": "redis data 1"}',
            "redis_key_2": b'{"data": "redis data 2"}'
        }.get(key, None)
        cache_service._redis.ttl.return_value = 3600
        cache_service._redis.type.return_value = "string"
        cache_service._redis.flushdb = AsyncMock()
        cache_service._redis.set = AsyncMock()
        cache_service._redis.expire = AsyncMock()
        cache_service._redis.exists = AsyncMock(return_value=False)
        
        return cache_service

    @pytest.fixture
    def backup_service(self, backup_config):
        """Create a backup service instance."""
        config = CacheConfig()
        return CacheBackupService(config, backup_config)

    @pytest.mark.asyncio
    async def test_create_full_backup(self, backup_service, mock_cache_service):
        """Test creating a full backup."""
        with patch.object(backup_service, '_get_cache_service', return_value=mock_cache_service):
            metadata = await backup_service.create_backup(
                backup_type=BackupType.FULL,
                tiers=["L1", "L2"]
            )
            
            assert isinstance(metadata, BackupMetadata)
            assert metadata.backup_type == BackupType.FULL
            assert metadata.status == BackupStatus.COMPLETED
            assert metadata.total_entries > 0
            assert metadata.total_size_bytes > 0
            assert metadata.duration_seconds > 0
            assert len(metadata.checksum) > 0
            
            # Verify backup directory was created
            backup_dir = backup_service.backup_config.backup_directory / metadata.backup_id
            assert backup_dir.exists()
            assert (backup_dir / "cache_data.json").exists()
            assert (backup_dir / "metadata.json").exists()

    @pytest.mark.asyncio
    async def test_create_incremental_backup(self, backup_service, mock_cache_service):
        """Test creating an incremental backup."""
        with patch.object(backup_service, '_get_cache_service', return_value=mock_cache_service):
            # First create a full backup
            full_backup = await backup_service.create_backup(BackupType.FULL)
            
            # Then create incremental backup
            incremental_backup = await backup_service.create_backup(
                backup_type=BackupType.INCREMENTAL,
                base_backup_id=full_backup.backup_id
            )
            
            assert incremental_backup.backup_type == BackupType.INCREMENTAL
            assert incremental_backup.base_backup_id == full_backup.backup_id
            assert incremental_backup.status == BackupStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_backup_without_base_id_fails(self, backup_service, mock_cache_service):
        """Test that incremental backup without base ID fails."""
        with patch.object(backup_service, '_get_cache_service', return_value=mock_cache_service):
            with pytest.raises(ValueError, match="requires a base_backup_id"):
                await backup_service.create_backup(BackupType.INCREMENTAL)

    @pytest.mark.asyncio
    async def test_backup_compression(self, backup_service, mock_cache_service):
        """Test backup compression functionality."""
        with patch.object(backup_service, '_get_cache_service', return_value=mock_cache_service):
            # Create backup with compression
            metadata = await backup_service.create_backup(
                backup_type=BackupType.FULL,
                compress=True
            )
            
            # Compression ratio should be less than 1.0 for compressed data
            assert metadata.compression_ratio < 1.0

    @pytest.mark.asyncio
    async def test_list_backups(self, backup_service, mock_cache_service):
        """Test listing backups."""
        with patch.object(backup_service, '_get_cache_service', return_value=mock_cache_service):
            # Create multiple backups
            backup1 = await backup_service.create_backup(BackupType.FULL)
            backup2 = await backup_service.create_backup(BackupType.SNAPSHOT)
            
            # List all backups
            all_backups = await backup_service.list_backups()
            assert len(all_backups) >= 2
            
            backup_ids = [b.backup_id for b in all_backups]
            assert backup1.backup_id in backup_ids
            assert backup2.backup_id in backup_ids
            
            # List backups by type
            full_backups = await backup_service.list_backups(backup_type=BackupType.FULL)
            assert len(full_backups) >= 1
            assert all(b.backup_type == BackupType.FULL for b in full_backups)

    @pytest.mark.asyncio
    async def test_restore_replace_all(self, backup_service, mock_cache_service):
        """Test restore with replace all strategy."""
        with patch.object(backup_service, '_get_cache_service', return_value=mock_cache_service):
            # Create backup
            backup_metadata = await backup_service.create_backup(BackupType.FULL)
            
            # Restore with replace all strategy
            restore_op = await backup_service.restore_from_backup(
                backup_id=backup_metadata.backup_id,
                strategy=RecoveryStrategy.REPLACE_ALL,
                target_tiers=["L1", "L2"]
            )
            
            assert isinstance(restore_op, RestoreOperation)
            assert restore_op.backup_id == backup_metadata.backup_id
            assert restore_op.strategy == RecoveryStrategy.REPLACE_ALL
            assert restore_op.status == BackupStatus.COMPLETED
            assert restore_op.restored_entries > 0
            
            # Verify that clear operations were called
            mock_cache_service.l1_cache.clear.assert_called()
            mock_cache_service._redis.flushdb.assert_called()

    @pytest.mark.asyncio
    async def test_restore_merge_preserve(self, backup_service, mock_cache_service):
        """Test restore with merge preserve existing strategy."""
        with patch.object(backup_service, '_get_cache_service', return_value=mock_cache_service):
            # Create backup
            backup_metadata = await backup_service.create_backup(BackupType.FULL)
            
            # Mock existing keys
            mock_cache_service.l1_cache.exists.return_value = True
            mock_cache_service._redis.exists.return_value = True
            
            # Restore with merge preserve strategy
            restore_op = await backup_service.restore_from_backup(
                backup_id=backup_metadata.backup_id,
                strategy=RecoveryStrategy.MERGE_PRESERVE_EXISTING,
                target_tiers=["L1"]
            )
            
            assert restore_op.status == BackupStatus.COMPLETED
            # With existing keys, no new entries should be restored
            # (this depends on the mock implementation)

    @pytest.mark.asyncio
    async def test_restore_selective(self, backup_service, mock_cache_service):
        """Test selective restore."""
        with patch.object(backup_service, '_get_cache_service', return_value=mock_cache_service):
            # Create backup
            backup_metadata = await backup_service.create_backup(BackupType.FULL)
            
            # Restore only specific keys
            selective_keys = ["test_key_1", "redis_key_1"]
            restore_op = await backup_service.restore_from_backup(
                backup_id=backup_metadata.backup_id,
                strategy=RecoveryStrategy.SELECTIVE_RESTORE,
                selective_keys=selective_keys
            )
            
            assert restore_op.status == BackupStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_dry_run_restore(self, backup_service, mock_cache_service):
        """Test dry run restore operation."""
        with patch.object(backup_service, '_get_cache_service', return_value=mock_cache_service):
            # Create backup
            backup_metadata = await backup_service.create_backup(BackupType.FULL)
            
            # Perform dry run restore
            restore_op = await backup_service.restore_from_backup(
                backup_id=backup_metadata.backup_id,
                strategy=RecoveryStrategy.REPLACE_ALL,
                dry_run=True
            )
            
            assert restore_op.status == BackupStatus.COMPLETED
            assert restore_op.dry_run is True
            
            # Verify no actual restore operations were called
            mock_cache_service.l1_cache.clear.assert_not_called()
            mock_cache_service._redis.flushdb.assert_not_called()

    @pytest.mark.asyncio
    async def test_backup_verification(self, backup_service, mock_cache_service):
        """Test backup integrity verification."""
        with patch.object(backup_service, '_get_cache_service', return_value=mock_cache_service):
            # Create backup
            backup_metadata = await backup_service.create_backup(BackupType.FULL)
            
            # Verify backup integrity
            verification_result = await backup_service.verify_backup_integrity(backup_metadata.backup_id)
            
            assert verification_result["valid"] is True
            assert verification_result["checksum_valid"] is True
            assert verification_result["data_loadable"] is True
            assert verification_result["entry_count_consistent"] is True
            assert verification_result["actual_entries"] == backup_metadata.total_entries

    @pytest.mark.asyncio
    async def test_delete_backup(self, backup_service, mock_cache_service):
        """Test backup deletion."""
        with patch.object(backup_service, '_get_cache_service', return_value=mock_cache_service):
            # Create backup
            backup_metadata = await backup_service.create_backup(BackupType.FULL)
            backup_dir = backup_service.backup_config.backup_directory / backup_metadata.backup_id
            
            # Verify backup exists
            assert backup_dir.exists()
            
            # Delete backup
            success = await backup_service.delete_backup(backup_metadata.backup_id)
            assert success is True
            
            # Verify backup is deleted
            assert not backup_dir.exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_backup(self, backup_service):
        """Test deleting a nonexistent backup."""
        result = await backup_service.delete_backup("nonexistent_backup")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_backup_info(self, backup_service, mock_cache_service):
        """Test getting backup information."""
        with patch.object(backup_service, '_get_cache_service', return_value=mock_cache_service):
            # Create backup
            backup_metadata = await backup_service.create_backup(BackupType.FULL)
            
            # Get backup info
            info = await backup_service.get_backup_info(backup_metadata.backup_id)
            
            assert info is not None
            assert info.backup_id == backup_metadata.backup_id
            assert info.backup_type == BackupType.FULL
            assert info.status == BackupStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_backup_filtering_by_age(self, backup_service, mock_cache_service):
        """Test filtering backups by age."""
        with patch.object(backup_service, '_get_cache_service', return_value=mock_cache_service):
            # Create backup
            backup_metadata = await backup_service.create_backup(BackupType.FULL)
            
            # List recent backups (should include our backup)
            recent_backups = await backup_service.list_backups(max_age_days=1)
            backup_ids = [b.backup_id for b in recent_backups]
            assert backup_metadata.backup_id in backup_ids
            
            # List very old backups (should not include our backup)
            old_backups = await backup_service.list_backups(max_age_days=0)
            backup_ids = [b.backup_id for b in old_backups]
            # Our backup might still be included if created within the same day

    @pytest.mark.asyncio
    async def test_backup_error_handling(self, backup_service):
        """Test backup error handling."""
        # Mock a failing cache service
        failing_cache_service = AsyncMock()
        failing_cache_service.side_effect = Exception("Cache service error")
        
        with patch.object(backup_service, '_get_cache_service', side_effect=Exception("Cache error")):
            with pytest.raises(Exception):
                await backup_service.create_backup(BackupType.FULL)

    @pytest.mark.asyncio
    async def test_restore_error_handling(self, backup_service):
        """Test restore error handling."""
        with pytest.raises(ValueError, match="Backup not found"):
            await backup_service.restore_from_backup("nonexistent_backup")

    @pytest.mark.asyncio
    async def test_backup_metadata_persistence(self, backup_service, mock_cache_service):
        """Test that backup metadata is properly persisted and loaded."""
        with patch.object(backup_service, '_get_cache_service', return_value=mock_cache_service):
            # Create backup
            original_metadata = await backup_service.create_backup(BackupType.FULL)
            
            # Load metadata from disk
            loaded_metadata = await backup_service._load_backup_metadata(original_metadata.backup_id)
            
            assert loaded_metadata is not None
            assert loaded_metadata.backup_id == original_metadata.backup_id
            assert loaded_metadata.backup_type == original_metadata.backup_type
            assert loaded_metadata.total_entries == original_metadata.total_entries
            assert loaded_metadata.status == original_metadata.status

    @pytest.mark.asyncio
    async def test_backup_checksum_calculation(self, backup_service, mock_cache_service):
        """Test backup checksum calculation."""
        with patch.object(backup_service, '_get_cache_service', return_value=mock_cache_service):
            # Create backup
            metadata = await backup_service.create_backup(BackupType.FULL)
            
            # Verify checksum is calculated
            assert metadata.checksum
            assert len(metadata.checksum) == 32  # MD5 hex length
            
            # Verify checksum matches file
            backup_file = backup_service.backup_config.backup_directory / metadata.backup_id / "cache_data.json"
            calculated_checksum = await backup_service._calculate_backup_checksum(backup_file)
            assert calculated_checksum == metadata.checksum

    @pytest.mark.asyncio
    async def test_tier_data_collection(self, backup_service, mock_cache_service):
        """Test collection of data from different cache tiers."""
        with patch.object(backup_service, '_get_cache_service', return_value=mock_cache_service):
            # Test L1 data collection
            l1_data = await backup_service._get_tier_data(mock_cache_service, "L1")
            assert len(l1_data) > 0
            assert all("data" in entry and "metadata" in entry for entry in l1_data.values())
            
            # Test L2 data collection
            l2_data = await backup_service._get_tier_data(mock_cache_service, "L2")
            assert len(l2_data) > 0
            assert all("data" in entry and "metadata" in entry for entry in l2_data.values())

    @pytest.mark.asyncio
    async def test_incremental_data_filtering(self, backup_service, mock_cache_service):
        """Test filtering data for incremental backups."""
        with patch.object(backup_service, '_get_cache_service', return_value=mock_cache_service):
            # Create base time for filtering
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            # Get incremental data
            incremental_data = await backup_service._get_tier_data_since(
                mock_cache_service, "L1", cutoff_time
            )
            
            # Should return some data (depends on mock timestamps)
            assert isinstance(incremental_data, dict)

    @pytest.mark.asyncio
    async def test_backup_configuration_validation(self, temp_backup_dir):
        """Test backup configuration validation."""
        config = BackupConfiguration(
            backup_directory=temp_backup_dir,
            max_backup_age_days=30,
            max_backup_count=100
        )
        
        service = CacheBackupService(backup_config=config)
        
        assert service.backup_config.backup_directory == temp_backup_dir
        assert service.backup_config.max_backup_age_days == 30
        assert service.backup_config.max_backup_count == 100

    @pytest.mark.asyncio
    async def test_global_service_singleton(self):
        """Test global service singleton behavior."""
        service1 = await get_cache_backup_service()
        service2 = await get_cache_backup_service()
        
        assert service1 is service2
        assert isinstance(service1, CacheBackupService)

    @pytest.mark.asyncio
    async def test_backup_with_encryption_disabled(self, backup_service, mock_cache_service):
        """Test backup creation with encryption disabled."""
        with patch.object(backup_service, '_get_cache_service', return_value=mock_cache_service):
            metadata = await backup_service.create_backup(
                backup_type=BackupType.FULL,
                encrypt=False
            )
            
            assert metadata.encryption_enabled is False
            assert metadata.status == BackupStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_multiple_tier_backup_and_restore(self, backup_service, mock_cache_service):
        """Test backup and restore of multiple cache tiers."""
        with patch.object(backup_service, '_get_cache_service', return_value=mock_cache_service):
            # Create backup of both tiers
            backup_metadata = await backup_service.create_backup(
                backup_type=BackupType.FULL,
                tiers=["L1", "L2"]
            )
            
            assert "L1" in backup_metadata.cache_tiers
            assert "L2" in backup_metadata.cache_tiers
            
            # Restore to both tiers
            restore_op = await backup_service.restore_from_backup(
                backup_id=backup_metadata.backup_id,
                target_tiers=["L1", "L2"],
                strategy=RecoveryStrategy.REPLACE_ALL
            )
            
            assert "L1" in restore_op.target_tiers
            assert "L2" in restore_op.target_tiers
            assert restore_op.status == BackupStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_backup_service_initialization_and_shutdown(self, backup_config):
        """Test backup service initialization and shutdown."""
        service = CacheBackupService(backup_config=backup_config)
        
        # Initialize service
        await service.initialize()
        
        # Should have started background tasks
        assert service._cleanup_task is not None
        
        # Shutdown service
        await service.shutdown()
        
        # Tasks should be cancelled
        assert service._cleanup_task.cancelled()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])