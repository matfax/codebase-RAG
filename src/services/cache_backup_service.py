"""
Cache backup and disaster recovery service for the Codebase RAG MCP Server.

This service provides comprehensive backup and disaster recovery capabilities for cache data,
including automated backups, point-in-time recovery, and cross-tier backup strategies.
"""

import asyncio
import gzip
import json
import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config.cache_config import CacheConfig, get_global_cache_config
from ..models.cache_models import CacheEntry
from ..services.cache_service import get_cache_service
from ..utils.encryption_utils import EncryptionUtils
from ..utils.telemetry import get_telemetry_manager, trace_cache_operation


class BackupType(Enum):
    """Types of cache backups."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"


class BackupStatus(Enum):
    """Backup operation status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RecoveryStrategy(Enum):
    """Cache recovery strategies."""
    REPLACE_ALL = "replace_all"
    MERGE_PRESERVE_EXISTING = "merge_preserve_existing"
    MERGE_OVERWRITE_EXISTING = "merge_overwrite_existing"
    SELECTIVE_RESTORE = "selective_restore"


@dataclass
class BackupMetadata:
    """Metadata for cache backup."""
    backup_id: str
    backup_type: BackupType
    timestamp: datetime
    cache_tiers: List[str]  # L1, L2, or both
    total_entries: int
    total_size_bytes: int
    compression_ratio: float
    encryption_enabled: bool
    checksum: str
    duration_seconds: float
    status: BackupStatus = BackupStatus.PENDING
    error_message: Optional[str] = None
    includes_metadata: bool = True
    base_backup_id: Optional[str] = None  # For incremental/differential backups


@dataclass
class RestoreOperation:
    """Represents a cache restore operation."""
    restore_id: str
    backup_id: str
    target_tiers: List[str]
    strategy: RecoveryStrategy
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: BackupStatus = BackupStatus.PENDING
    restored_entries: int = 0
    failed_entries: int = 0
    error_message: Optional[str] = None
    dry_run: bool = False


@dataclass
class BackupConfiguration:
    """Configuration for backup operations."""
    backup_directory: Path
    max_backup_age_days: int = 30
    max_backup_count: int = 100
    compression_enabled: bool = True
    encryption_enabled: bool = True
    incremental_backup_interval_hours: int = 6
    full_backup_interval_days: int = 7
    verify_backup_integrity: bool = True
    parallel_backup_workers: int = 4
    chunk_size_bytes: int = 1024 * 1024  # 1MB chunks


class CacheBackupService:
    """Service for cache backup and disaster recovery operations."""

    def __init__(self, config: Optional[CacheConfig] = None, backup_config: Optional[BackupConfiguration] = None):
        """Initialize the cache backup service."""
        self.config = config or get_global_cache_config()
        self.backup_config = backup_config or BackupConfiguration(
            backup_directory=Path("./cache_backups")
        )
        self.logger = logging.getLogger(__name__)
        self._cache_service = None
        self._encryption_utils = EncryptionUtils()
        self._telemetry = get_telemetry_manager()
        
        # Ensure backup directory exists
        self.backup_config.backup_directory.mkdir(parents=True, exist_ok=True)
        
        # Active operations tracking
        self._active_backups: Dict[str, BackupMetadata] = {}
        self._active_restores: Dict[str, RestoreOperation] = {}
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._auto_backup_task: Optional[asyncio.Task] = None

    async def _get_cache_service(self):
        """Get cache service instance."""
        if self._cache_service is None:
            self._cache_service = await get_cache_service()
        return self._cache_service

    async def initialize(self):
        """Initialize the backup service and start background tasks."""
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        # Start auto-backup task if configured
        if hasattr(self.backup_config, 'auto_backup_enabled') and self.backup_config.auto_backup_enabled:
            self._auto_backup_task = asyncio.create_task(self._auto_backup_loop())

    async def shutdown(self):
        """Shutdown the backup service and cleanup resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._auto_backup_task:
            self._auto_backup_task.cancel()
            try:
                await self._auto_backup_task
            except asyncio.CancelledError:
                pass

    @trace_cache_operation("backup_cache")
    async def create_backup(
        self,
        backup_type: BackupType = BackupType.FULL,
        tiers: Optional[List[str]] = None,
        include_metadata: bool = True,
        compress: Optional[bool] = None,
        encrypt: Optional[bool] = None,
        base_backup_id: Optional[str] = None
    ) -> BackupMetadata:
        """
        Create a cache backup.
        
        Args:
            backup_type: Type of backup to create
            tiers: Cache tiers to backup (None for all)
            include_metadata: Whether to include cache metadata
            compress: Whether to compress backup data (None for config default)
            encrypt: Whether to encrypt backup data (None for config default)
            base_backup_id: Base backup ID for incremental/differential backups
            
        Returns:
            BackupMetadata object with backup information
        """
        backup_id = f"backup_{int(time.time())}_{backup_type.value}"
        
        # Set defaults
        if tiers is None:
            tiers = ["L1", "L2"]
        if compress is None:
            compress = self.backup_config.compression_enabled
        if encrypt is None:
            encrypt = self.backup_config.encryption_enabled
        
        # Validate incremental/differential backup requirements
        if backup_type in [BackupType.INCREMENTAL, BackupType.DIFFERENTIAL] and not base_backup_id:
            raise ValueError(f"{backup_type.value} backup requires a base_backup_id")
        
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            timestamp=datetime.now(),
            cache_tiers=tiers,
            total_entries=0,
            total_size_bytes=0,
            compression_ratio=1.0,
            encryption_enabled=encrypt,
            checksum="",
            duration_seconds=0.0,
            status=BackupStatus.IN_PROGRESS,
            includes_metadata=include_metadata,
            base_backup_id=base_backup_id
        )
        
        self._active_backups[backup_id] = metadata
        
        try:
            start_time = time.time()
            self.logger.info(f"Starting {backup_type.value} backup: {backup_id}")
            
            # Get cache service
            cache_service = await self._get_cache_service()
            
            # Create backup directory
            backup_dir = self.backup_config.backup_directory / backup_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Collect cache data based on backup type
            if backup_type == BackupType.FULL:
                cache_data = await self._collect_full_backup_data(cache_service, tiers)
            elif backup_type == BackupType.INCREMENTAL:
                cache_data = await self._collect_incremental_backup_data(
                    cache_service, tiers, base_backup_id
                )
            elif backup_type == BackupType.DIFFERENTIAL:
                cache_data = await self._collect_differential_backup_data(
                    cache_service, tiers, base_backup_id
                )
            elif backup_type == BackupType.SNAPSHOT:
                cache_data = await self._collect_snapshot_backup_data(cache_service, tiers)
            else:
                raise ValueError(f"Unsupported backup type: {backup_type}")
            
            # Write backup data
            backup_file = backup_dir / "cache_data.json"
            metadata_file = backup_dir / "metadata.json"
            
            # Process and write data
            original_size = await self._write_backup_data(
                cache_data, backup_file, compress, encrypt
            )
            
            # Calculate compression ratio
            final_size = backup_file.stat().st_size
            metadata.compression_ratio = original_size / final_size if final_size > 0 else 1.0
            
            # Update metadata
            metadata.total_entries = len(cache_data)
            metadata.total_size_bytes = final_size
            metadata.duration_seconds = time.time() - start_time
            metadata.checksum = await self._calculate_backup_checksum(backup_file)
            metadata.status = BackupStatus.COMPLETED
            
            # Write metadata
            await self._write_metadata(metadata, metadata_file)
            
            # Verify backup if configured
            if self.backup_config.verify_backup_integrity:
                await self._verify_backup_integrity(backup_id)
            
            self.logger.info(
                f"Backup completed: {backup_id}, "
                f"entries: {metadata.total_entries}, "
                f"size: {metadata.total_size_bytes} bytes, "
                f"duration: {metadata.duration_seconds:.2f}s"
            )
            
            return metadata
            
        except Exception as e:
            metadata.status = BackupStatus.FAILED
            metadata.error_message = str(e)
            self.logger.error(f"Backup failed: {backup_id}, error: {e}")
            raise
        finally:
            # Update active backups
            if backup_id in self._active_backups:
                self._active_backups[backup_id] = metadata

    async def restore_from_backup(
        self,
        backup_id: str,
        strategy: RecoveryStrategy = RecoveryStrategy.REPLACE_ALL,
        target_tiers: Optional[List[str]] = None,
        selective_keys: Optional[List[str]] = None,
        dry_run: bool = False
    ) -> RestoreOperation:
        """
        Restore cache from backup.
        
        Args:
            backup_id: ID of backup to restore from
            strategy: Recovery strategy to use
            target_tiers: Target cache tiers (None for all)
            selective_keys: Specific keys to restore (for selective strategy)
            dry_run: Whether to perform a dry run without actual restoration
            
        Returns:
            RestoreOperation object with restoration details
        """
        restore_id = f"restore_{int(time.time())}_{backup_id}"
        
        if target_tiers is None:
            target_tiers = ["L1", "L2"]
        
        restore_op = RestoreOperation(
            restore_id=restore_id,
            backup_id=backup_id,
            target_tiers=target_tiers,
            strategy=strategy,
            started_at=datetime.now(),
            status=BackupStatus.IN_PROGRESS,
            dry_run=dry_run
        )
        
        self._active_restores[restore_id] = restore_op
        
        try:
            self.logger.info(f"Starting restore operation: {restore_id} (dry_run: {dry_run})")
            
            # Load backup metadata
            backup_metadata = await self._load_backup_metadata(backup_id)
            if not backup_metadata:
                raise ValueError(f"Backup not found: {backup_id}")
            
            # Load backup data
            backup_data = await self._load_backup_data(backup_id)
            
            # Filter data if selective restore
            if strategy == RecoveryStrategy.SELECTIVE_RESTORE and selective_keys:
                backup_data = {k: v for k, v in backup_data.items() if k in selective_keys}
            
            # Get cache service
            cache_service = await self._get_cache_service()
            
            # Perform restoration based on strategy
            if not dry_run:
                if strategy == RecoveryStrategy.REPLACE_ALL:
                    await self._restore_replace_all(cache_service, backup_data, target_tiers)
                elif strategy == RecoveryStrategy.MERGE_PRESERVE_EXISTING:
                    await self._restore_merge_preserve(cache_service, backup_data, target_tiers)
                elif strategy == RecoveryStrategy.MERGE_OVERWRITE_EXISTING:
                    await self._restore_merge_overwrite(cache_service, backup_data, target_tiers)
                elif strategy == RecoveryStrategy.SELECTIVE_RESTORE:
                    await self._restore_selective(cache_service, backup_data, target_tiers)
                else:
                    raise ValueError(f"Unsupported recovery strategy: {strategy}")
            
            # Update restore operation
            restore_op.completed_at = datetime.now()
            restore_op.status = BackupStatus.COMPLETED
            restore_op.restored_entries = len(backup_data)
            
            self.logger.info(
                f"Restore completed: {restore_id}, "
                f"entries: {restore_op.restored_entries}, "
                f"strategy: {strategy.value}"
            )
            
            return restore_op
            
        except Exception as e:
            restore_op.status = BackupStatus.FAILED
            restore_op.error_message = str(e)
            self.logger.error(f"Restore failed: {restore_id}, error: {e}")
            raise
        finally:
            # Update active restores
            if restore_id in self._active_restores:
                self._active_restores[restore_id] = restore_op

    async def list_backups(
        self,
        backup_type: Optional[BackupType] = None,
        max_age_days: Optional[int] = None
    ) -> List[BackupMetadata]:
        """
        List available backups.
        
        Args:
            backup_type: Filter by backup type
            max_age_days: Maximum age in days
            
        Returns:
            List of backup metadata
        """
        backups = []
        
        if not self.backup_config.backup_directory.exists():
            return backups
        
        cutoff_date = None
        if max_age_days:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        for backup_dir in self.backup_config.backup_directory.iterdir():
            if not backup_dir.is_dir():
                continue
            
            metadata_file = backup_dir / "metadata.json"
            if not metadata_file.exists():
                continue
            
            try:
                metadata = await self._load_backup_metadata(backup_dir.name)
                if not metadata:
                    continue
                
                # Apply filters
                if backup_type and metadata.backup_type != backup_type:
                    continue
                
                if cutoff_date and metadata.timestamp < cutoff_date:
                    continue
                
                backups.append(metadata)
                
            except Exception as e:
                self.logger.warning(f"Error loading backup metadata {backup_dir.name}: {e}")
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x.timestamp, reverse=True)
        return backups

    async def delete_backup(self, backup_id: str) -> bool:
        """
        Delete a backup.
        
        Args:
            backup_id: ID of backup to delete
            
        Returns:
            True if backup was deleted successfully
        """
        backup_dir = self.backup_config.backup_directory / backup_id
        
        if not backup_dir.exists():
            self.logger.warning(f"Backup directory not found: {backup_id}")
            return False
        
        try:
            shutil.rmtree(backup_dir)
            self.logger.info(f"Deleted backup: {backup_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting backup {backup_id}: {e}")
            return False

    async def get_backup_info(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get detailed information about a backup."""
        return await self._load_backup_metadata(backup_id)

    async def verify_backup_integrity(self, backup_id: str) -> Dict[str, Any]:
        """
        Verify the integrity of a backup.
        
        Args:
            backup_id: ID of backup to verify
            
        Returns:
            Dictionary with verification results
        """
        try:
            # Load metadata
            metadata = await self._load_backup_metadata(backup_id)
            if not metadata:
                return {"valid": False, "error": "Backup metadata not found"}
            
            # Check file existence
            backup_dir = self.backup_config.backup_directory / backup_id
            backup_file = backup_dir / "cache_data.json"
            
            if not backup_file.exists():
                return {"valid": False, "error": "Backup data file not found"}
            
            # Verify checksum
            current_checksum = await self._calculate_backup_checksum(backup_file)
            checksum_valid = current_checksum == metadata.checksum
            
            # Try to load data
            data_loadable = False
            entry_count = 0
            try:
                backup_data = await self._load_backup_data(backup_id)
                data_loadable = True
                entry_count = len(backup_data)
            except Exception as e:
                return {
                    "valid": False,
                    "error": f"Cannot load backup data: {e}",
                    "checksum_valid": checksum_valid
                }
            
            # Check entry count consistency
            count_consistent = entry_count == metadata.total_entries
            
            return {
                "valid": checksum_valid and data_loadable and count_consistent,
                "checksum_valid": checksum_valid,
                "data_loadable": data_loadable,
                "entry_count_consistent": count_consistent,
                "expected_entries": metadata.total_entries,
                "actual_entries": entry_count,
                "backup_size_bytes": backup_file.stat().st_size,
                "backup_age_hours": (datetime.now() - metadata.timestamp).total_seconds() / 3600
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}

    # Internal helper methods
    
    async def _collect_full_backup_data(self, cache_service, tiers: List[str]) -> Dict[str, Any]:
        """Collect data for a full backup."""
        backup_data = {}
        
        for tier in tiers:
            tier_data = await self._get_tier_data(cache_service, tier)
            backup_data[tier] = tier_data
        
        return backup_data

    async def _collect_incremental_backup_data(
        self, cache_service, tiers: List[str], base_backup_id: str
    ) -> Dict[str, Any]:
        """Collect data for an incremental backup."""
        # Load base backup timestamp
        base_metadata = await self._load_backup_metadata(base_backup_id)
        if not base_metadata:
            raise ValueError(f"Base backup not found: {base_backup_id}")
        
        backup_data = {}
        cutoff_time = base_metadata.timestamp
        
        for tier in tiers:
            tier_data = await self._get_tier_data_since(cache_service, tier, cutoff_time)
            backup_data[tier] = tier_data
        
        return backup_data

    async def _collect_differential_backup_data(
        self, cache_service, tiers: List[str], base_backup_id: str
    ) -> Dict[str, Any]:
        """Collect data for a differential backup."""
        # For differential backup, we collect all changes since the last full backup
        # This is similar to incremental but may include more data
        return await self._collect_incremental_backup_data(cache_service, tiers, base_backup_id)

    async def _collect_snapshot_backup_data(self, cache_service, tiers: List[str]) -> Dict[str, Any]:
        """Collect data for a snapshot backup."""
        # Snapshot is similar to full backup but may be optimized for speed
        return await self._collect_full_backup_data(cache_service, tiers)

    async def _get_tier_data(self, cache_service, tier: str) -> Dict[str, Any]:
        """Get all data from a specific cache tier."""
        tier_data = {}
        
        if tier == "L1" and hasattr(cache_service, '_l1_cache') and cache_service._l1_cache:
            # Get L1 cache data
            l1_cache = cache_service._l1_cache
            if hasattr(l1_cache, '_cache'):
                for key, entry in l1_cache._cache.items():
                    if isinstance(entry, CacheEntry):
                        tier_data[key] = {
                            "data": entry.data,
                            "metadata": {
                                "size": entry.size,
                                "created_at": entry.created_at.isoformat() if entry.created_at else None,
                                "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                                "checksum": getattr(entry, 'checksum', None)
                            }
                        }
        
        elif tier == "L2" and hasattr(cache_service, '_redis') and cache_service._redis:
            # Get L2 (Redis) cache data
            redis = cache_service._redis
            keys = await redis.keys("*")
            
            for key in keys:
                if isinstance(key, bytes):
                    key = key.decode()
                
                try:
                    data = await redis.get(key)
                    if data:
                        tier_data[key] = {
                            "data": data,
                            "metadata": {
                                "ttl": await redis.ttl(key),
                                "type": await redis.type(key)
                            }
                        }
                except Exception as e:
                    self.logger.warning(f"Error backing up key {key}: {e}")
        
        return tier_data

    async def _get_tier_data_since(self, cache_service, tier: str, cutoff_time: datetime) -> Dict[str, Any]:
        """Get tier data modified since cutoff time."""
        # For incremental backups, we would need modification timestamps
        # This is a simplified version - in practice, you'd need to track modification times
        all_data = await self._get_tier_data(cache_service, tier)
        
        # Filter by creation time if available
        filtered_data = {}
        for key, entry_data in all_data.items():
            metadata = entry_data.get("metadata", {})
            created_at_str = metadata.get("created_at")
            
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    if created_at > cutoff_time:
                        filtered_data[key] = entry_data
                except Exception:
                    # Include if we can't parse timestamp
                    filtered_data[key] = entry_data
            else:
                # Include if no timestamp available
                filtered_data[key] = entry_data
        
        return filtered_data

    async def _write_backup_data(
        self, data: Dict[str, Any], backup_file: Path, compress: bool, encrypt: bool
    ) -> int:
        """Write backup data to file with optional compression and encryption."""
        # Serialize data
        json_data = json.dumps(data, default=str, indent=2)
        original_size = len(json_data.encode())
        
        # Prepare data for writing
        final_data = json_data.encode()
        
        # Compress if requested
        if compress:
            final_data = gzip.compress(final_data)
        
        # Encrypt if requested
        if encrypt:
            final_data = self._encryption_utils.encrypt_data(final_data)
        
        # Write to file
        backup_file.write_bytes(final_data)
        
        return original_size

    async def _load_backup_data(self, backup_id: str) -> Dict[str, Any]:
        """Load backup data from file."""
        backup_dir = self.backup_config.backup_directory / backup_id
        backup_file = backup_dir / "cache_data.json"
        metadata_file = backup_dir / "metadata.json"
        
        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")
        
        # Load metadata to determine processing options
        metadata = await self._load_backup_metadata(backup_id)
        if not metadata:
            raise ValueError(f"Cannot load backup metadata for {backup_id}")
        
        # Read file data
        file_data = backup_file.read_bytes()
        
        # Decrypt if needed
        if metadata.encryption_enabled:
            file_data = self._encryption_utils.decrypt_data(file_data)
        
        # Decompress if needed
        if metadata.compression_ratio < 1.0:  # Indicates compression was used
            file_data = gzip.decompress(file_data)
        
        # Parse JSON
        json_data = file_data.decode()
        return json.loads(json_data)

    async def _load_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Load backup metadata from file."""
        metadata_file = self.backup_config.backup_directory / backup_id / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            metadata_dict = json.loads(metadata_file.read_text())
            
            # Convert string timestamps back to datetime objects
            metadata_dict["timestamp"] = datetime.fromisoformat(metadata_dict["timestamp"])
            metadata_dict["backup_type"] = BackupType(metadata_dict["backup_type"])
            metadata_dict["status"] = BackupStatus(metadata_dict["status"])
            
            return BackupMetadata(**metadata_dict)
        except Exception as e:
            self.logger.error(f"Error loading backup metadata {backup_id}: {e}")
            return None

    async def _write_metadata(self, metadata: BackupMetadata, metadata_file: Path):
        """Write backup metadata to file."""
        metadata_dict = {
            "backup_id": metadata.backup_id,
            "backup_type": metadata.backup_type.value,
            "timestamp": metadata.timestamp.isoformat(),
            "cache_tiers": metadata.cache_tiers,
            "total_entries": metadata.total_entries,
            "total_size_bytes": metadata.total_size_bytes,
            "compression_ratio": metadata.compression_ratio,
            "encryption_enabled": metadata.encryption_enabled,
            "checksum": metadata.checksum,
            "duration_seconds": metadata.duration_seconds,
            "status": metadata.status.value,
            "error_message": metadata.error_message,
            "includes_metadata": metadata.includes_metadata,
            "base_backup_id": metadata.base_backup_id
        }
        
        metadata_file.write_text(json.dumps(metadata_dict, indent=2))

    async def _calculate_backup_checksum(self, backup_file: Path) -> str:
        """Calculate checksum for backup file."""
        import hashlib
        
        hash_md5 = hashlib.md5()
        with backup_file.open("rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()

    async def _verify_backup_integrity(self, backup_id: str):
        """Verify backup integrity after creation."""
        verification_result = await self.verify_backup_integrity(backup_id)
        if not verification_result.get("valid", False):
            raise ValueError(f"Backup integrity verification failed: {verification_result.get('error')}")

    async def _restore_replace_all(self, cache_service, backup_data: Dict[str, Any], target_tiers: List[str]):
        """Restore with replace all strategy."""
        for tier in target_tiers:
            if tier in backup_data:
                await self._clear_tier(cache_service, tier)
                await self._restore_tier_data(cache_service, tier, backup_data[tier])

    async def _restore_merge_preserve(self, cache_service, backup_data: Dict[str, Any], target_tiers: List[str]):
        """Restore with merge (preserve existing) strategy."""
        for tier in target_tiers:
            if tier in backup_data:
                tier_data = backup_data[tier]
                for key, entry_data in tier_data.items():
                    # Only restore if key doesn't exist
                    exists = await self._key_exists_in_tier(cache_service, tier, key)
                    if not exists:
                        await self._restore_single_entry(cache_service, tier, key, entry_data)

    async def _restore_merge_overwrite(self, cache_service, backup_data: Dict[str, Any], target_tiers: List[str]):
        """Restore with merge (overwrite existing) strategy."""
        for tier in target_tiers:
            if tier in backup_data:
                await self._restore_tier_data(cache_service, tier, backup_data[tier])

    async def _restore_selective(self, cache_service, backup_data: Dict[str, Any], target_tiers: List[str]):
        """Restore with selective strategy."""
        # This is handled at the data filtering level
        await self._restore_merge_overwrite(cache_service, backup_data, target_tiers)

    async def _clear_tier(self, cache_service, tier: str):
        """Clear all data from a cache tier."""
        if tier == "L1" and hasattr(cache_service, 'l1_cache'):
            cache_service.l1_cache.clear()
        elif tier == "L2" and hasattr(cache_service, '_redis'):
            await cache_service._redis.flushdb()

    async def _restore_tier_data(self, cache_service, tier: str, tier_data: Dict[str, Any]):
        """Restore data to a specific tier."""
        for key, entry_data in tier_data.items():
            await self._restore_single_entry(cache_service, tier, key, entry_data)

    async def _restore_single_entry(self, cache_service, tier: str, key: str, entry_data: Dict[str, Any]):
        """Restore a single cache entry."""
        try:
            data = entry_data["data"]
            metadata = entry_data.get("metadata", {})
            
            # Create cache entry
            if tier == "L1":
                # Restore to L1 cache
                if hasattr(cache_service, 'l1_cache'):
                    cache_service.l1_cache.set(key, data)
            elif tier == "L2":
                # Restore to L2 (Redis) cache
                if hasattr(cache_service, '_redis'):
                    await cache_service._redis.set(key, data)
                    
                    # Set TTL if available
                    ttl = metadata.get("ttl")
                    if ttl and ttl > 0:
                        await cache_service._redis.expire(key, ttl)
        except Exception as e:
            self.logger.error(f"Error restoring entry {key} to {tier}: {e}")

    async def _key_exists_in_tier(self, cache_service, tier: str, key: str) -> bool:
        """Check if a key exists in a specific tier."""
        try:
            if tier == "L1" and hasattr(cache_service, 'l1_cache'):
                return cache_service.l1_cache.exists(key)
            elif tier == "L2" and hasattr(cache_service, '_redis'):
                return await cache_service._redis.exists(key)
        except Exception:
            pass
        return False

    async def _periodic_cleanup(self):
        """Periodic cleanup of old backups."""
        while True:
            try:
                await asyncio.sleep(24 * 3600)  # Run daily
                await self._cleanup_old_backups()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in backup cleanup: {e}")

    async def _cleanup_old_backups(self):
        """Clean up old backups based on configuration."""
        backups = await self.list_backups()
        
        # Remove backups older than configured age
        cutoff_date = datetime.now() - timedelta(days=self.backup_config.max_backup_age_days)
        expired_backups = [b for b in backups if b.timestamp < cutoff_date]
        
        # Remove excess backups beyond max count
        if len(backups) > self.backup_config.max_backup_count:
            excess_backups = backups[self.backup_config.max_backup_count:]
            expired_backups.extend(excess_backups)
        
        # Delete expired backups
        for backup in expired_backups:
            try:
                await self.delete_backup(backup.backup_id)
                self.logger.info(f"Cleaned up expired backup: {backup.backup_id}")
            except Exception as e:
                self.logger.error(f"Error cleaning up backup {backup.backup_id}: {e}")

    async def _auto_backup_loop(self):
        """Automatic backup creation loop."""
        while True:
            try:
                # Calculate next backup times
                now = datetime.now()
                
                # Check if it's time for a full backup
                last_full_backup = await self._get_last_backup_time(BackupType.FULL)
                if not last_full_backup or (now - last_full_backup).days >= self.backup_config.full_backup_interval_days:
                    await self.create_backup(BackupType.FULL)
                
                # Check if it's time for an incremental backup
                last_incremental = await self._get_last_backup_time(BackupType.INCREMENTAL)
                hours_since_incremental = (now - last_incremental).total_seconds() / 3600 if last_incremental else float('inf')
                
                if hours_since_incremental >= self.backup_config.incremental_backup_interval_hours:
                    # Find last full backup as base
                    last_full = await self._get_last_backup(BackupType.FULL)
                    if last_full:
                        await self.create_backup(
                            BackupType.INCREMENTAL,
                            base_backup_id=last_full.backup_id
                        )
                
                # Sleep until next check (every hour)
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in auto backup loop: {e}")
                await asyncio.sleep(3600)  # Continue after error

    async def _get_last_backup_time(self, backup_type: BackupType) -> Optional[datetime]:
        """Get timestamp of last backup of specified type."""
        backups = await self.list_backups(backup_type=backup_type)
        return backups[0].timestamp if backups else None

    async def _get_last_backup(self, backup_type: BackupType) -> Optional[BackupMetadata]:
        """Get last backup of specified type."""
        backups = await self.list_backups(backup_type=backup_type)
        return backups[0] if backups else None


# Global service instance
_backup_service = None


async def get_cache_backup_service() -> CacheBackupService:
    """Get the global cache backup service instance."""
    global _backup_service
    if _backup_service is None:
        _backup_service = CacheBackupService()
        await _backup_service.initialize()
    return _backup_service


async def cleanup_backup_service():
    """Clean up the backup service instance."""
    global _backup_service
    if _backup_service is not None:
        await _backup_service.shutdown()
        _backup_service = None