"""
Cache consistency verification service for the Codebase RAG MCP Server.

This service provides comprehensive cache consistency verification across L1/L2 tiers,
data integrity checks, and automated consistency maintenance.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config.cache_config import CacheConfig, get_global_cache_config
from ..models.cache_models import CacheEntry, CacheMetadata
from ..services.cache_service import get_cache_service
from ..utils.encryption_utils import EncryptionUtils
from ..utils.telemetry import get_telemetry_manager, trace_cache_operation


class ConsistencyCheckLevel(Enum):
    """Levels of consistency checking."""
    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"
    DEEP = "deep"


class ConsistencyIssueType(Enum):
    """Types of consistency issues."""
    L1_L2_MISMATCH = "l1_l2_mismatch"
    CORRUPTED_DATA = "corrupted_data"
    EXPIRED_DATA = "expired_data"
    INVALID_CHECKSUM = "invalid_checksum"
    ORPHANED_ENTRY = "orphaned_entry"
    DUPLICATE_ENTRY = "duplicate_entry"
    METADATA_MISMATCH = "metadata_mismatch"


@dataclass
class ConsistencyIssue:
    """Represents a cache consistency issue."""
    issue_type: ConsistencyIssueType
    cache_key: str
    description: str
    severity: str = "medium"  # low, medium, high, critical
    l1_value: Optional[Any] = None
    l2_value: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.now)
    resolution_action: Optional[str] = None


@dataclass
class ConsistencyReport:
    """Comprehensive consistency verification report."""
    check_level: ConsistencyCheckLevel
    check_duration: float
    total_keys_checked: int
    issues_found: List[ConsistencyIssue]
    l1_stats: Dict[str, Any]
    l2_stats: Dict[str, Any]
    consistency_score: float  # 0.0 to 1.0
    recommendations: List[str]
    checked_at: datetime = field(default_factory=datetime.now)

    @property
    def is_consistent(self) -> bool:
        """Check if cache is considered consistent."""
        return self.consistency_score >= 0.95 and len([i for i in self.issues_found if i.severity in ["high", "critical"]]) == 0


class CacheConsistencyService:
    """Service for verifying and maintaining cache consistency."""

    def __init__(self, config: Optional[CacheConfig] = None):
        """Initialize the cache consistency service."""
        self.config = config or get_global_cache_config()
        self.logger = logging.getLogger(__name__)
        self._cache_service = None
        self._encryption_utils = EncryptionUtils()
        self._telemetry = get_telemetry_manager()

        # Consistency check configuration
        self.check_config = {
            "checksum_validation": True,
            "metadata_validation": True,
            "expiration_validation": True,
            "size_validation": True,
            "format_validation": True,
            "max_sample_size": 1000,  # For deep checks
            "consistency_threshold": 0.95,
        }

    async def _get_cache_service(self):
        """Get cache service instance."""
        if self._cache_service is None:
            self._cache_service = await get_cache_service()
        return self._cache_service

    @trace_cache_operation("consistency_check")
    async def verify_consistency(
        self,
        check_level: ConsistencyCheckLevel = ConsistencyCheckLevel.BASIC,
        cache_keys: Optional[List[str]] = None,
        fix_issues: bool = False
    ) -> ConsistencyReport:
        """
        Verify cache consistency across L1/L2 tiers.
        
        Args:
            check_level: Level of consistency checking to perform
            cache_keys: Specific keys to check (None for all keys)
            fix_issues: Whether to automatically fix detected issues
            
        Returns:
            Consistency report with findings and recommendations
        """
        start_time = time.time()
        cache_service = await self._get_cache_service()
        
        self.logger.info(f"Starting {check_level.value} consistency check")
        
        # Get keys to check
        if cache_keys is None:
            cache_keys = await self._get_all_cache_keys(cache_service)
        
        # Limit keys for performance
        if check_level == ConsistencyCheckLevel.DEEP and len(cache_keys) > self.check_config["max_sample_size"]:
            import random
            cache_keys = random.sample(cache_keys, self.check_config["max_sample_size"])
        
        issues = []
        l1_stats = {"total_keys": 0, "valid_entries": 0, "corrupted_entries": 0}
        l2_stats = {"total_keys": 0, "valid_entries": 0, "corrupted_entries": 0}
        
        # Perform consistency checks based on level
        if check_level in [ConsistencyCheckLevel.BASIC, ConsistencyCheckLevel.COMPREHENSIVE, ConsistencyCheckLevel.DEEP]:
            l1_l2_issues = await self._check_l1_l2_consistency(cache_service, cache_keys)
            issues.extend(l1_l2_issues)
        
        if check_level in [ConsistencyCheckLevel.COMPREHENSIVE, ConsistencyCheckLevel.DEEP]:
            checksum_issues = await self._check_data_integrity(cache_service, cache_keys)
            issues.extend(checksum_issues)
            
            expiration_issues = await self._check_expiration_consistency(cache_service, cache_keys)
            issues.extend(expiration_issues)
        
        if check_level == ConsistencyCheckLevel.DEEP:
            metadata_issues = await self._check_metadata_consistency(cache_service, cache_keys)
            issues.extend(metadata_issues)
            
            orphan_issues = await self._check_orphaned_entries(cache_service)
            issues.extend(orphan_issues)
        
        # Fix issues if requested
        if fix_issues and issues:
            await self._fix_consistency_issues(cache_service, issues)
        
        # Calculate statistics
        l1_stats, l2_stats = await self._calculate_tier_statistics(cache_service, cache_keys)
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(issues, len(cache_keys))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues, consistency_score)
        
        duration = time.time() - start_time
        
        report = ConsistencyReport(
            check_level=check_level,
            check_duration=duration,
            total_keys_checked=len(cache_keys),
            issues_found=issues,
            l1_stats=l1_stats,
            l2_stats=l2_stats,
            consistency_score=consistency_score,
            recommendations=recommendations
        )
        
        self.logger.info(
            f"Consistency check completed: {len(issues)} issues found, "
            f"score: {consistency_score:.3f}, duration: {duration:.2f}s"
        )
        
        return report

    async def _get_all_cache_keys(self, cache_service) -> List[str]:
        """Get all cache keys from both L1 and L2."""
        try:
            l1_keys = set()
            l2_keys = set()
            
            # Get L1 keys
            if hasattr(cache_service, '_l1_cache') and cache_service._l1_cache:
                l1_keys = set(cache_service._l1_cache.keys())
            
            # Get L2 keys (Redis)
            if hasattr(cache_service, '_redis') and cache_service._redis:
                redis_keys = await cache_service._redis.keys("*")
                l2_keys = {key.decode() if isinstance(key, bytes) else key for key in redis_keys}
            
            # Combine and return all unique keys
            all_keys = list(l1_keys.union(l2_keys))
            return all_keys
            
        except Exception as e:
            self.logger.error(f"Error getting cache keys: {e}")
            return []

    async def _check_l1_l2_consistency(self, cache_service, cache_keys: List[str]) -> List[ConsistencyIssue]:
        """Check consistency between L1 and L2 cache tiers."""
        issues = []
        
        for key in cache_keys:
            try:
                # Get values from both tiers
                l1_value = None
                l2_value = None
                
                # Check L1
                if hasattr(cache_service, '_l1_cache') and cache_service._l1_cache:
                    l1_entry = cache_service._l1_cache.get(key)
                    if l1_entry and isinstance(l1_entry, CacheEntry):
                        l1_value = l1_entry.data
                
                # Check L2
                if hasattr(cache_service, '_redis') and cache_service._redis:
                    l2_data = await cache_service._redis.get(key)
                    if l2_data:
                        try:
                            l2_entry = CacheEntry.from_json(l2_data)
                            l2_value = l2_entry.data
                        except Exception:
                            # Handle non-CacheEntry data
                            l2_value = l2_data
                
                # Compare values
                if l1_value is not None and l2_value is not None:
                    if not self._values_equal(l1_value, l2_value):
                        issues.append(ConsistencyIssue(
                            issue_type=ConsistencyIssueType.L1_L2_MISMATCH,
                            cache_key=key,
                            description=f"L1 and L2 values differ for key {key}",
                            severity="medium",
                            l1_value=l1_value,
                            l2_value=l2_value,
                            resolution_action="sync_l1_l2"
                        ))
                elif l1_value is not None and l2_value is None:
                    # L1 has value but L2 doesn't - could be orphaned
                    issues.append(ConsistencyIssue(
                        issue_type=ConsistencyIssueType.ORPHANED_ENTRY,
                        cache_key=key,
                        description=f"Key {key} exists in L1 but not L2",
                        severity="low",
                        l1_value=l1_value,
                        resolution_action="promote_to_l2"
                    ))
                elif l1_value is None and l2_value is not None:
                    # L2 has value but L1 doesn't - normal for evicted items
                    pass
                    
            except Exception as e:
                self.logger.error(f"Error checking L1/L2 consistency for key {key}: {e}")
                issues.append(ConsistencyIssue(
                    issue_type=ConsistencyIssueType.CORRUPTED_DATA,
                    cache_key=key,
                    description=f"Error accessing key {key}: {str(e)}",
                    severity="high",
                    resolution_action="remove_corrupted"
                ))
        
        return issues

    async def _check_data_integrity(self, cache_service, cache_keys: List[str]) -> List[ConsistencyIssue]:
        """Check data integrity using checksums."""
        issues = []
        
        for key in cache_keys:
            try:
                # Get entry from cache
                entry = await cache_service.get(key)
                if entry and isinstance(entry, CacheEntry):
                    # Verify checksum if available
                    if hasattr(entry, 'checksum') and entry.checksum:
                        calculated_checksum = self._calculate_checksum(entry.data)
                        if calculated_checksum != entry.checksum:
                            issues.append(ConsistencyIssue(
                                issue_type=ConsistencyIssueType.INVALID_CHECKSUM,
                                cache_key=key,
                                description=f"Checksum mismatch for key {key}",
                                severity="high",
                                metadata={
                                    "expected_checksum": entry.checksum,
                                    "actual_checksum": calculated_checksum
                                },
                                resolution_action="recalculate_checksum"
                            ))
                    
                    # Check for data corruption
                    if not self._is_data_valid(entry.data):
                        issues.append(ConsistencyIssue(
                            issue_type=ConsistencyIssueType.CORRUPTED_DATA,
                            cache_key=key,
                            description=f"Corrupted data detected for key {key}",
                            severity="critical",
                            resolution_action="remove_corrupted"
                        ))
                        
            except Exception as e:
                self.logger.error(f"Error checking data integrity for key {key}: {e}")
                issues.append(ConsistencyIssue(
                    issue_type=ConsistencyIssueType.CORRUPTED_DATA,
                    cache_key=key,
                    description=f"Error validating key {key}: {str(e)}",
                    severity="high",
                    resolution_action="remove_corrupted"
                ))
        
        return issues

    async def _check_expiration_consistency(self, cache_service, cache_keys: List[str]) -> List[ConsistencyIssue]:
        """Check for expired entries that haven't been cleaned up."""
        issues = []
        current_time = datetime.now()
        
        for key in cache_keys:
            try:
                entry = await cache_service.get(key)
                if entry and isinstance(entry, CacheEntry):
                    if hasattr(entry, 'expires_at') and entry.expires_at:
                        if entry.expires_at < current_time:
                            issues.append(ConsistencyIssue(
                                issue_type=ConsistencyIssueType.EXPIRED_DATA,
                                cache_key=key,
                                description=f"Expired entry found for key {key}",
                                severity="low",
                                metadata={
                                    "expired_at": entry.expires_at.isoformat(),
                                    "current_time": current_time.isoformat()
                                },
                                resolution_action="remove_expired"
                            ))
                            
            except Exception as e:
                self.logger.error(f"Error checking expiration for key {key}: {e}")
        
        return issues

    async def _check_metadata_consistency(self, cache_service, cache_keys: List[str]) -> List[ConsistencyIssue]:
        """Check metadata consistency and validity."""
        issues = []
        
        for key in cache_keys:
            try:
                entry = await cache_service.get(key)
                if entry and isinstance(entry, CacheEntry):
                    # Check metadata validity
                    if hasattr(entry, 'metadata') and entry.metadata:
                        if not self._is_metadata_valid(entry.metadata):
                            issues.append(ConsistencyIssue(
                                issue_type=ConsistencyIssueType.METADATA_MISMATCH,
                                cache_key=key,
                                description=f"Invalid metadata for key {key}",
                                severity="medium",
                                resolution_action="repair_metadata"
                            ))
                    
                    # Check size consistency
                    if hasattr(entry, 'size') and entry.size:
                        actual_size = len(str(entry.data))
                        if abs(actual_size - entry.size) > (entry.size * 0.1):  # 10% tolerance
                            issues.append(ConsistencyIssue(
                                issue_type=ConsistencyIssueType.METADATA_MISMATCH,
                                cache_key=key,
                                description=f"Size mismatch for key {key}",
                                severity="low",
                                metadata={
                                    "reported_size": entry.size,
                                    "actual_size": actual_size
                                },
                                resolution_action="update_size"
                            ))
                            
            except Exception as e:
                self.logger.error(f"Error checking metadata for key {key}: {e}")
        
        return issues

    async def _check_orphaned_entries(self, cache_service) -> List[ConsistencyIssue]:
        """Check for orphaned entries and duplicates."""
        issues = []
        
        try:
            # Get all keys from both tiers
            l1_keys = set()
            l2_keys = set()
            
            if hasattr(cache_service, '_l1_cache') and cache_service._l1_cache:
                l1_keys = set(cache_service._l1_cache.keys())
            
            if hasattr(cache_service, '_redis') and cache_service._redis:
                redis_keys = await cache_service._redis.keys("*")
                l2_keys = {key.decode() if isinstance(key, bytes) else key for key in redis_keys}
            
            # Find orphaned L1 entries (exist in L1 but not L2 for too long)
            orphaned_l1 = l1_keys - l2_keys
            for key in orphaned_l1:
                try:
                    entry = cache_service._l1_cache.get(key)
                    if entry and isinstance(entry, CacheEntry):
                        # Check if it's been orphaned for too long (configurable threshold)
                        age_threshold = timedelta(hours=1)  # Configurable
                        if hasattr(entry, 'created_at') and entry.created_at:
                            age = datetime.now() - entry.created_at
                            if age > age_threshold:
                                issues.append(ConsistencyIssue(
                                    issue_type=ConsistencyIssueType.ORPHANED_ENTRY,
                                    cache_key=key,
                                    description=f"L1 entry {key} orphaned for {age}",
                                    severity="low",
                                    resolution_action="sync_to_l2"
                                ))
                except Exception as e:
                    self.logger.error(f"Error checking orphaned L1 entry {key}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error checking orphaned entries: {e}")
        
        return issues

    async def _fix_consistency_issues(self, cache_service, issues: List[ConsistencyIssue]) -> Dict[str, int]:
        """Automatically fix consistency issues where possible."""
        fix_results = {
            "fixed": 0,
            "failed": 0,
            "skipped": 0
        }
        
        for issue in issues:
            try:
                if issue.resolution_action == "sync_l1_l2":
                    # Sync L1 and L2 values (prefer L2 as source of truth)
                    if issue.l2_value is not None:
                        await cache_service.set(issue.cache_key, issue.l2_value)
                        fix_results["fixed"] += 1
                    else:
                        fix_results["skipped"] += 1
                
                elif issue.resolution_action == "remove_corrupted":
                    # Remove corrupted entries
                    await cache_service.delete(issue.cache_key)
                    fix_results["fixed"] += 1
                
                elif issue.resolution_action == "remove_expired":
                    # Remove expired entries
                    await cache_service.delete(issue.cache_key)
                    fix_results["fixed"] += 1
                
                elif issue.resolution_action == "recalculate_checksum":
                    # Recalculate and update checksum
                    entry = await cache_service.get(issue.cache_key)
                    if entry and isinstance(entry, CacheEntry):
                        entry.checksum = self._calculate_checksum(entry.data)
                        await cache_service.set(issue.cache_key, entry)
                        fix_results["fixed"] += 1
                
                else:
                    fix_results["skipped"] += 1
                    
            except Exception as e:
                self.logger.error(f"Error fixing issue for key {issue.cache_key}: {e}")
                fix_results["failed"] += 1
        
        self.logger.info(f"Fix results: {fix_results}")
        return fix_results

    async def _calculate_tier_statistics(self, cache_service, cache_keys: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Calculate statistics for each cache tier."""
        l1_stats = {"total_keys": 0, "valid_entries": 0, "corrupted_entries": 0, "total_size": 0}
        l2_stats = {"total_keys": 0, "valid_entries": 0, "corrupted_entries": 0, "total_size": 0}
        
        for key in cache_keys:
            try:
                # Check L1
                if hasattr(cache_service, '_l1_cache') and cache_service._l1_cache:
                    if key in cache_service._l1_cache:
                        l1_stats["total_keys"] += 1
                        entry = cache_service._l1_cache[key]
                        if self._is_entry_valid(entry):
                            l1_stats["valid_entries"] += 1
                            if hasattr(entry, 'size') and entry.size:
                                l1_stats["total_size"] += entry.size
                        else:
                            l1_stats["corrupted_entries"] += 1
                
                # Check L2
                if hasattr(cache_service, '_redis') and cache_service._redis:
                    if await cache_service._redis.exists(key):
                        l2_stats["total_keys"] += 1
                        try:
                            data = await cache_service._redis.get(key)
                            if data:
                                l2_stats["valid_entries"] += 1
                                l2_stats["total_size"] += len(str(data))
                        except Exception:
                            l2_stats["corrupted_entries"] += 1
                            
            except Exception as e:
                self.logger.error(f"Error calculating statistics for key {key}: {e}")
        
        return l1_stats, l2_stats

    def _calculate_consistency_score(self, issues: List[ConsistencyIssue], total_keys: int) -> float:
        """Calculate overall consistency score (0.0 to 1.0)."""
        if total_keys == 0:
            return 1.0
        
        # Weight issues by severity
        severity_weights = {
            "low": 0.1,
            "medium": 0.3,
            "high": 0.7,
            "critical": 1.0
        }
        
        total_penalty = 0.0
        for issue in issues:
            weight = severity_weights.get(issue.severity, 0.5)
            total_penalty += weight
        
        # Calculate score
        max_penalty = total_keys * 1.0  # Assume worst case
        score = max(0.0, 1.0 - (total_penalty / max_penalty))
        
        return score

    def _generate_recommendations(self, issues: List[ConsistencyIssue], consistency_score: float) -> List[str]:
        """Generate recommendations based on consistency check results."""
        recommendations = []
        
        if consistency_score < 0.8:
            recommendations.append("Cache consistency is below acceptable threshold (80%). Consider immediate attention.")
        
        issue_counts = {}
        for issue in issues:
            issue_counts[issue.issue_type] = issue_counts.get(issue.issue_type, 0) + 1
        
        if issue_counts.get(ConsistencyIssueType.L1_L2_MISMATCH, 0) > 0:
            recommendations.append("Multiple L1/L2 mismatches detected. Review cache write strategies.")
        
        if issue_counts.get(ConsistencyIssueType.CORRUPTED_DATA, 0) > 0:
            recommendations.append("Corrupted data detected. Check storage and serialization mechanisms.")
        
        if issue_counts.get(ConsistencyIssueType.EXPIRED_DATA, 0) > 10:
            recommendations.append("Many expired entries found. Review TTL cleanup mechanisms.")
        
        if issue_counts.get(ConsistencyIssueType.ORPHANED_ENTRY, 0) > 0:
            recommendations.append("Orphaned entries detected. Review cache promotion/demotion policies.")
        
        if not recommendations:
            recommendations.append("Cache consistency is good. Continue regular monitoring.")
        
        return recommendations

    def _values_equal(self, value1: Any, value2: Any) -> bool:
        """Check if two cache values are equal."""
        try:
            # Handle different data types
            if isinstance(value1, (dict, list)) and isinstance(value2, (dict, list)):
                return json.dumps(value1, sort_keys=True) == json.dumps(value2, sort_keys=True)
            elif isinstance(value1, str) and isinstance(value2, bytes):
                return value1 == value2.decode()
            elif isinstance(value1, bytes) and isinstance(value2, str):
                return value1.decode() == value2
            else:
                return value1 == value2
        except Exception:
            return False

    def _calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for data."""
        try:
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data, sort_keys=True)
            else:
                data_str = str(data)
            
            return hashlib.sha256(data_str.encode()).hexdigest()
        except Exception:
            return ""

    def _is_data_valid(self, data: Any) -> bool:
        """Check if data is valid (not corrupted)."""
        try:
            # Basic validation - data should be serializable
            json.dumps(data, default=str)
            return True
        except Exception:
            return False

    def _is_metadata_valid(self, metadata: CacheMetadata) -> bool:
        """Check if metadata is valid."""
        try:
            # Check required fields
            if not hasattr(metadata, 'created_at') or not metadata.created_at:
                return False
            
            # Check timestamps are reasonable
            now = datetime.now()
            if metadata.created_at > now:
                return False
            
            if hasattr(metadata, 'expires_at') and metadata.expires_at:
                if metadata.expires_at <= metadata.created_at:
                    return False
            
            return True
        except Exception:
            return False

    def _is_entry_valid(self, entry: Any) -> bool:
        """Check if a cache entry is valid."""
        try:
            if not isinstance(entry, CacheEntry):
                return False
            
            # Check basic structure
            if not hasattr(entry, 'data') or entry.data is None:
                return False
            
            # Check if data is valid
            if not self._is_data_valid(entry.data):
                return False
            
            return True
        except Exception:
            return False


# Global service instance
_consistency_service = None


async def get_cache_consistency_service() -> CacheConsistencyService:
    """Get the global cache consistency service instance."""
    global _consistency_service
    if _consistency_service is None:
        _consistency_service = CacheConsistencyService()
    return _consistency_service


async def cleanup_consistency_service():
    """Clean up the consistency service instance."""
    global _consistency_service
    if _consistency_service is not None:
        _consistency_service = None