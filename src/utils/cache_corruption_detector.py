"""
Cache Corruption Detection and Recovery System.

This module provides comprehensive corruption detection, verification, and
automatic recovery capabilities for cache data integrity.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ..config.cache_config import CacheConfig


class CorruptionType(Enum):
    """Types of cache corruption."""

    SERIALIZATION_ERROR = "serialization_error"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    FORMAT_INVALID = "format_invalid"
    ENCODING_ERROR = "encoding_error"
    TRUNCATED_DATA = "truncated_data"
    METADATA_CORRUPTION = "metadata_corruption"
    TYPE_MISMATCH = "type_mismatch"
    STRUCTURE_INVALID = "structure_invalid"


class CorruptionSeverity(Enum):
    """Severity levels of corruption."""

    LOW = "low"  # Minor issues that don't affect functionality
    MEDIUM = "medium"  # Issues that may cause degraded performance
    HIGH = "high"  # Issues that cause operation failures
    CRITICAL = "critical"  # Issues that could cause system instability


class RecoveryAction(Enum):
    """Actions for corruption recovery."""

    DELETE = "delete"  # Delete corrupted data
    REGENERATE = "regenerate"  # Regenerate from source
    RESTORE_FROM_BACKUP = "restore"  # Restore from backup
    REPAIR_IN_PLACE = "repair"  # Attempt in-place repair
    QUARANTINE = "quarantine"  # Move to quarantine for analysis
    SKIP = "skip"  # Skip and continue


@dataclass
class CorruptionConfig:
    """Configuration for corruption detection and recovery."""

    # Detection settings
    enabled: bool = True
    scan_interval: float = 3600.0  # 1 hour
    batch_size: int = 100
    max_scan_time: float = 300.0  # 5 minutes max scan time

    # Verification settings
    verify_checksums: bool = True
    verify_serialization: bool = True
    verify_data_types: bool = True
    verify_data_structure: bool = True

    # Recovery settings
    auto_recovery_enabled: bool = True
    max_recovery_attempts: int = 3
    recovery_timeout: float = 30.0
    quarantine_enabled: bool = True

    # Backup settings
    backup_before_recovery: bool = True
    backup_retention_hours: int = 24

    # Monitoring
    alert_on_corruption: bool = True
    corruption_threshold: int = 10  # Alert if more than 10 corruptions found

    # Performance
    concurrent_checks: int = 5
    check_timeout: float = 10.0


@dataclass
class CorruptionReport:
    """Report of cache corruption."""

    key: str
    corruption_type: CorruptionType
    severity: CorruptionSeverity
    description: str
    detected_at: float = field(default_factory=time.time)
    data_size: int = 0
    expected_type: str | None = None
    actual_type: str | None = None
    checksum_expected: str | None = None
    checksum_actual: str | None = None
    recovery_action: RecoveryAction | None = None
    recovery_success: bool | None = None
    recovery_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "corruption_type": self.corruption_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "detected_at": self.detected_at,
            "data_size": self.data_size,
            "expected_type": self.expected_type,
            "actual_type": self.actual_type,
            "checksum_expected": self.checksum_expected,
            "checksum_actual": self.checksum_actual,
            "recovery_action": self.recovery_action.value if self.recovery_action else None,
            "recovery_success": self.recovery_success,
            "recovery_error": self.recovery_error,
        }


@dataclass
class CorruptionMetrics:
    """Metrics for corruption detection and recovery."""

    total_scans: int = 0
    total_items_scanned: int = 0
    total_corruptions_detected: int = 0
    corruptions_by_type: dict[str, int] = field(default_factory=dict)
    corruptions_by_severity: dict[str, int] = field(default_factory=dict)

    # Recovery metrics
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    recovery_actions_taken: dict[str, int] = field(default_factory=dict)

    # Performance metrics
    average_scan_time: float = 0.0
    average_check_time: float = 0.0
    last_scan_time: float | None = None

    # Recent activity
    recent_corruptions: list[dict[str, Any]] = field(default_factory=list)

    def record_corruption(self, report: CorruptionReport) -> None:
        """Record corruption detection."""
        self.total_corruptions_detected += 1

        # Update type counts
        corruption_type = report.corruption_type.value
        self.corruptions_by_type[corruption_type] = self.corruptions_by_type.get(corruption_type, 0) + 1

        # Update severity counts
        severity = report.severity.value
        self.corruptions_by_severity[severity] = self.corruptions_by_severity.get(severity, 0) + 1

        # Record recent activity
        self.recent_corruptions.append(report.to_dict())
        if len(self.recent_corruptions) > 100:  # Keep last 100
            self.recent_corruptions.pop(0)

    def record_recovery(self, report: CorruptionReport) -> None:
        """Record recovery attempt."""
        self.recovery_attempts += 1

        if report.recovery_success:
            self.successful_recoveries += 1
        else:
            self.failed_recoveries += 1

        if report.recovery_action:
            action = report.recovery_action.value
            self.recovery_actions_taken[action] = self.recovery_actions_taken.get(action, 0) + 1


class CorruptionDetector(ABC):
    """Abstract base class for corruption detectors."""

    def __init__(self, config: CorruptionConfig):
        """Initialize corruption detector."""
        self.config = config
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def detect(self, key: str, data: Any, metadata: dict | None = None) -> CorruptionReport | None:
        """Detect corruption in cache data."""
        pass

    @abstractmethod
    def get_detector_name(self) -> str:
        """Get detector name."""
        pass


class ChecksumCorruptionDetector(CorruptionDetector):
    """Detector for checksum-based corruption."""

    async def detect(self, key: str, data: Any, metadata: dict | None = None) -> CorruptionReport | None:
        """Detect checksum corruption."""
        if not self.config.verify_checksums or not metadata:
            return None

        try:
            # Calculate current checksum
            if isinstance(data, bytes):
                current_checksum = hashlib.md5(data).hexdigest()
            elif isinstance(data, str):
                current_checksum = hashlib.md5(data.encode("utf-8")).hexdigest()
            else:
                # Serialize object and calculate checksum
                serialized = pickle.dumps(data)
                current_checksum = hashlib.md5(serialized).hexdigest()

            # Get expected checksum from metadata
            expected_checksum = metadata.get("checksum")
            if not expected_checksum:
                return None  # No checksum to verify against

            # Compare checksums
            if current_checksum != expected_checksum:
                return CorruptionReport(
                    key=key,
                    corruption_type=CorruptionType.CHECKSUM_MISMATCH,
                    severity=CorruptionSeverity.HIGH,
                    description=f"Checksum mismatch for key '{key}'",
                    data_size=len(data) if isinstance(data, (str, bytes)) else 0,
                    checksum_expected=expected_checksum,
                    checksum_actual=current_checksum,
                )

            return None  # No corruption detected

        except Exception as e:
            return CorruptionReport(
                key=key,
                corruption_type=CorruptionType.CHECKSUM_MISMATCH,
                severity=CorruptionSeverity.MEDIUM,
                description=f"Failed to verify checksum for key '{key}': {e}",
                data_size=0,
            )

    def get_detector_name(self) -> str:
        """Get detector name."""
        return "checksum_detector"


class SerializationCorruptionDetector(CorruptionDetector):
    """Detector for serialization corruption."""

    async def detect(self, key: str, data: Any, metadata: dict | None = None) -> CorruptionReport | None:
        """Detect serialization corruption."""
        if not self.config.verify_serialization:
            return None

        try:
            # Test serialization round-trip
            if isinstance(data, (str, int, float, bool, list, dict)):
                # JSON serializable types
                json_str = json.dumps(data)
                deserialized = json.loads(json_str)
                if data != deserialized:
                    return CorruptionReport(
                        key=key,
                        corruption_type=CorruptionType.SERIALIZATION_ERROR,
                        severity=CorruptionSeverity.MEDIUM,
                        description=f"JSON serialization round-trip failed for key '{key}'",
                        data_size=len(str(data)),
                    )
            else:
                # Try pickle serialization
                pickled = pickle.dumps(data)
                unpickled = pickle.loads(pickled)

                # For complex objects, we can't always compare directly
                # Just check if we can serialize/deserialize without error

            return None  # No corruption detected

        except (json.JSONDecodeError, pickle.PickleError, TypeError, ValueError) as e:
            return CorruptionReport(
                key=key,
                corruption_type=CorruptionType.SERIALIZATION_ERROR,
                severity=CorruptionSeverity.HIGH,
                description=f"Serialization error for key '{key}': {e}",
                data_size=len(str(data)) if data else 0,
            )
        except Exception as e:
            return CorruptionReport(
                key=key,
                corruption_type=CorruptionType.SERIALIZATION_ERROR,
                severity=CorruptionSeverity.MEDIUM,
                description=f"Unexpected serialization check error for key '{key}': {e}",
                data_size=0,
            )

    def get_detector_name(self) -> str:
        """Get detector name."""
        return "serialization_detector"


class TypeCorruptionDetector(CorruptionDetector):
    """Detector for data type corruption."""

    async def detect(self, key: str, data: Any, metadata: dict | None = None) -> CorruptionReport | None:
        """Detect type corruption."""
        if not self.config.verify_data_types or not metadata:
            return None

        try:
            expected_type = metadata.get("data_type")
            if not expected_type:
                return None  # No type information to verify

            actual_type = type(data).__name__

            if actual_type != expected_type:
                # Check for compatible types
                if self._are_types_compatible(expected_type, actual_type):
                    return None  # Types are compatible

                return CorruptionReport(
                    key=key,
                    corruption_type=CorruptionType.TYPE_MISMATCH,
                    severity=CorruptionSeverity.MEDIUM,
                    description=f"Type mismatch for key '{key}': expected {expected_type}, got {actual_type}",
                    expected_type=expected_type,
                    actual_type=actual_type,
                    data_size=len(str(data)) if data else 0,
                )

            return None  # No corruption detected

        except Exception as e:
            return CorruptionReport(
                key=key,
                corruption_type=CorruptionType.TYPE_MISMATCH,
                severity=CorruptionSeverity.LOW,
                description=f"Type verification error for key '{key}': {e}",
                data_size=0,
            )

    def _are_types_compatible(self, expected: str, actual: str) -> bool:
        """Check if types are compatible."""
        # Define compatible type mappings
        compatible_types = {
            "int": ["int", "float"],
            "float": ["int", "float"],
            "str": ["str", "unicode"],
            "list": ["list", "tuple"],
            "tuple": ["list", "tuple"],
            "dict": ["dict"],
            "bool": ["bool"],
        }

        return actual in compatible_types.get(expected, [expected])

    def get_detector_name(self) -> str:
        """Get detector name."""
        return "type_detector"


class StructureCorruptionDetector(CorruptionDetector):
    """Detector for data structure corruption."""

    async def detect(self, key: str, data: Any, metadata: dict | None = None) -> CorruptionReport | None:
        """Detect structure corruption."""
        if not self.config.verify_data_structure:
            return None

        try:
            # Check for common structure issues
            if isinstance(data, dict):
                return await self._check_dict_structure(key, data, metadata)
            elif isinstance(data, (list, tuple)):
                return await self._check_list_structure(key, data, metadata)
            elif isinstance(data, str):
                return await self._check_string_structure(key, data, metadata)

            return None  # No issues detected

        except Exception as e:
            return CorruptionReport(
                key=key,
                corruption_type=CorruptionType.STRUCTURE_INVALID,
                severity=CorruptionSeverity.LOW,
                description=f"Structure check error for key '{key}': {e}",
                data_size=0,
            )

    async def _check_dict_structure(self, key: str, data: dict, metadata: dict | None) -> CorruptionReport | None:
        """Check dictionary structure."""
        # Check for circular references
        try:
            json.dumps(data)  # Will fail if circular references exist
        except ValueError as e:
            if "circular reference" in str(e).lower():
                return CorruptionReport(
                    key=key,
                    corruption_type=CorruptionType.STRUCTURE_INVALID,
                    severity=CorruptionSeverity.HIGH,
                    description=f"Circular reference detected in dict for key '{key}'",
                    data_size=len(str(data)),
                )

        # Check for expected keys if metadata provides schema
        if metadata and "required_keys" in metadata:
            required_keys = set(metadata["required_keys"])
            actual_keys = set(data.keys())
            missing_keys = required_keys - actual_keys

            if missing_keys:
                return CorruptionReport(
                    key=key,
                    corruption_type=CorruptionType.STRUCTURE_INVALID,
                    severity=CorruptionSeverity.MEDIUM,
                    description=f"Missing required keys in dict for key '{key}': {missing_keys}",
                    data_size=len(str(data)),
                )

        return None

    async def _check_list_structure(self, key: str, data: list | tuple, metadata: dict | None) -> CorruptionReport | None:
        """Check list/tuple structure."""
        # Check for reasonable size limits
        if len(data) > 100000:  # Arbitrary large size check
            return CorruptionReport(
                key=key,
                corruption_type=CorruptionType.STRUCTURE_INVALID,
                severity=CorruptionSeverity.MEDIUM,
                description=f"Unusually large list/tuple for key '{key}': {len(data)} items",
                data_size=len(data),
            )

        # Check for nested depth issues
        max_depth = self._get_nested_depth(data)
        if max_depth > 100:  # Very deep nesting might indicate corruption
            return CorruptionReport(
                key=key,
                corruption_type=CorruptionType.STRUCTURE_INVALID,
                severity=CorruptionSeverity.MEDIUM,
                description=f"Excessive nesting depth for key '{key}': {max_depth}",
                data_size=len(data),
            )

        return None

    async def _check_string_structure(self, key: str, data: str, metadata: dict | None) -> CorruptionReport | None:
        """Check string structure."""
        # Check for encoding issues
        try:
            data.encode("utf-8")
        except UnicodeEncodeError:
            return CorruptionReport(
                key=key,
                corruption_type=CorruptionType.ENCODING_ERROR,
                severity=CorruptionSeverity.HIGH,
                description=f"Unicode encoding error for string key '{key}'",
                data_size=len(data),
            )

        # Check for truncation (null bytes in middle of string)
        if "\x00" in data:
            return CorruptionReport(
                key=key,
                corruption_type=CorruptionType.TRUNCATED_DATA,
                severity=CorruptionSeverity.HIGH,
                description=f"Null bytes detected in string for key '{key}'",
                data_size=len(data),
            )

        return None

    def _get_nested_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Get maximum nesting depth of an object."""
        if current_depth > 100:  # Prevent infinite recursion
            return current_depth

        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_nested_depth(value, current_depth + 1) for value in obj.values())
        elif isinstance(obj, (list, tuple)):
            if not obj:
                return current_depth
            return max(self._get_nested_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth

    def get_detector_name(self) -> str:
        """Get detector name."""
        return "structure_detector"


class CorruptionRecoveryEngine:
    """Engine for recovering from cache corruption."""

    def __init__(self, config: CorruptionConfig):
        """Initialize corruption recovery engine."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._cache_service_ref: weakref.ReferenceType | None = None
        self._backup_storage: dict[str, Any] = {}  # Simple in-memory backup

    def register_cache_service(self, cache_service) -> None:
        """Register cache service for recovery operations."""
        self._cache_service_ref = weakref.ref(cache_service)

    async def recover(self, report: CorruptionReport) -> bool:
        """Recover from corruption."""
        try:
            # Backup corrupted data if enabled
            if self.config.backup_before_recovery:
                await self._backup_corrupted_data(report)

            # Determine recovery action
            recovery_action = self._determine_recovery_action(report)
            report.recovery_action = recovery_action

            # Execute recovery action
            success = await self._execute_recovery_action(report, recovery_action)
            report.recovery_success = success

            if success:
                self.logger.info(f"Successfully recovered from corruption: {report.key} using {recovery_action.value}")
            else:
                self.logger.error(f"Failed to recover from corruption: {report.key}")

            return success

        except Exception as e:
            report.recovery_error = str(e)
            report.recovery_success = False
            self.logger.error(f"Recovery failed for key '{report.key}': {e}")
            return False

    def _determine_recovery_action(self, report: CorruptionReport) -> RecoveryAction:
        """Determine appropriate recovery action based on corruption type and severity."""
        if report.severity == CorruptionSeverity.CRITICAL:
            return RecoveryAction.DELETE
        elif report.corruption_type == CorruptionType.CHECKSUM_MISMATCH:
            return RecoveryAction.REGENERATE
        elif report.corruption_type == CorruptionType.SERIALIZATION_ERROR:
            return RecoveryAction.DELETE
        elif report.corruption_type == CorruptionType.TYPE_MISMATCH:
            return RecoveryAction.REPAIR_IN_PLACE
        elif report.corruption_type == CorruptionType.STRUCTURE_INVALID:
            return RecoveryAction.DELETE
        elif report.corruption_type == CorruptionType.ENCODING_ERROR:
            return RecoveryAction.DELETE
        elif report.corruption_type == CorruptionType.TRUNCATED_DATA:
            return RecoveryAction.DELETE
        else:
            return RecoveryAction.QUARANTINE

    async def _execute_recovery_action(self, report: CorruptionReport, action: RecoveryAction) -> bool:
        """Execute the specified recovery action."""
        cache_service = self._cache_service_ref() if self._cache_service_ref else None
        if not cache_service:
            return False

        try:
            if action == RecoveryAction.DELETE:
                return await self._delete_corrupted_data(report, cache_service)
            elif action == RecoveryAction.REGENERATE:
                return await self._regenerate_data(report, cache_service)
            elif action == RecoveryAction.RESTORE_FROM_BACKUP:
                return await self._restore_from_backup(report, cache_service)
            elif action == RecoveryAction.REPAIR_IN_PLACE:
                return await self._repair_in_place(report, cache_service)
            elif action == RecoveryAction.QUARANTINE:
                return await self._quarantine_data(report, cache_service)
            elif action == RecoveryAction.SKIP:
                return True  # Do nothing
            else:
                return False

        except Exception as e:
            self.logger.error(f"Failed to execute recovery action {action.value}: {e}")
            return False

    async def _delete_corrupted_data(self, report: CorruptionReport, cache_service) -> bool:
        """Delete corrupted data."""
        try:
            if hasattr(cache_service, "delete"):
                return await cache_service.delete(report.key)
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete corrupted data for key '{report.key}': {e}")
            return False

    async def _regenerate_data(self, report: CorruptionReport, cache_service) -> bool:
        """Regenerate data from source."""
        # This would need application-specific logic to regenerate data
        # For now, we'll just delete the corrupted data
        self.logger.info(f"Regeneration not implemented, deleting corrupted data: {report.key}")
        return await self._delete_corrupted_data(report, cache_service)

    async def _restore_from_backup(self, report: CorruptionReport, cache_service) -> bool:
        """Restore data from backup."""
        if report.key in self._backup_storage:
            try:
                if hasattr(cache_service, "set"):
                    backup_data = self._backup_storage[report.key]
                    return await cache_service.set(report.key, backup_data)
            except Exception as e:
                self.logger.error(f"Failed to restore from backup for key '{report.key}': {e}")

        return False

    async def _repair_in_place(self, report: CorruptionReport, cache_service) -> bool:
        """Attempt to repair data in place."""
        # For type mismatches, we might be able to convert data
        if report.corruption_type == CorruptionType.TYPE_MISMATCH:
            try:
                # This is a simplified example
                # Real implementation would need more sophisticated type conversion
                self.logger.info(f"In-place repair not fully implemented for key '{report.key}'")
                return False
            except Exception:
                return False

        return False

    async def _quarantine_data(self, report: CorruptionReport, cache_service) -> bool:
        """Quarantine corrupted data for analysis."""
        if self.config.quarantine_enabled:
            try:
                # Move data to quarantine (simplified implementation)
                quarantine_key = f"quarantine:{report.key}:{int(time.time())}"

                # Get the original data
                if hasattr(cache_service, "get"):
                    data = await cache_service.get(report.key)
                    if data is not None:
                        # Store in quarantine with metadata
                        quarantine_data = {
                            "original_key": report.key,
                            "data": data,
                            "corruption_report": report.to_dict(),
                            "quarantined_at": time.time(),
                        }

                        if hasattr(cache_service, "set"):
                            await cache_service.set(quarantine_key, quarantine_data)

                        # Delete original corrupted data
                        await self._delete_corrupted_data(report, cache_service)

                        self.logger.info(f"Quarantined corrupted data: {report.key} -> {quarantine_key}")
                        return True

            except Exception as e:
                self.logger.error(f"Failed to quarantine data for key '{report.key}': {e}")

        return False

    async def _backup_corrupted_data(self, report: CorruptionReport) -> None:
        """Backup corrupted data before recovery."""
        try:
            cache_service = self._cache_service_ref() if self._cache_service_ref else None
            if cache_service and hasattr(cache_service, "get"):
                data = await cache_service.get(report.key)
                if data is not None:
                    backup_key = f"backup:{report.key}:{int(time.time())}"
                    self._backup_storage[backup_key] = data

                    # Clean up old backups
                    self._cleanup_old_backups()

        except Exception as e:
            self.logger.error(f"Failed to backup corrupted data for key '{report.key}': {e}")

    def _cleanup_old_backups(self) -> None:
        """Clean up old backup data."""
        cutoff_time = time.time() - (self.config.backup_retention_hours * 3600)

        keys_to_remove = []
        for key in self._backup_storage:
            if key.startswith("backup:"):
                try:
                    timestamp = float(key.split(":")[-1])
                    if timestamp < cutoff_time:
                        keys_to_remove.append(key)
                except (ValueError, IndexError):
                    pass

        for key in keys_to_remove:
            del self._backup_storage[key]


class CacheCorruptionDetectionSystem:
    """
    Comprehensive cache corruption detection and recovery system.

    This system continuously monitors cache data for corruption,
    detects various types of data integrity issues, and automatically
    recovers from corruption when possible.
    """

    def __init__(self, config: CorruptionConfig | None = None):
        """Initialize corruption detection system."""
        self.config = config or CorruptionConfig()
        self.logger = logging.getLogger(__name__)

        # System state
        self.enabled = self.config.enabled
        self.running = False

        # Detection components
        self.detectors: list[CorruptionDetector] = []
        self._initialize_detectors()

        # Recovery component
        self.recovery_engine = CorruptionRecoveryEngine(self.config)

        # Metrics and monitoring
        self.metrics = CorruptionMetrics()

        # Tasks
        self._scan_task: asyncio.Task | None = None

        # Cache service reference
        self._cache_service_ref: weakref.ReferenceType | None = None

    def _initialize_detectors(self) -> None:
        """Initialize corruption detectors."""
        self.detectors = [
            ChecksumCorruptionDetector(self.config),
            SerializationCorruptionDetector(self.config),
            TypeCorruptionDetector(self.config),
            StructureCorruptionDetector(self.config),
        ]

    async def start(self) -> None:
        """Start the corruption detection system."""
        if self.running:
            return

        if not self.enabled:
            self.logger.info("Corruption detection system is disabled")
            return

        self.running = True

        # Start scanning task
        self._scan_task = asyncio.create_task(self._scanning_loop())

        self.logger.info("Corruption detection system started")

    async def stop(self) -> None:
        """Stop the corruption detection system."""
        if not self.running:
            return

        self.running = False

        # Cancel scanning task
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Corruption detection system stopped")

    def register_cache_service(self, cache_service) -> None:
        """Register cache service for corruption detection."""
        self._cache_service_ref = weakref.ref(cache_service)
        self.recovery_engine.register_cache_service(cache_service)

    async def scan_key(self, key: str, data: Any, metadata: dict | None = None) -> list[CorruptionReport]:
        """Scan a specific key for corruption."""
        reports = []

        for detector in self.detectors:
            try:
                report = await asyncio.wait_for(detector.detect(key, data, metadata), timeout=self.config.check_timeout)

                if report:
                    reports.append(report)
                    self.metrics.record_corruption(report)

                    # Attempt automatic recovery if enabled
                    if self.config.auto_recovery_enabled:
                        recovery_success = await self.recovery_engine.recover(report)
                        self.metrics.record_recovery(report)

            except asyncio.TimeoutError:
                self.logger.warning(f"Corruption check timeout for key '{key}' using {detector.get_detector_name()}")
            except Exception as e:
                self.logger.error(f"Corruption check failed for key '{key}' using {detector.get_detector_name()}: {e}")

        return reports

    async def scan_cache(self, key_pattern: str = "*") -> list[CorruptionReport]:
        """Scan cache for corruption."""
        cache_service = self._cache_service_ref() if self._cache_service_ref else None
        if not cache_service:
            self.logger.error("No cache service registered for corruption scanning")
            return []

        all_reports = []
        start_time = time.time()
        self.metrics.total_scans += 1

        try:
            # Get keys to scan
            keys_to_scan = await self._get_keys_to_scan(cache_service, key_pattern)

            # Process keys in batches
            for i in range(0, len(keys_to_scan), self.config.batch_size):
                if time.time() - start_time > self.config.max_scan_time:
                    self.logger.warning("Scan time limit reached, stopping early")
                    break

                batch = keys_to_scan[i : i + self.config.batch_size]
                batch_reports = await self._scan_batch(cache_service, batch)
                all_reports.extend(batch_reports)

            # Update metrics
            scan_duration = time.time() - start_time
            self.metrics.total_items_scanned += len(keys_to_scan)

            if self.metrics.total_scans == 1:
                self.metrics.average_scan_time = scan_duration
            else:
                self.metrics.average_scan_time = (
                    self.metrics.average_scan_time * (self.metrics.total_scans - 1) + scan_duration
                ) / self.metrics.total_scans

            self.metrics.last_scan_time = time.time()

            self.logger.info(f"Corruption scan completed: {len(all_reports)} corruptions found in {scan_duration:.2f}s")

            # Alert if corruption threshold exceeded
            if self.config.alert_on_corruption and len(all_reports) > self.config.corruption_threshold:
                await self._trigger_corruption_alert(len(all_reports))

        except Exception as e:
            self.logger.error(f"Corruption scan failed: {e}")

        return all_reports

    async def _get_keys_to_scan(self, cache_service, pattern: str) -> list[str]:
        """Get list of keys to scan."""
        # This would need to be implemented based on the specific cache service
        # For now, return an empty list as we don't have access to key enumeration
        # In a real implementation, this would use Redis SCAN or similar
        return []

    async def _scan_batch(self, cache_service, keys: list[str]) -> list[CorruptionReport]:
        """Scan a batch of keys for corruption."""
        all_reports = []

        # Create semaphore for concurrent checks
        semaphore = asyncio.Semaphore(self.config.concurrent_checks)

        async def scan_single_key(key: str) -> list[CorruptionReport]:
            async with semaphore:
                try:
                    # Get data and metadata
                    data = await cache_service.get(key) if hasattr(cache_service, "get") else None
                    metadata = await self._get_metadata(cache_service, key)

                    if data is not None:
                        return await self.scan_key(key, data, metadata)

                except Exception as e:
                    self.logger.error(f"Failed to scan key '{key}': {e}")

                return []

        # Execute scans concurrently
        tasks = [scan_single_key(key) for key in keys]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        for result in results:
            if isinstance(result, list):
                all_reports.extend(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Scan task failed: {result}")

        return all_reports

    async def _get_metadata(self, cache_service, key: str) -> dict | None:
        """Get metadata for a cache key."""
        # This would need to be implemented based on how metadata is stored
        # For now, return None
        return None

    async def _scanning_loop(self) -> None:
        """Main scanning loop."""
        while self.running:
            try:
                await asyncio.sleep(self.config.scan_interval)

                if self.enabled:
                    await self.scan_cache()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in scanning loop: {e}")

    async def _trigger_corruption_alert(self, corruption_count: int) -> None:
        """Trigger alert for high corruption count."""
        self.logger.critical(
            f"High corruption count detected: {corruption_count} corruptions found " f"(threshold: {self.config.corruption_threshold})"
        )

        # This could integrate with external alerting systems
        # For now, just log the alert

    def get_metrics(self) -> CorruptionMetrics:
        """Get corruption detection metrics."""
        return self.metrics

    def get_system_status(self) -> dict[str, Any]:
        """Get system status."""
        return {
            "enabled": self.enabled,
            "running": self.running,
            "detectors_count": len(self.detectors),
            "metrics": {
                "total_scans": self.metrics.total_scans,
                "total_items_scanned": self.metrics.total_items_scanned,
                "total_corruptions_detected": self.metrics.total_corruptions_detected,
                "successful_recoveries": self.metrics.successful_recoveries,
                "failed_recoveries": self.metrics.failed_recoveries,
                "last_scan_time": self.metrics.last_scan_time,
                "average_scan_time": self.metrics.average_scan_time,
            },
            "recent_corruptions": self.metrics.recent_corruptions[-10:],  # Last 10
        }


# Factory function
async def create_corruption_detection_system(config: CorruptionConfig | None = None) -> CacheCorruptionDetectionSystem:
    """Create and start corruption detection system."""
    system = CacheCorruptionDetectionSystem(config)
    await system.start()
    return system


# Global corruption detection system instance
_global_corruption_system: CacheCorruptionDetectionSystem | None = None


async def get_corruption_detection_system(config: CorruptionConfig | None = None) -> CacheCorruptionDetectionSystem:
    """Get or create global corruption detection system."""
    global _global_corruption_system
    if _global_corruption_system is None:
        _global_corruption_system = await create_corruption_detection_system(config)
    return _global_corruption_system


async def shutdown_corruption_detection_system() -> None:
    """Shutdown global corruption detection system."""
    global _global_corruption_system
    if _global_corruption_system:
        await _global_corruption_system.stop()
        _global_corruption_system = None
