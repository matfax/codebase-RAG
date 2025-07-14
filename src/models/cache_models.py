"""
Cache data models and structures for the Codebase RAG MCP Server.

This module defines comprehensive cache data models including cache entries,
metadata structures, statistics, metrics, and validation/integrity checking.
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Union
from uuid import uuid4

from utils.cache_utils import CompressionFormat, SerializationFormat


class CacheEntryStatus(Enum):
    """Status of a cache entry."""

    VALID = "valid"
    EXPIRED = "expired"
    CORRUPTED = "corrupted"
    INVALIDATED = "invalidated"
    PENDING = "pending"


class CacheEntryType(Enum):
    """Type of cache entry."""

    EMBEDDING = "embedding"
    SEARCH_RESULT = "search_result"
    PROJECT_INFO = "project_info"
    FILE_METADATA = "file_metadata"
    PARSED_CONTENT = "parsed_content"
    ANALYTICS = "analytics"
    HEALTH_CHECK = "health_check"


class CacheCompressionLevel(Enum):
    """Compression levels for cache entries."""

    NONE = 0
    LOW = 1
    MEDIUM = 5
    HIGH = 9


@dataclass
class CacheEntryMetadata:
    """Metadata associated with a cache entry."""

    # Core metadata
    key: str
    entry_type: CacheEntryType
    created_at: float
    last_accessed: float
    last_modified: float
    expires_at: float | None = None

    # Size and storage information
    size_bytes: int = 0
    compressed_size_bytes: int | None = None
    compression_format: CompressionFormat = CompressionFormat.NONE
    compression_level: CacheCompressionLevel = CacheCompressionLevel.NONE
    serialization_format: SerializationFormat = SerializationFormat.JSON

    # Access and usage statistics
    access_count: int = 0
    hit_count: int = 0
    miss_count: int = 0

    # Integrity and validation
    content_hash: str = ""
    checksum: str = ""
    integrity_verified: bool = False
    last_integrity_check: float = 0.0

    # Cache behavior
    ttl_seconds: int = 3600
    is_persistent: bool = False
    is_encrypted: bool = False

    # Additional metadata
    tags: set[str] = field(default_factory=set)
    custom_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize derived fields."""
        if self.created_at == 0:
            self.created_at = time.time()
        if self.last_accessed == 0:
            self.last_accessed = self.created_at
        if self.last_modified == 0:
            self.last_modified = self.created_at
        if self.expires_at is None and self.ttl_seconds > 0:
            self.expires_at = self.created_at + self.ttl_seconds

    def is_expired(self) -> bool:
        """Check if the cache entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def time_to_expiry(self) -> float:
        """Get time until expiry in seconds."""
        if self.expires_at is None:
            return float("inf")
        return max(0, self.expires_at - time.time())

    def update_access_stats(self, hit: bool = True) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()
        if hit:
            self.hit_count += 1
        else:
            self.miss_count += 1

    def get_hit_rate(self) -> float:
        """Calculate hit rate."""
        total_requests = self.hit_count + self.miss_count
        if total_requests == 0:
            return 0.0
        return self.hit_count / total_requests

    def get_compression_ratio(self) -> float:
        """Calculate compression ratio."""
        if self.compressed_size_bytes is None or self.size_bytes == 0:
            return 1.0
        return self.compressed_size_bytes / self.size_bytes

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert set to list for JSON serialization
        data["tags"] = list(self.tags)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntryMetadata":
        """Create from dictionary."""
        # Convert list back to set
        if "tags" in data and isinstance(data["tags"], list):
            data["tags"] = set(data["tags"])
        return cls(**data)


@dataclass
class CacheEntry:
    """Complete cache entry with data and metadata."""

    # Core data
    key: str
    value: Any
    metadata: CacheEntryMetadata

    # Additional fields
    created_at: float = field(default_factory=time.time)
    status: CacheEntryStatus = CacheEntryStatus.VALID

    def __post_init__(self):
        """Initialize derived fields."""
        if self.metadata.key != self.key:
            self.metadata.key = self.key

    def is_valid(self) -> bool:
        """Check if the cache entry is valid."""
        return self.status == CacheEntryStatus.VALID and not self.metadata.is_expired()

    def get_serialized_value(self) -> bytes:
        """Get serialized value based on serialization format."""
        if self.metadata.serialization_format == SerializationFormat.JSON:
            return json.dumps(self.value).encode("utf-8")
        elif self.metadata.serialization_format == SerializationFormat.PICKLE:
            import pickle

            return pickle.dumps(self.value)
        else:
            # Default to JSON
            return json.dumps(self.value).encode("utf-8")

    def calculate_content_hash(self) -> str:
        """Calculate content hash for integrity checking."""
        serialized_value = self.get_serialized_value()
        return hashlib.sha256(serialized_value).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify cache entry integrity."""
        if not self.metadata.content_hash:
            return True  # No hash to verify against

        current_hash = self.calculate_content_hash()
        is_valid = current_hash == self.metadata.content_hash

        self.metadata.integrity_verified = is_valid
        self.metadata.last_integrity_check = time.time()

        if not is_valid:
            self.status = CacheEntryStatus.CORRUPTED

        return is_valid

    def update_metadata(self, **kwargs) -> None:
        """Update metadata fields."""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)

        self.metadata.last_modified = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "metadata": self.metadata.to_dict(),
            "created_at": self.created_at,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        metadata = CacheEntryMetadata.from_dict(data["metadata"])
        status = CacheEntryStatus(data["status"])

        return cls(
            key=data["key"],
            value=data["value"],
            metadata=metadata,
            created_at=data["created_at"],
            status=status,
        )


@dataclass
class CacheStatistics:
    """Cache statistics and metrics."""

    # Basic statistics
    total_entries: int = 0
    total_size_bytes: int = 0
    total_compressed_size_bytes: int = 0

    # Access statistics
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    # Performance metrics
    average_access_time_ms: float = 0.0
    average_serialization_time_ms: float = 0.0
    average_compression_time_ms: float = 0.0

    # Entry type distribution
    entry_type_distribution: dict[str, int] = field(default_factory=dict)

    # Size distribution
    size_distribution: dict[str, int] = field(default_factory=dict)

    # TTL distribution
    ttl_distribution: dict[str, int] = field(default_factory=dict)

    # Compression statistics
    compression_savings_bytes: int = 0
    compression_ratio: float = 1.0

    # Error statistics
    corruption_count: int = 0
    expiration_count: int = 0
    eviction_count: int = 0

    # Timestamp information
    last_updated: float = field(default_factory=time.time)
    collection_start_time: float = field(default_factory=time.time)

    def get_hit_rate(self) -> float:
        """Calculate overall hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    def get_miss_rate(self) -> float:
        """Calculate overall miss rate."""
        return 1.0 - self.get_hit_rate()

    def get_average_entry_size(self) -> float:
        """Calculate average entry size."""
        if self.total_entries == 0:
            return 0.0
        return self.total_size_bytes / self.total_entries

    def get_compression_savings_ratio(self) -> float:
        """Calculate compression savings ratio."""
        if self.total_size_bytes == 0:
            return 0.0
        return self.compression_savings_bytes / self.total_size_bytes

    def update_from_entry(self, entry: CacheEntry, operation: str) -> None:
        """Update statistics from a cache entry operation."""
        if operation == "add":
            self.total_entries += 1
            self.total_size_bytes += entry.metadata.size_bytes
            if entry.metadata.compressed_size_bytes:
                self.total_compressed_size_bytes += entry.metadata.compressed_size_bytes
                self.compression_savings_bytes += entry.metadata.size_bytes - entry.metadata.compressed_size_bytes

            # Update type distribution
            entry_type = entry.metadata.entry_type.value
            self.entry_type_distribution[entry_type] = self.entry_type_distribution.get(entry_type, 0) + 1

            # Update size distribution
            size_range = self._get_size_range(entry.metadata.size_bytes)
            self.size_distribution[size_range] = self.size_distribution.get(size_range, 0) + 1

            # Update TTL distribution
            ttl_range = self._get_ttl_range(entry.metadata.ttl_seconds)
            self.ttl_distribution[ttl_range] = self.ttl_distribution.get(ttl_range, 0) + 1

        elif operation == "remove":
            self.total_entries = max(0, self.total_entries - 1)
            self.total_size_bytes = max(0, self.total_size_bytes - entry.metadata.size_bytes)
            if entry.metadata.compressed_size_bytes:
                self.total_compressed_size_bytes = max(0, self.total_compressed_size_bytes - entry.metadata.compressed_size_bytes)

        elif operation == "hit":
            self.cache_hits += 1
            self.total_requests += 1

        elif operation == "miss":
            self.cache_misses += 1
            self.total_requests += 1

        elif operation == "corruption":
            self.corruption_count += 1

        elif operation == "expiration":
            self.expiration_count += 1

        elif operation == "eviction":
            self.eviction_count += 1

        self.last_updated = time.time()

    def _get_size_range(self, size_bytes: int) -> str:
        """Get size range category."""
        if size_bytes < 1024:
            return "< 1KB"
        elif size_bytes < 1024 * 1024:
            return "1KB - 1MB"
        elif size_bytes < 10 * 1024 * 1024:
            return "1MB - 10MB"
        elif size_bytes < 100 * 1024 * 1024:
            return "10MB - 100MB"
        else:
            return "> 100MB"

    def _get_ttl_range(self, ttl_seconds: int) -> str:
        """Get TTL range category."""
        if ttl_seconds < 60:
            return "< 1 minute"
        elif ttl_seconds < 3600:
            return "1 minute - 1 hour"
        elif ttl_seconds < 86400:
            return "1 hour - 1 day"
        elif ttl_seconds < 604800:
            return "1 day - 1 week"
        else:
            return "> 1 week"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheStatistics":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CacheMetrics:
    """Real-time cache metrics."""

    # Performance metrics
    operations_per_second: float = 0.0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Memory usage
    memory_usage_bytes: int = 0
    memory_usage_percent: float = 0.0
    memory_pressure: bool = False

    # Connection metrics
    active_connections: int = 0
    failed_connections: int = 0
    connection_pool_usage: float = 0.0

    # Cache levels (L1 memory, L2 Redis)
    l1_hit_rate: float = 0.0
    l2_hit_rate: float = 0.0
    l1_size_bytes: int = 0
    l2_size_bytes: int = 0

    # Error rates
    error_rate: float = 0.0
    timeout_rate: float = 0.0

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheMetrics":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CacheHealthStatus:
    """Cache health status information."""

    # Overall health
    is_healthy: bool = True
    health_score: float = 1.0  # 0.0 to 1.0

    # Component health
    redis_healthy: bool = True
    memory_cache_healthy: bool = True

    # Connection status
    redis_connected: bool = True
    redis_latency_ms: float = 0.0

    # Performance indicators
    hit_rate: float = 0.0
    error_rate: float = 0.0
    memory_usage_percent: float = 0.0

    # Issues and warnings
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Timestamp
    timestamp: float = field(default_factory=time.time)

    def add_issue(self, issue: str) -> None:
        """Add a health issue."""
        self.issues.append(issue)
        self.is_healthy = False
        self.health_score = max(0.0, self.health_score - 0.2)

    def add_warning(self, warning: str) -> None:
        """Add a health warning."""
        self.warnings.append(warning)
        self.health_score = max(0.0, self.health_score - 0.1)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheHealthStatus":
        """Create from dictionary."""
        return cls(**data)


class CacheValidator:
    """Cache validation and integrity checking utilities."""

    def __init__(self):
        """Initialize the cache validator."""
        self.logger = logging.getLogger(__name__)

    def validate_entry(self, entry: CacheEntry) -> tuple[bool, list[str]]:
        """
        Validate a cache entry.

        Args:
            entry: Cache entry to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check basic structure
        if not entry.key or not entry.key.strip():
            errors.append("Entry key is empty")

        if entry.value is None:
            errors.append("Entry value is None")

        if not entry.metadata:
            errors.append("Entry metadata is missing")

        # Check metadata consistency
        if entry.metadata and entry.metadata.key != entry.key:
            errors.append(f"Metadata key mismatch: {entry.metadata.key} != {entry.key}")

        # Check expiration
        if entry.metadata and entry.metadata.is_expired():
            errors.append("Entry is expired")

        # Check size consistency
        if entry.metadata and entry.metadata.size_bytes > 0:
            try:
                actual_size = len(entry.get_serialized_value())
                if abs(actual_size - entry.metadata.size_bytes) > 100:  # Allow small variance
                    errors.append(f"Size mismatch: actual={actual_size}, recorded={entry.metadata.size_bytes}")
            except Exception as e:
                errors.append(f"Failed to calculate size: {e}")

        # Check integrity
        if entry.metadata and entry.metadata.content_hash:
            try:
                if not entry.verify_integrity():
                    errors.append("Integrity check failed")
            except Exception as e:
                errors.append(f"Integrity check error: {e}")

        return len(errors) == 0, errors

    def validate_metadata(self, metadata: CacheEntryMetadata) -> tuple[bool, list[str]]:
        """
        Validate cache entry metadata.

        Args:
            metadata: Cache entry metadata to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required fields
        if not metadata.key or not metadata.key.strip():
            errors.append("Metadata key is empty")

        if metadata.created_at <= 0:
            errors.append("Invalid created_at timestamp")

        if metadata.last_accessed <= 0:
            errors.append("Invalid last_accessed timestamp")

        # Check consistency
        if metadata.last_accessed < metadata.created_at:
            errors.append("last_accessed is before created_at")

        if metadata.expires_at and metadata.expires_at <= metadata.created_at:
            errors.append("expires_at is before created_at")

        # Check size fields
        if metadata.size_bytes < 0:
            errors.append("Negative size_bytes")

        if metadata.compressed_size_bytes and metadata.compressed_size_bytes < 0:
            errors.append("Negative compressed_size_bytes")

        if metadata.compressed_size_bytes and metadata.compressed_size_bytes > metadata.size_bytes:
            errors.append("Compressed size larger than original size")

        # Check counts
        if metadata.access_count < 0:
            errors.append("Negative access_count")

        if metadata.hit_count < 0:
            errors.append("Negative hit_count")

        if metadata.miss_count < 0:
            errors.append("Negative miss_count")

        if metadata.access_count < (metadata.hit_count + metadata.miss_count):
            errors.append("Access count less than hit + miss count")

        return len(errors) == 0, errors

    def validate_statistics(self, stats: CacheStatistics) -> tuple[bool, list[str]]:
        """
        Validate cache statistics.

        Args:
            stats: Cache statistics to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check non-negative values
        if stats.total_entries < 0:
            errors.append("Negative total_entries")

        if stats.total_size_bytes < 0:
            errors.append("Negative total_size_bytes")

        if stats.cache_hits < 0:
            errors.append("Negative cache_hits")

        if stats.cache_misses < 0:
            errors.append("Negative cache_misses")

        # Check consistency
        if stats.total_requests != (stats.cache_hits + stats.cache_misses):
            errors.append("Total requests != hits + misses")

        if stats.total_compressed_size_bytes > stats.total_size_bytes:
            errors.append("Total compressed size > total size")

        # Check percentages
        hit_rate = stats.get_hit_rate()
        if hit_rate < 0 or hit_rate > 1:
            errors.append(f"Invalid hit rate: {hit_rate}")

        return len(errors) == 0, errors

    def repair_entry(self, entry: CacheEntry) -> CacheEntry:
        """
        Attempt to repair a cache entry.

        Args:
            entry: Cache entry to repair

        Returns:
            CacheEntry: Repaired cache entry
        """
        # Update metadata key if mismatched
        if entry.metadata.key != entry.key:
            entry.metadata.key = entry.key

        # Fix size if incorrect
        try:
            actual_size = len(entry.get_serialized_value())
            if entry.metadata.size_bytes != actual_size:
                entry.metadata.size_bytes = actual_size
        except Exception as e:
            self.logger.warning(f"Failed to fix size for entry {entry.key}: {e}")

        # Update content hash
        try:
            entry.metadata.content_hash = entry.calculate_content_hash()
        except Exception as e:
            self.logger.warning(f"Failed to update content hash for entry {entry.key}: {e}")

        # Fix timestamps
        current_time = time.time()
        if entry.metadata.created_at <= 0:
            entry.metadata.created_at = current_time

        if entry.metadata.last_accessed <= 0:
            entry.metadata.last_accessed = entry.metadata.created_at

        if entry.metadata.last_accessed < entry.metadata.created_at:
            entry.metadata.last_accessed = entry.metadata.created_at

        # Update expiration
        if entry.metadata.expires_at and entry.metadata.expires_at <= entry.metadata.created_at:
            entry.metadata.expires_at = entry.metadata.created_at + entry.metadata.ttl_seconds

        return entry


# Global cache validator instance
_cache_validator: CacheValidator | None = None


def get_cache_validator() -> CacheValidator:
    """
    Get the global cache validator instance.

    Returns:
        CacheValidator: The global cache validator instance
    """
    global _cache_validator
    if _cache_validator is None:
        _cache_validator = CacheValidator()
    return _cache_validator


def create_cache_entry(
    key: str, value: Any, entry_type: CacheEntryType, ttl_seconds: int = 3600, tags: set[str] | None = None, **kwargs
) -> CacheEntry:
    """
    Create a new cache entry with proper metadata.

    Args:
        key: Cache key
        value: Cache value
        entry_type: Type of cache entry
        ttl_seconds: Time to live in seconds
        tags: Optional tags for the entry
        **kwargs: Additional metadata fields

    Returns:
        CacheEntry: New cache entry
    """
    # Calculate size
    try:
        if isinstance(value, str):
            size_bytes = len(value.encode("utf-8"))
        elif isinstance(value, bytes):
            size_bytes = len(value)
        else:
            size_bytes = len(json.dumps(value).encode("utf-8"))
    except Exception:
        size_bytes = 0

    # Create metadata
    metadata = CacheEntryMetadata(
        key=key,
        entry_type=entry_type,
        created_at=time.time(),
        last_accessed=time.time(),
        last_modified=time.time(),
        size_bytes=size_bytes,
        ttl_seconds=ttl_seconds,
        tags=tags or set(),
        **kwargs,
    )

    # Create entry
    entry = CacheEntry(
        key=key,
        value=value,
        metadata=metadata,
    )

    # Set content hash
    entry.metadata.content_hash = entry.calculate_content_hash()

    return entry
