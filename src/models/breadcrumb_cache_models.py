"""
Data models for breadcrumb resolution caching with TTL and file modification tracking.

This module provides specialized cache models for the enhanced function call detection
system, including TTL-based caching with file modification time awareness for optimal
cache invalidation strategies.
"""

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


@dataclass
class FileModificationTracker:
    """
    Tracks file modification times for cache invalidation decisions.
    """

    file_path: str
    last_modified: float
    content_hash: str | None = None
    dependencies: set[str] = field(default_factory=set)

    def __post_init__(self):
        """Compute content hash if not provided."""
        if self.content_hash is None:
            self.content_hash = self._compute_content_hash()

    def _compute_content_hash(self) -> str:
        """Compute hash of file content for change detection."""
        try:
            path = Path(self.file_path)
            if path.exists():
                content = path.read_text(encoding="utf-8", errors="ignore")
                return hashlib.sha256(content.encode()).hexdigest()[:16]
            return ""
        except Exception:
            return ""

    def is_stale(self) -> bool:
        """Check if the file has been modified since last tracking."""
        try:
            path = Path(self.file_path)
            if not path.exists():
                return True

            current_mtime = path.stat().st_mtime
            if current_mtime > self.last_modified:
                return True

            # Also check content hash for additional safety
            current_hash = self._compute_content_hash()
            return current_hash != self.content_hash
        except Exception:
            return True

    def update_tracking(self) -> bool:
        """Update tracking information. Returns True if changes were detected."""
        try:
            path = Path(self.file_path)
            if not path.exists():
                return True

            old_mtime = self.last_modified
            old_hash = self.content_hash

            self.last_modified = path.stat().st_mtime
            self.content_hash = self._compute_content_hash()

            return self.last_modified > old_mtime or self.content_hash != old_hash
        except Exception:
            return True


@dataclass
class BreadcrumbCacheEntry:
    """
    Cache entry for breadcrumb resolution results with TTL and dependency tracking.
    """

    cache_key: str
    result: Any  # BreadcrumbResolutionResult
    timestamp: float
    ttl_seconds: float
    file_dependencies: list[FileModificationTracker] = field(default_factory=list)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    confidence_score: float = 0.0

    def is_expired(self) -> bool:
        """Check if cache entry has expired based on TTL."""
        return time.time() > (self.timestamp + self.ttl_seconds)

    def is_stale(self) -> bool:
        """Check if cache entry is stale due to file modifications."""
        return any(tracker.is_stale() for tracker in self.file_dependencies)

    def is_valid(self) -> bool:
        """Check if cache entry is valid (not expired and not stale)."""
        return not self.is_expired() and not self.is_stale()

    def touch(self):
        """Update access tracking."""
        self.access_count += 1
        self.last_accessed = time.time()

    def get_age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.timestamp

    def update_dependencies(self):
        """Update all file dependency trackers."""
        changes_detected = False
        for tracker in self.file_dependencies:
            if tracker.update_tracking():
                changes_detected = True
        return changes_detected


@dataclass
class CacheStats:
    """Statistics for breadcrumb cache performance monitoring."""

    hits: int = 0
    misses: int = 0
    invalidations: int = 0
    evictions: int = 0
    total_entries: int = 0
    memory_usage_bytes: int = 0
    average_ttl_seconds: float = 0.0
    hit_rate: float = 0.0

    def record_hit(self):
        """Record a cache hit."""
        self.hits += 1
        self._update_hit_rate()

    def record_miss(self):
        """Record a cache miss."""
        self.misses += 1
        self._update_hit_rate()

    def record_invalidation(self):
        """Record a cache invalidation."""
        self.invalidations += 1

    def record_eviction(self):
        """Record a cache eviction."""
        self.evictions += 1

    def _update_hit_rate(self):
        """Update hit rate calculation."""
        total = self.hits + self.misses
        self.hit_rate = self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for reporting."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "invalidations": self.invalidations,
            "evictions": self.evictions,
            "total_entries": self.total_entries,
            "memory_usage_mb": self.memory_usage_bytes / (1024 * 1024),
            "average_ttl_seconds": self.average_ttl_seconds,
            "hit_rate": self.hit_rate,
            "total_requests": self.hits + self.misses,
        }


@dataclass
class BreadcrumbCacheConfig:
    """Configuration for breadcrumb resolution caching."""

    enabled: bool = True
    max_entries: int = 10000
    default_ttl_seconds: float = 3600.0  # 1 hour
    file_check_ttl_seconds: float = 300.0  # 5 minutes
    confidence_threshold: float = 0.7
    memory_limit_mb: int = 100
    eviction_policy: str = "LRU"
    enable_dependency_tracking: bool = True
    enable_metrics: bool = True
    cleanup_interval_seconds: float = 600.0  # 10 minutes

    def get_ttl_for_confidence(self, confidence: float) -> float:
        """Get TTL based on confidence score."""
        if confidence >= 0.9:
            return self.default_ttl_seconds * 2  # High confidence, longer TTL
        elif confidence >= 0.7:
            return self.default_ttl_seconds  # Normal TTL
        elif confidence >= 0.5:
            return self.default_ttl_seconds * 0.5  # Lower confidence, shorter TTL
        else:
            return self.default_ttl_seconds * 0.1  # Very low confidence, very short TTL

    @classmethod
    def from_env(cls) -> "BreadcrumbCacheConfig":
        """Create configuration from environment variables."""
        import os

        return cls(
            enabled=os.getenv("BREADCRUMB_CACHE_ENABLED", "true").lower() == "true",
            max_entries=int(os.getenv("BREADCRUMB_CACHE_MAX_ENTRIES", "10000")),
            default_ttl_seconds=float(os.getenv("BREADCRUMB_CACHE_TTL_SECONDS", "3600")),
            file_check_ttl_seconds=float(os.getenv("BREADCRUMB_CACHE_FILE_CHECK_TTL", "300")),
            confidence_threshold=float(os.getenv("BREADCRUMB_CACHE_CONFIDENCE_THRESHOLD", "0.7")),
            memory_limit_mb=int(os.getenv("BREADCRUMB_CACHE_MEMORY_LIMIT_MB", "100")),
            eviction_policy=os.getenv("BREADCRUMB_CACHE_EVICTION_POLICY", "LRU"),
            enable_dependency_tracking=os.getenv("BREADCRUMB_CACHE_DEPENDENCY_TRACKING", "true").lower() == "true",
            enable_metrics=os.getenv("BREADCRUMB_CACHE_METRICS", "true").lower() == "true",
            cleanup_interval_seconds=float(os.getenv("BREADCRUMB_CACHE_CLEANUP_INTERVAL", "600")),
        )
