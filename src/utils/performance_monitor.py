"""
Performance monitoring and progress tracking utilities for codebase indexing operations.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psutil


@dataclass
class ProcessingStats:
    """Statistics for a processing operation."""

    total_items: int
    processed_items: int = 0
    failed_items: int = 0
    start_time: float = field(default_factory=time.time)
    stage_times: dict[str, float] = field(default_factory=dict)
    memory_usage_mb: float = 0.0

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_items == 0:
            return 100.0
        return (self.processed_items / self.total_items) * 100.0

    @property
    def processing_rate(self) -> float:
        """Calculate items processed per second."""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.processed_items / elapsed

    @property
    def eta_seconds(self) -> float | None:
        """Calculate estimated time to completion in seconds."""
        remaining_items = self.total_items - self.processed_items
        if remaining_items <= 0 or self.processing_rate == 0:
            return None
        return remaining_items / self.processing_rate

    @property
    def eta_formatted(self) -> str:
        """Format ETA as human-readable string."""
        eta = self.eta_seconds
        if eta is None:
            return "N/A"

        if eta < 60:
            return f"{eta:.0f}s"
        elif eta < 3600:
            return f"{eta / 60:.0f}m {eta % 60:.0f}s"
        else:
            hours = eta // 3600
            minutes = (eta % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"


class ProgressTracker:
    """Thread-safe progress tracker with ETA estimation and memory monitoring."""

    def __init__(self, total_items: int, description: str = "Processing"):
        self.stats = ProcessingStats(total_items=total_items)
        self.description = description
        self._lock = threading.Lock()
        self._stage_start_times: dict[str, float] = {}

    def start_stage(self, stage_name: str) -> None:
        """Mark the start of a processing stage."""
        with self._lock:
            self._stage_start_times[stage_name] = time.time()

    def end_stage(self, stage_name: str) -> None:
        """Mark the end of a processing stage and record timing."""
        with self._lock:
            if stage_name in self._stage_start_times:
                duration = time.time() - self._stage_start_times[stage_name]
                self.stats.stage_times[stage_name] = duration
                del self._stage_start_times[stage_name]

    def increment_processed(self, count: int = 1) -> None:
        """Increment the count of processed items."""
        with self._lock:
            self.stats.processed_items += count
            self._update_memory_usage()

    def increment_failed(self, count: int = 1) -> None:
        """Increment the count of failed items."""
        with self._lock:
            self.stats.failed_items += count
            self._update_memory_usage()

    def _update_memory_usage(self) -> None:
        """Update current memory usage."""
        try:
            process = psutil.Process()
            self.stats.memory_usage_mb = process.memory_info().rss / 1024 / 1024
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Fallback if psutil fails
            self.stats.memory_usage_mb = 0.0

    def get_progress_summary(self) -> dict[str, Any]:
        """Get a comprehensive progress summary."""
        with self._lock:
            elapsed_time = time.time() - self.stats.start_time

            return {
                "description": self.description,
                "total_items": self.stats.total_items,
                "processed_items": self.stats.processed_items,
                "failed_items": self.stats.failed_items,
                "completion_percentage": round(self.stats.completion_percentage, 1),
                "processing_rate": round(self.stats.processing_rate, 2),
                "eta": self.stats.eta_formatted,
                "elapsed_time": f"{elapsed_time:.1f}s",
                "memory_usage_mb": round(self.stats.memory_usage_mb, 1),
                "stage_times": {k: f"{v:.2f}s" for k, v in self.stats.stage_times.items()},
            }

    def log_progress(self, logger, level: str = "info") -> None:
        """Log current progress to the provided logger."""
        summary = self.get_progress_summary()

        message = (
            f"{self.description}: {summary['processed_items']}/{summary['total_items']} "
            f"({summary['completion_percentage']}%) - "
            f"{summary['processing_rate']} items/s - "
            f"ETA: {summary['eta']} - "
            f"Memory: {summary['memory_usage_mb']}MB"
        )

        if summary["failed_items"] > 0:
            message += f" - Failed: {summary['failed_items']}"

        getattr(logger, level)(message)

    def is_complete(self) -> bool:
        """Check if processing is complete."""
        with self._lock:
            return self.stats.processed_items + self.stats.failed_items >= self.stats.total_items


class MemoryMonitor:
    """Memory usage monitor with configurable warning thresholds."""

    def __init__(self, warning_threshold_mb: float = 1000.0):
        self.warning_threshold_mb = warning_threshold_mb
        self._last_warning_time = 0.0
        self._warning_cooldown = 30.0  # Seconds between warnings
        self._monitoring = False
        self._monitor_thread = None

    def check_memory_usage(self, logger) -> dict[str, Any]:
        """Check current memory usage and log warnings if needed."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = process.memory_percent()

            memory_stats = {
                "memory_mb": round(memory_mb, 1),
                "memory_percent": round(memory_percent, 1),
                "threshold_mb": self.warning_threshold_mb,
                "above_threshold": memory_mb > self.warning_threshold_mb,
            }

            # Log warning if above threshold and cooldown has passed
            current_time = time.time()
            if memory_mb > self.warning_threshold_mb and current_time - self._last_warning_time > self._warning_cooldown:
                logger.warning(
                    f"Memory usage above threshold: {memory_mb:.1f}MB " f"(threshold: {self.warning_threshold_mb}MB, {memory_percent:.1f}%)"
                )
                self._last_warning_time = current_time

            return memory_stats

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.error(f"Failed to get memory info: {e}")
            return {
                "memory_mb": 0.0,
                "memory_percent": 0.0,
                "threshold_mb": self.warning_threshold_mb,
                "above_threshold": False,
                "error": str(e),
            }

    def start_monitoring(self):
        """Start memory monitoring."""
        self._monitoring = True

    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring = False

    def get_current_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0

    def get_system_memory_info(self) -> dict[str, Any]:
        """Get system-wide memory information."""
        try:
            memory = psutil.virtual_memory()
            return {
                "total_mb": round(memory.total / 1024 / 1024, 1),
                "available_mb": round(memory.available / 1024 / 1024, 1),
                "used_mb": round(memory.used / 1024 / 1024, 1),
                "percent_used": round(memory.percent, 1),
            }
        except Exception as e:
            return {"error": str(e)}


def format_duration(seconds: float) -> str:
    """Format duration in seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes:.0f}m {remaining_seconds:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


def format_memory_size(bytes_value: int) -> str:
    """Format bytes as human-readable memory size."""
    if bytes_value < 1024:
        return f"{bytes_value}B"
    elif bytes_value < 1024 * 1024:
        return f"{bytes_value / 1024:.1f}KB"
    elif bytes_value < 1024 * 1024 * 1024:
        return f"{bytes_value / (1024 * 1024):.1f}MB"
    else:
        return f"{bytes_value / (1024 * 1024 * 1024):.1f}GB"
