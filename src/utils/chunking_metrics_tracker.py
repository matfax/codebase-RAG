"""
Chunking Metrics Tracker for monitoring code parsing and chunking performance.

This module provides comprehensive metrics tracking for the intelligent code chunking system,
including success rates per language, performance metrics, and error analytics.
"""

import logging
import time
from typing import Dict, List, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
from pathlib import Path

from models.code_chunk import ChunkType, ParseResult, CodeSyntaxError


class ChunkingMetric(NamedTuple):
    """Individual chunking operation metric."""
    timestamp: datetime
    file_path: str
    language: str
    success: bool
    chunk_count: int
    error_count: int
    processing_time_ms: float
    fallback_used: bool
    error_recovery_used: bool
    chunk_types: Dict[str, int]
    file_size_bytes: int


@dataclass
class LanguageMetrics:
    """Aggregated metrics for a programming language."""
    
    # Basic counters
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    fallback_files: int = 0
    error_recovery_files: int = 0
    
    # Chunk statistics
    total_chunks: int = 0
    chunk_types: Dict[str, int] = field(default_factory=dict)
    
    # Error statistics
    total_errors: int = 0
    error_types: Dict[str, int] = field(default_factory=dict)
    
    # Performance statistics
    total_processing_time_ms: float = 0.0
    min_processing_time_ms: float = float('inf')
    max_processing_time_ms: float = 0.0
    
    # File size statistics
    total_file_size: int = 0
    min_file_size: int = float('inf')
    max_file_size: int = 0
    
    # Quality metrics
    quality_issues: int = 0
    repaired_chunks: int = 0
    
    # Recent performance tracking (last 100 operations)
    recent_operations: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.successful_files / self.total_files) * 100
    
    def average_processing_time_ms(self) -> float:
        """Calculate average processing time."""
        if self.total_files == 0:
            return 0.0
        return self.total_processing_time_ms / self.total_files
    
    def average_chunks_per_file(self) -> float:
        """Calculate average chunks per file."""
        if self.successful_files == 0:
            return 0.0
        return self.total_chunks / self.successful_files
    
    def average_file_size(self) -> float:
        """Calculate average file size in bytes."""
        if self.total_files == 0:
            return 0.0
        return self.total_file_size / self.total_files
    
    def chunks_per_second(self) -> float:
        """Calculate chunks processed per second."""
        if self.total_processing_time_ms == 0:
            return 0.0
        return (self.total_chunks / self.total_processing_time_ms) * 1000
    
    def recent_success_rate(self) -> float:
        """Calculate success rate for recent operations."""
        if not self.recent_operations:
            return 0.0
        successful = sum(1 for op in self.recent_operations if op.success)
        return (successful / len(self.recent_operations)) * 100


@dataclass
class GlobalMetrics:
    """Global aggregated metrics across all languages."""
    
    start_time: datetime = field(default_factory=datetime.now)
    total_operations: int = 0
    successful_operations: int = 0
    
    # Language-specific metrics
    languages: Dict[str, LanguageMetrics] = field(default_factory=dict)
    
    # Performance tracking
    peak_chunks_per_second: float = 0.0
    peak_memory_usage_mb: float = 0.0
    
    # Session statistics
    session_files_processed: int = 0
    session_errors: int = 0
    session_start: datetime = field(default_factory=datetime.now)
    
    def overall_success_rate(self) -> float:
        """Calculate overall success rate across all languages."""
        if self.total_operations == 0:
            return 0.0
        return (self.successful_operations / self.total_operations) * 100
    
    def uptime(self) -> timedelta:
        """Calculate tracker uptime."""
        return datetime.now() - self.start_time


class ChunkingMetricsTracker:
    """
    Comprehensive metrics tracking system for code chunking operations.
    
    Tracks performance, success rates, error patterns, and quality metrics
    across different programming languages and file types.
    """
    
    def __init__(self, enable_persistence: bool = True, 
                 metrics_file: Optional[str] = None):
        """
        Initialize the metrics tracker.
        
        Args:
            enable_persistence: Whether to persist metrics to disk
            metrics_file: Path to metrics file (default: chunking_metrics.json)
        """
        self.logger = logging.getLogger(__name__)
        self.enable_persistence = enable_persistence
        self.metrics_file = Path(metrics_file or "chunking_metrics.json")
        
        # Initialize metrics
        self.global_metrics = GlobalMetrics()
        self.recent_metrics: deque = deque(maxlen=1000)  # Last 1000 operations
        
        # Load existing metrics if available
        if self.enable_persistence and self.metrics_file.exists():
            self._load_metrics()
        
        self.logger.info("ChunkingMetricsTracker initialized")
    
    def record_parsing_operation(self, parse_result: ParseResult, 
                                file_size_bytes: int = 0,
                                quality_issues: int = 0,
                                repaired_chunks: int = 0) -> None:
        """
        Record metrics for a parsing operation.
        
        Args:
            parse_result: Result of the parsing operation
            file_size_bytes: Size of the parsed file in bytes
            quality_issues: Number of quality issues found
            repaired_chunks: Number of chunks that were repaired
        """
        # Create chunking metric
        chunk_types_count = {}
        for chunk in parse_result.chunks:
            chunk_type = chunk.chunk_type.value
            chunk_types_count[chunk_type] = chunk_types_count.get(chunk_type, 0) + 1
        
        metric = ChunkingMetric(
            timestamp=datetime.now(),
            file_path=parse_result.file_path,
            language=parse_result.language,
            success=parse_result.parse_success and not parse_result.fallback_used,
            chunk_count=len(parse_result.chunks),
            error_count=parse_result.error_count,
            processing_time_ms=parse_result.processing_time_ms,
            fallback_used=parse_result.fallback_used,
            error_recovery_used=parse_result.error_recovery_used,
            chunk_types=chunk_types_count,
            file_size_bytes=file_size_bytes
        )
        
        # Update language-specific metrics
        self._update_language_metrics(metric, quality_issues, repaired_chunks)
        
        # Update global metrics
        self._update_global_metrics(metric)
        
        # Store recent metric
        self.recent_metrics.append(metric)
        
        # Persist metrics if enabled
        if self.enable_persistence:
            self._persist_metrics()
        
        # Log performance warning if needed
        self._check_performance_warnings(metric)
    
    def _update_language_metrics(self, metric: ChunkingMetric, 
                               quality_issues: int, repaired_chunks: int) -> None:
        """Update language-specific metrics."""
        language = metric.language
        
        if language not in self.global_metrics.languages:
            self.global_metrics.languages[language] = LanguageMetrics()
        
        lang_metrics = self.global_metrics.languages[language]
        
        # Update counters
        lang_metrics.total_files += 1
        if metric.success:
            lang_metrics.successful_files += 1
        else:
            lang_metrics.failed_files += 1
        
        if metric.fallback_used:
            lang_metrics.fallback_files += 1
        
        if metric.error_recovery_used:
            lang_metrics.error_recovery_files += 1
        
        # Update chunk statistics
        lang_metrics.total_chunks += metric.chunk_count
        for chunk_type, count in metric.chunk_types.items():
            lang_metrics.chunk_types[chunk_type] = lang_metrics.chunk_types.get(chunk_type, 0) + count
        
        # Update error statistics
        lang_metrics.total_errors += metric.error_count
        
        # Update performance statistics
        lang_metrics.total_processing_time_ms += metric.processing_time_ms
        lang_metrics.min_processing_time_ms = min(lang_metrics.min_processing_time_ms, metric.processing_time_ms)
        lang_metrics.max_processing_time_ms = max(lang_metrics.max_processing_time_ms, metric.processing_time_ms)
        
        # Update file size statistics
        lang_metrics.total_file_size += metric.file_size_bytes
        if metric.file_size_bytes > 0:
            lang_metrics.min_file_size = min(lang_metrics.min_file_size, metric.file_size_bytes)
            lang_metrics.max_file_size = max(lang_metrics.max_file_size, metric.file_size_bytes)
        
        # Update quality metrics
        lang_metrics.quality_issues += quality_issues
        lang_metrics.repaired_chunks += repaired_chunks
        
        # Add to recent operations
        lang_metrics.recent_operations.append(metric)
    
    def _update_global_metrics(self, metric: ChunkingMetric) -> None:
        """Update global metrics."""
        self.global_metrics.total_operations += 1
        if metric.success:
            self.global_metrics.successful_operations += 1
        
        # Update session statistics
        self.global_metrics.session_files_processed += 1
        if metric.error_count > 0:
            self.global_metrics.session_errors += metric.error_count
        
        # Update peak performance
        if metric.processing_time_ms > 0:
            chunks_per_sec = (metric.chunk_count / metric.processing_time_ms) * 1000
            self.global_metrics.peak_chunks_per_second = max(
                self.global_metrics.peak_chunks_per_second, chunks_per_sec
            )
    
    def _check_performance_warnings(self, metric: ChunkingMetric) -> None:
        """Check for performance warnings and log them."""
        # Warn about slow processing
        if metric.processing_time_ms > 5000:  # > 5 seconds
            self.logger.warning(
                f"Slow processing detected: {metric.file_path} took "
                f"{metric.processing_time_ms:.0f}ms ({metric.language})"
            )
        
        # Warn about high error rates
        if metric.error_count > 10:
            self.logger.warning(
                f"High error count: {metric.error_count} errors in {metric.file_path}"
            )
        
        # Warn about excessive fallback usage
        language = metric.language
        if language in self.global_metrics.languages:
            lang_metrics = self.global_metrics.languages[language]
            if lang_metrics.total_files >= 10:  # Only after processing some files
                fallback_rate = (lang_metrics.fallback_files / lang_metrics.total_files) * 100
                if fallback_rate > 20:  # > 20% fallback rate
                    self.logger.warning(
                        f"High fallback rate for {language}: {fallback_rate:.1f}%"
                    )
    
    def get_language_metrics(self, language: str) -> Optional[LanguageMetrics]:
        """Get metrics for a specific language."""
        return self.global_metrics.languages.get(language)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        summary = {
            "global": {
                "uptime_hours": self.global_metrics.uptime().total_seconds() / 3600,
                "total_operations": self.global_metrics.total_operations,
                "successful_operations": self.global_metrics.successful_operations,
                "overall_success_rate": self.global_metrics.overall_success_rate(),
                "peak_chunks_per_second": self.global_metrics.peak_chunks_per_second,
                "session_files_processed": self.global_metrics.session_files_processed,
                "session_errors": self.global_metrics.session_errors
            },
            "by_language": {},
            "recent_performance": self._get_recent_performance_summary()
        }
        
        # Add language-specific metrics
        for language, metrics in self.global_metrics.languages.items():
            summary["by_language"][language] = {
                "total_files": metrics.total_files,
                "success_rate": metrics.success_rate(),
                "recent_success_rate": metrics.recent_success_rate(),
                "average_chunks_per_file": metrics.average_chunks_per_file(),
                "average_processing_time_ms": metrics.average_processing_time_ms(),
                "chunks_per_second": metrics.chunks_per_second(),
                "fallback_rate": (metrics.fallback_files / max(1, metrics.total_files)) * 100,
                "error_recovery_rate": (metrics.error_recovery_files / max(1, metrics.total_files)) * 100,
                "total_chunks": metrics.total_chunks,
                "total_errors": metrics.total_errors,
                "chunk_types": dict(metrics.chunk_types),
                "quality_issues": metrics.quality_issues,
                "repaired_chunks": metrics.repaired_chunks,
                "average_file_size_kb": metrics.average_file_size() / 1024
            }
        
        return summary
    
    def _get_recent_performance_summary(self) -> Dict[str, Any]:
        """Get summary of recent performance metrics."""
        if not self.recent_metrics:
            return {"no_data": True}
        
        recent_window = list(self.recent_metrics)[-100:]  # Last 100 operations
        
        successful = sum(1 for m in recent_window if m.success)
        total_time = sum(m.processing_time_ms for m in recent_window)
        total_chunks = sum(m.chunk_count for m in recent_window)
        
        return {
            "operations_count": len(recent_window),
            "success_rate": (successful / len(recent_window)) * 100,
            "average_processing_time_ms": total_time / len(recent_window),
            "total_chunks_processed": total_chunks,
            "chunks_per_second": (total_chunks / max(1, total_time / 1000)),
            "languages_processed": len(set(m.language for m in recent_window))
        }
    
    def get_performance_report(self) -> str:
        """Generate a human-readable performance report."""
        metrics = self.get_all_metrics()
        
        report = ["=== Chunking Performance Report ===\n"]
        
        # Global summary
        global_m = metrics["global"]
        report.append(f"Uptime: {global_m['uptime_hours']:.1f} hours")
        report.append(f"Total Operations: {global_m['total_operations']}")
        report.append(f"Overall Success Rate: {global_m['overall_success_rate']:.1f}%")
        report.append(f"Peak Performance: {global_m['peak_chunks_per_second']:.1f} chunks/sec")
        report.append("")
        
        # Language breakdown
        report.append("=== By Language ===")
        for language, lang_m in metrics["by_language"].items():
            report.append(f"\n{language.upper()}:")
            report.append(f"  Files Processed: {lang_m['total_files']}")
            report.append(f"  Success Rate: {lang_m['success_rate']:.1f}%")
            report.append(f"  Recent Success Rate: {lang_m['recent_success_rate']:.1f}%")
            report.append(f"  Avg Chunks/File: {lang_m['average_chunks_per_file']:.1f}")
            report.append(f"  Performance: {lang_m['chunks_per_second']:.1f} chunks/sec")
            report.append(f"  Fallback Rate: {lang_m['fallback_rate']:.1f}%")
            
            if lang_m['chunk_types']:
                top_chunks = sorted(lang_m['chunk_types'].items(), key=lambda x: x[1], reverse=True)[:3]
                report.append(f"  Top Chunk Types: {', '.join(f'{k}({v})' for k, v in top_chunks)}")
        
        # Recent performance
        recent = metrics["recent_performance"]
        if not recent.get("no_data"):
            report.append(f"\n=== Recent Performance (last {recent['operations_count']} ops) ===")
            report.append(f"Success Rate: {recent['success_rate']:.1f}%")
            report.append(f"Performance: {recent['chunks_per_second']:.1f} chunks/sec")
            report.append(f"Languages: {recent['languages_processed']}")
        
        return "\n".join(report)
    
    def _persist_metrics(self) -> None:
        """Persist metrics to disk."""
        try:
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "global_metrics": {
                    "start_time": self.global_metrics.start_time.isoformat(),
                    "total_operations": self.global_metrics.total_operations,
                    "successful_operations": self.global_metrics.successful_operations,
                    "peak_chunks_per_second": self.global_metrics.peak_chunks_per_second,
                    "session_files_processed": self.global_metrics.session_files_processed,
                    "session_errors": self.global_metrics.session_errors
                },
                "language_metrics": {}
            }
            
            # Serialize language metrics
            for language, metrics in self.global_metrics.languages.items():
                metrics_data["language_metrics"][language] = {
                    "total_files": metrics.total_files,
                    "successful_files": metrics.successful_files,
                    "failed_files": metrics.failed_files,
                    "fallback_files": metrics.fallback_files,
                    "error_recovery_files": metrics.error_recovery_files,
                    "total_chunks": metrics.total_chunks,
                    "chunk_types": dict(metrics.chunk_types),
                    "total_errors": metrics.total_errors,
                    "error_types": dict(metrics.error_types),
                    "total_processing_time_ms": metrics.total_processing_time_ms,
                    "min_processing_time_ms": metrics.min_processing_time_ms if metrics.min_processing_time_ms != float('inf') else 0,
                    "max_processing_time_ms": metrics.max_processing_time_ms,
                    "total_file_size": metrics.total_file_size,
                    "min_file_size": metrics.min_file_size if metrics.min_file_size != float('inf') else 0,
                    "max_file_size": metrics.max_file_size,
                    "quality_issues": metrics.quality_issues,
                    "repaired_chunks": metrics.repaired_chunks
                }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to persist metrics: {e}")
    
    def _load_metrics(self) -> None:
        """Load metrics from disk."""
        try:
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
            
            # Load global metrics
            global_data = data.get("global_metrics", {})
            self.global_metrics.start_time = datetime.fromisoformat(
                global_data.get("start_time", datetime.now().isoformat())
            )
            self.global_metrics.total_operations = global_data.get("total_operations", 0)
            self.global_metrics.successful_operations = global_data.get("successful_operations", 0)
            self.global_metrics.peak_chunks_per_second = global_data.get("peak_chunks_per_second", 0.0)
            self.global_metrics.session_files_processed = global_data.get("session_files_processed", 0)
            self.global_metrics.session_errors = global_data.get("session_errors", 0)
            
            # Load language metrics
            lang_data = data.get("language_metrics", {})
            for language, metrics_dict in lang_data.items():
                metrics = LanguageMetrics()
                metrics.total_files = metrics_dict.get("total_files", 0)
                metrics.successful_files = metrics_dict.get("successful_files", 0)
                metrics.failed_files = metrics_dict.get("failed_files", 0)
                metrics.fallback_files = metrics_dict.get("fallback_files", 0)
                metrics.error_recovery_files = metrics_dict.get("error_recovery_files", 0)
                metrics.total_chunks = metrics_dict.get("total_chunks", 0)
                metrics.chunk_types = metrics_dict.get("chunk_types", {})
                metrics.total_errors = metrics_dict.get("total_errors", 0)
                metrics.error_types = metrics_dict.get("error_types", {})
                metrics.total_processing_time_ms = metrics_dict.get("total_processing_time_ms", 0.0)
                metrics.min_processing_time_ms = metrics_dict.get("min_processing_time_ms", float('inf'))
                metrics.max_processing_time_ms = metrics_dict.get("max_processing_time_ms", 0.0)
                metrics.total_file_size = metrics_dict.get("total_file_size", 0)
                metrics.min_file_size = metrics_dict.get("min_file_size", float('inf'))
                metrics.max_file_size = metrics_dict.get("max_file_size", 0)
                metrics.quality_issues = metrics_dict.get("quality_issues", 0)
                metrics.repaired_chunks = metrics_dict.get("repaired_chunks", 0)
                
                self.global_metrics.languages[language] = metrics
            
            self.logger.info(f"Loaded metrics for {len(self.global_metrics.languages)} languages")
            
        except Exception as e:
            self.logger.warning(f"Failed to load existing metrics: {e}")
    
    def reset_session_metrics(self) -> None:
        """Reset session-specific metrics."""
        self.global_metrics.session_files_processed = 0
        self.global_metrics.session_errors = 0
        self.global_metrics.session_start = datetime.now()
        self.logger.info("Session metrics reset")
    
    def export_metrics(self, export_path: str) -> None:
        """Export metrics to a specified file."""
        try:
            metrics = self.get_all_metrics()
            with open(export_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            self.logger.info(f"Metrics exported to {export_path}")
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")


# Global instance for easy access
chunking_metrics_tracker = ChunkingMetricsTracker()