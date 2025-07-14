"""
Time estimation service for indexing operations.

This service provides intelligent time estimation for indexing operations
to help users understand expected completion times and make informed
decisions about using manual indexing tools for heavy operations.
"""

import logging
from dataclasses import dataclass
from typing import Any

from services.project_analysis_service import ProjectAnalysisService
from src.utils.file_system_utils import format_file_size


@dataclass
class IndexingEstimate:
    """
    Represents an indexing time estimate with detailed breakdown.
    """

    file_count: int
    total_size_mb: float
    estimated_minutes: float
    complexity_factors: dict[str, float]
    recommendation: str
    confidence_level: str  # 'high', 'medium', 'low'

    def exceeds_threshold(self, threshold_minutes: float = 5.0) -> bool:
        """Check if estimated time exceeds threshold."""
        return self.estimated_minutes > threshold_minutes

    def get_recommendation_message(self) -> str:
        """Get user-friendly recommendation message."""
        if self.exceeds_threshold():
            return (
                f"⚠️ Estimated indexing time: {self.estimated_minutes:.1f} minutes\n"
                f"Consider using the manual indexing tool:\n"
                f"  python manual_indexing.py -d <directory> -m clear_existing"
            )
        else:
            return f"✅ Quick operation: ~{self.estimated_minutes:.1f} minutes"


class TimeEstimatorService:
    """
    Service for estimating indexing operation completion times.

    This service analyzes repository characteristics and provides intelligent
    time estimates to help users choose the best indexing approach.
    """

    def __init__(self):
        """Initialize the time estimator service."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.project_analysis = ProjectAnalysisService()

        # Base performance metrics (files per minute)
        # These are conservative estimates that can be adjusted based on system performance
        self.base_rates = {
            "small_files": 150,  # < 10KB files per minute
            "medium_files": 80,  # 10KB - 100KB files per minute
            "large_files": 30,  # > 100KB files per minute
            "binary_files": 200,  # Binary files (mostly skipped) per minute
        }

        # Complexity factors that affect processing speed
        self.complexity_factors = {
            "high_file_count": 0.9,  # Slight slowdown for many files
            "large_codebase": 0.8,  # Slowdown for very large codebases
            "many_languages": 0.95,  # Slight overhead for multiple languages
            "deep_directories": 0.9,  # File system traversal overhead
            "concurrent_load": 0.7,  # System under load
        }

    def estimate_indexing_time(self, directory: str, mode: str = "clear_existing") -> IndexingEstimate:
        """
        Estimate indexing time for a directory.

        Args:
            directory: Target directory path
            mode: Indexing mode ('clear_existing' or 'incremental')

        Returns:
            IndexingEstimate with detailed time prediction
        """
        try:
            # Analyze repository structure
            analysis = self.project_analysis.analyze_directory_structure(directory)

            if mode == "incremental":
                return self._estimate_incremental_time(directory, analysis)
            else:
                return self._estimate_full_indexing_time(directory, analysis)

        except Exception as e:
            self.logger.error(f"Error estimating indexing time: {e}")
            # Return conservative fallback estimate
            return IndexingEstimate(
                file_count=0,
                total_size_mb=0.0,
                estimated_minutes=10.0,  # Conservative fallback
                complexity_factors={},
                recommendation="Unable to estimate - proceed with caution",
                confidence_level="low",
            )

    def _estimate_full_indexing_time(self, directory: str, analysis: dict[str, Any]) -> IndexingEstimate:
        """
        Estimate time for full indexing operation.

        Args:
            directory: Target directory
            analysis: Repository analysis results

        Returns:
            IndexingEstimate for full indexing
        """
        file_count = analysis.get("relevant_files", 0)
        size_analysis = analysis.get("size_analysis", {})
        total_size_mb = size_analysis.get("total_size_mb", 0.0)

        if file_count == 0:
            return IndexingEstimate(
                file_count=0,
                total_size_mb=0.0,
                estimated_minutes=0.0,
                complexity_factors={},
                recommendation="No files to index",
                confidence_level="high",
            )

        # Calculate base time based on file sizes
        small_files = size_analysis.get("small_files", 0)
        medium_files = size_analysis.get("medium_files", 0)
        large_files = size_analysis.get("large_files", 0)

        # Estimate processing time
        base_minutes = (
            small_files / self.base_rates["small_files"]
            + medium_files / self.base_rates["medium_files"]
            + large_files / self.base_rates["large_files"]
        )

        # Apply complexity factors
        applied_factors = self._calculate_complexity_factors(analysis)
        total_factor = 1.0
        for _factor_name, factor_value in applied_factors.items():
            total_factor *= factor_value

        estimated_minutes = base_minutes / total_factor

        # Add embedding generation overhead (typically 20-30% of processing time)
        embedding_overhead = 0.25
        estimated_minutes *= 1 + embedding_overhead

        # Determine confidence level
        confidence = self._determine_confidence_level(file_count, total_size_mb, applied_factors)

        # Generate recommendation
        if estimated_minutes > 5.0:
            recommendation = "manual_tool_recommended"
        elif estimated_minutes > 2.0:
            recommendation = "consider_manual_tool"
        else:
            recommendation = "quick_operation"

        return IndexingEstimate(
            file_count=file_count,
            total_size_mb=total_size_mb,
            estimated_minutes=estimated_minutes,
            complexity_factors=applied_factors,
            recommendation=recommendation,
            confidence_level=confidence,
        )

    def _estimate_incremental_time(self, directory: str, analysis: dict[str, Any]) -> IndexingEstimate:
        """
        Estimate time for incremental indexing operation.

        Args:
            directory: Target directory
            analysis: Repository analysis results

        Returns:
            IndexingEstimate for incremental indexing
        """
        # For incremental mode, we need to estimate change rate
        # This is a rough estimate since we don't know actual changes without detection

        total_files = analysis.get("relevant_files", 0)

        # Assume typical development change rate (5-15% of files modified)
        estimated_change_rate = 0.10  # 10% change rate
        estimated_changed_files = int(total_files * estimated_change_rate)

        # Minimum files to check (change detection overhead)
        min_files_to_process = max(1, estimated_changed_files)

        # Use smaller file count for time calculation
        modified_analysis = analysis.copy()
        size_analysis = modified_analysis.get("size_analysis", {})

        # Scale down file counts based on change rate
        for key in ["small_files", "medium_files", "large_files"]:
            if key in size_analysis:
                size_analysis[key] = int(size_analysis[key] * estimated_change_rate)

        modified_analysis["relevant_files"] = min_files_to_process
        modified_analysis["size_analysis"] = size_analysis

        # Calculate base estimate for changed files
        base_estimate = self._estimate_full_indexing_time(directory, modified_analysis)

        # Add change detection overhead (usually very fast)
        change_detection_overhead = 0.5  # 30 seconds
        base_estimate.estimated_minutes += change_detection_overhead

        # Update recommendation for incremental mode
        base_estimate.recommendation = "incremental_fast"

        return base_estimate

    def _calculate_complexity_factors(self, analysis: dict[str, Any]) -> dict[str, float]:
        """
        Calculate complexity factors that affect indexing speed.

        Args:
            analysis: Repository analysis results

        Returns:
            Dictionary of applicable complexity factors
        """
        applied_factors = {}

        file_count = analysis.get("relevant_files", 0)
        total_size_mb = analysis.get("size_analysis", {}).get("total_size_mb", 0.0)
        language_breakdown = analysis.get("language_breakdown", {})

        # High file count factor
        if file_count > 1000:
            applied_factors["high_file_count"] = self.complexity_factors["high_file_count"]

        # Large codebase factor
        if total_size_mb > 100:  # > 100MB
            applied_factors["large_codebase"] = self.complexity_factors["large_codebase"]

        # Multiple languages factor
        if len(language_breakdown) > 5:
            applied_factors["many_languages"] = self.complexity_factors["many_languages"]

        # System load factor (simplified check)
        if self._detect_system_load():
            applied_factors["concurrent_load"] = self.complexity_factors["concurrent_load"]

        return applied_factors

    def _determine_confidence_level(self, file_count: int, size_mb: float, factors: dict[str, float]) -> str:
        """
        Determine confidence level of the estimate.

        Args:
            file_count: Number of files
            size_mb: Total size in MB
            factors: Applied complexity factors

        Returns:
            Confidence level: 'high', 'medium', or 'low'
        """
        # Start with high confidence
        confidence_score = 1.0

        # Reduce confidence for edge cases
        if file_count > 5000:
            confidence_score -= 0.3

        if size_mb > 500:  # Very large codebase
            confidence_score -= 0.2

        if len(factors) > 2:  # Many complexity factors
            confidence_score -= 0.2

        if confidence_score >= 0.7:
            return "high"
        elif confidence_score >= 0.4:
            return "medium"
        else:
            return "low"

    def _detect_system_load(self) -> bool:
        """
        Detect if system is under heavy load.

        Returns:
            True if system appears to be under load
        """
        try:
            import psutil

            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                return True

            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return True

            return False

        except ImportError:
            # psutil not available, assume normal load
            return False
        except Exception:
            # Error checking system load, assume normal
            return False

    def get_manual_tool_command(self, directory: str, mode: str = "clear_existing") -> str:
        """
        Generate manual indexing tool command for user.

        Args:
            directory: Target directory
            mode: Indexing mode

        Returns:
            Command string for manual indexing
        """
        return f'python manual_indexing.py -d "{directory}" -m {mode}'

    def should_recommend_manual_tool(self, estimate: IndexingEstimate, threshold_minutes: float = 5.0) -> bool:
        """
        Determine if manual tool should be recommended.

        Args:
            estimate: Indexing estimate
            threshold_minutes: Threshold for recommendation

        Returns:
            True if manual tool should be recommended
        """
        return estimate.exceeds_threshold(threshold_minutes)

    def get_detailed_estimate_summary(self, estimate: IndexingEstimate) -> str:
        """
        Generate detailed estimate summary for user display.

        Args:
            estimate: Indexing estimate

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("📊 INDEXING TIME ESTIMATE")
        lines.append("=" * 30)
        lines.append(f"📁 Files to process: {estimate.file_count:,}")
        lines.append(f"💾 Total size: {format_file_size(int(estimate.total_size_mb * 1024 * 1024))}")
        lines.append(f"⏱️  Estimated time: {estimate.estimated_minutes:.1f} minutes")
        lines.append(f"🎯 Confidence: {estimate.confidence_level}")

        if estimate.complexity_factors:
            lines.append("\n🔍 Complexity factors:")
            for factor, value in estimate.complexity_factors.items():
                impact = int((1 - value) * 100)
                lines.append(f"   • {factor.replace('_', ' ').title()}: +{impact}% time")

        lines.append(f"\n💡 {estimate.get_recommendation_message()}")

        return "\n".join(lines)
