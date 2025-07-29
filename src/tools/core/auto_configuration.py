"""
Auto-Configuration Service for MCP Tools

This module provides automatic configuration detection and optimization
for MCP tools to reduce user configuration burden and improve performance.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)


@dataclass
class SystemCapabilities:
    """System capabilities and resource information."""

    cpu_count: int
    total_memory_gb: float
    available_memory_gb: float
    has_gpu: bool
    disk_space_gb: float
    network_available: bool


@dataclass
class ProjectCharacteristics:
    """Project characteristics for configuration optimization."""

    file_count: int
    total_size_mb: float
    languages: list[str]
    complexity_level: str  # simple, moderate, complex
    has_large_files: bool
    estimated_chunks: int


@dataclass
class AutoConfiguration:
    """Auto-generated configuration for MCP tools."""

    # Search configuration
    default_n_results: int
    enable_multi_modal_by_default: bool
    default_multi_modal_mode: str | None
    performance_timeout_seconds: int

    # Indexing configuration
    recommended_batch_size: int
    enable_incremental_by_default: bool
    parallel_processing_workers: int

    # Cache configuration
    enable_caching: bool
    cache_ttl_seconds: int
    max_cache_size_mb: int

    # Performance configuration
    enable_performance_monitoring: bool
    enable_optimization: bool
    optimization_level: str  # conservative, balanced, aggressive

    # System recommendations
    recommendations: list[str]
    warnings: list[str]


class AutoConfigurationService:
    """Service for automatic configuration detection and optimization."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def analyze_system_capabilities(self) -> SystemCapabilities:
        """Analyze system capabilities and resources."""
        try:
            # CPU information
            cpu_count = psutil.cpu_count(logical=True)

            # Memory information
            memory = psutil.virtual_memory()
            total_memory_gb = memory.total / (1024**3)
            available_memory_gb = memory.available / (1024**3)

            # GPU detection (basic check)
            has_gpu = self._detect_gpu()

            # Disk space (current directory)
            disk_usage = psutil.disk_usage(".")
            disk_space_gb = disk_usage.free / (1024**3)

            # Network availability
            network_available = self._check_network_connectivity()

            return SystemCapabilities(
                cpu_count=cpu_count,
                total_memory_gb=total_memory_gb,
                available_memory_gb=available_memory_gb,
                has_gpu=has_gpu,
                disk_space_gb=disk_space_gb,
                network_available=network_available,
            )

        except Exception as e:
            self.logger.warning(f"Failed to analyze system capabilities: {e}")
            # Return conservative defaults
            return SystemCapabilities(
                cpu_count=1,
                total_memory_gb=4.0,
                available_memory_gb=2.0,
                has_gpu=False,
                disk_space_gb=10.0,
                network_available=True,
            )

    async def analyze_project_characteristics(self, directory: str = ".") -> ProjectCharacteristics:
        """Analyze project characteristics for configuration optimization."""
        try:
            dir_path = Path(directory).resolve()

            # Count files and calculate size
            file_count = 0
            total_size = 0
            languages = set()
            large_files = 0

            # Common language extensions
            language_map = {
                ".py": "Python",
                ".js": "JavaScript",
                ".ts": "TypeScript",
                ".tsx": "TypeScript",
                ".jsx": "JavaScript",
                ".java": "Java",
                ".cpp": "C++",
                ".c": "C",
                ".hpp": "C++",
                ".h": "C/C++",
                ".rs": "Rust",
                ".go": "Go",
                ".php": "PHP",
                ".rb": "Ruby",
                ".cs": "C#",
                ".scala": "Scala",
                ".kt": "Kotlin",
                ".swift": "Swift",
            }

            # Analyze files
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    try:
                        file_size = file_path.stat().st_size
                        file_count += 1
                        total_size += file_size

                        # Check for large files (>5MB)
                        if file_size > 5 * 1024 * 1024:
                            large_files += 1

                        # Detect language
                        suffix = file_path.suffix.lower()
                        if suffix in language_map:
                            languages.add(language_map[suffix])

                    except (OSError, PermissionError):
                        continue

            total_size_mb = total_size / (1024 * 1024)

            # Determine complexity level
            if file_count < 50:
                complexity_level = "simple"
            elif file_count < 500:
                complexity_level = "moderate"
            else:
                complexity_level = "complex"

            # Estimate chunks (rough calculation)
            estimated_chunks = max(file_count * 2, int(total_size_mb / 0.5))  # ~0.5MB per chunk

            return ProjectCharacteristics(
                file_count=file_count,
                total_size_mb=total_size_mb,
                languages=list(languages),
                complexity_level=complexity_level,
                has_large_files=large_files > 0,
                estimated_chunks=estimated_chunks,
            )

        except Exception as e:
            self.logger.warning(f"Failed to analyze project characteristics: {e}")
            # Return conservative defaults
            return ProjectCharacteristics(
                file_count=10,
                total_size_mb=1.0,
                languages=["Python"],
                complexity_level="simple",
                has_large_files=False,
                estimated_chunks=20,
            )

    async def generate_auto_configuration(
        self,
        directory: str = ".",
        usage_pattern: str = "balanced",  # conservative, balanced, performance
    ) -> AutoConfiguration:
        """Generate automatic configuration based on system and project analysis."""

        # Analyze system and project
        system_caps = await self.analyze_system_capabilities()
        project_chars = await self.analyze_project_characteristics(directory)

        recommendations = []
        warnings = []

        # Search configuration
        if project_chars.complexity_level == "simple":
            default_n_results = 5
            enable_multi_modal_by_default = False
            default_multi_modal_mode = None
            performance_timeout_seconds = 10
        elif project_chars.complexity_level == "moderate":
            default_n_results = 10
            enable_multi_modal_by_default = True
            default_multi_modal_mode = "hybrid"
            performance_timeout_seconds = 15
        else:  # complex
            default_n_results = 15
            enable_multi_modal_by_default = True
            default_multi_modal_mode = "mix"
            performance_timeout_seconds = 20

        # Indexing configuration
        if system_caps.cpu_count >= 8:
            parallel_processing_workers = min(8, system_caps.cpu_count)
            recommended_batch_size = 100
        elif system_caps.cpu_count >= 4:
            parallel_processing_workers = 4
            recommended_batch_size = 50
        else:
            parallel_processing_workers = 2
            recommended_batch_size = 25

        # Memory-based adjustments
        if system_caps.available_memory_gb < 2.0:
            warnings.append("Low available memory detected. Consider reducing batch sizes.")
            recommended_batch_size = max(10, recommended_batch_size // 2)
            default_n_results = min(5, default_n_results)

        # Enable incremental by default for large projects
        enable_incremental_by_default = project_chars.file_count > 100

        # Cache configuration
        if system_caps.available_memory_gb >= 4.0:
            enable_caching = True
            max_cache_size_mb = min(512, int(system_caps.available_memory_gb * 128))
            cache_ttl_seconds = 3600  # 1 hour
        elif system_caps.available_memory_gb >= 2.0:
            enable_caching = True
            max_cache_size_mb = 128
            cache_ttl_seconds = 1800  # 30 minutes
        else:
            enable_caching = False
            max_cache_size_mb = 64
            cache_ttl_seconds = 600  # 10 minutes
            warnings.append("Limited memory available. Caching disabled for optimal performance.")

        # Performance configuration
        if usage_pattern == "performance":
            enable_performance_monitoring = True
            enable_optimization = True
            optimization_level = "aggressive"
        elif usage_pattern == "balanced":
            enable_performance_monitoring = True
            enable_optimization = True
            optimization_level = "balanced"
        else:  # conservative
            enable_performance_monitoring = False
            enable_optimization = False
            optimization_level = "conservative"

        # System-specific recommendations
        if system_caps.has_gpu:
            recommendations.append("GPU detected. Consider enabling GPU acceleration for embeddings.")

        if not system_caps.network_available:
            warnings.append("Network connectivity issues detected. Some features may be limited.")

        if system_caps.disk_space_gb < 5.0:
            warnings.append("Low disk space detected. Monitor cache and index storage.")

        if project_chars.has_large_files:
            recommendations.append("Large files detected. Consider using streaming processing.")

        # Performance recommendations
        if project_chars.complexity_level == "complex":
            recommendations.append("Complex project detected. Enable multi-modal search for better results.")
            recommendations.append("Consider using incremental indexing for faster updates.")

        if system_caps.cpu_count >= 8:
            recommendations.append("High CPU count detected. Enable parallel processing for optimal performance.")

        return AutoConfiguration(
            default_n_results=default_n_results,
            enable_multi_modal_by_default=enable_multi_modal_by_default,
            default_multi_modal_mode=default_multi_modal_mode,
            performance_timeout_seconds=performance_timeout_seconds,
            recommended_batch_size=recommended_batch_size,
            enable_incremental_by_default=enable_incremental_by_default,
            parallel_processing_workers=parallel_processing_workers,
            enable_caching=enable_caching,
            cache_ttl_seconds=cache_ttl_seconds,
            max_cache_size_mb=max_cache_size_mb,
            enable_performance_monitoring=enable_performance_monitoring,
            enable_optimization=enable_optimization,
            optimization_level=optimization_level,
            recommendations=recommendations,
            warnings=warnings,
        )

    def _detect_gpu(self) -> bool:
        """Detect GPU availability (basic check)."""
        try:
            # Check for NVIDIA GPU
            import subprocess

            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            if result.returncode == 0:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            pass

        # Check for Apple Silicon GPU
        try:
            if os.uname().machine == "arm64" and "Darwin" in os.uname().sysname:
                return True
        except Exception:
            pass

        return False

    def _check_network_connectivity(self) -> bool:
        """Check basic network connectivity."""
        try:
            import socket

            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False


# Global instance
_auto_config_service = None


def get_auto_configuration_service() -> AutoConfigurationService:
    """Get or create the auto-configuration service instance."""
    global _auto_config_service
    if _auto_config_service is None:
        _auto_config_service = AutoConfigurationService()
    return _auto_config_service


async def get_recommended_configuration(
    directory: str = ".",
    usage_pattern: str = "balanced",
) -> dict[str, Any]:
    """Get recommended configuration for MCP tools."""
    service = get_auto_configuration_service()
    config = await service.generate_auto_configuration(directory, usage_pattern)

    return {
        "search": {
            "default_n_results": config.default_n_results,
            "enable_multi_modal_by_default": config.enable_multi_modal_by_default,
            "default_multi_modal_mode": config.default_multi_modal_mode,
            "performance_timeout_seconds": config.performance_timeout_seconds,
        },
        "indexing": {
            "recommended_batch_size": config.recommended_batch_size,
            "enable_incremental_by_default": config.enable_incremental_by_default,
            "parallel_processing_workers": config.parallel_processing_workers,
        },
        "cache": {
            "enable_caching": config.enable_caching,
            "cache_ttl_seconds": config.cache_ttl_seconds,
            "max_cache_size_mb": config.max_cache_size_mb,
        },
        "performance": {
            "enable_performance_monitoring": config.enable_performance_monitoring,
            "enable_optimization": config.enable_optimization,
            "optimization_level": config.optimization_level,
        },
        "recommendations": config.recommendations,
        "warnings": config.warnings,
    }
