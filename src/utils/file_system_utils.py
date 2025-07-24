"""
File system utility functions for incremental indexing.

This module provides file system operations needed for change detection
and file management in the incremental indexing workflow.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def get_file_mtime(file_path: str | Path) -> float | None:
    """
    Get file modification time safely.

    Args:
        file_path: Path to the file

    Returns:
        Modification time as Unix timestamp, or None if file doesn't exist
    """
    try:
        return Path(file_path).stat().st_mtime
    except (OSError, FileNotFoundError):
        return None


def get_file_size(file_path: str | Path) -> int | None:
    """
    Get file size safely.

    Args:
        file_path: Path to the file

    Returns:
        File size in bytes, or None if file doesn't exist
    """
    try:
        return Path(file_path).stat().st_size
    except (OSError, FileNotFoundError):
        return None


def file_exists(file_path: str | Path) -> bool:
    """
    Check if file exists.

    Args:
        file_path: Path to check

    Returns:
        True if file exists and is accessible
    """
    try:
        return Path(file_path).exists()
    except (OSError, PermissionError):
        return False


def is_file_readable(file_path: str | Path) -> bool:
    """
    Check if file is readable.

    Args:
        file_path: Path to the file

    Returns:
        True if file exists and is readable
    """
    try:
        path = Path(file_path)
        return path.exists() and path.is_file() and os.access(path, os.R_OK)
    except (OSError, PermissionError):
        return False


def get_file_stats(
    file_path: str | Path,
) -> dict[str, float | int] | None:
    """
    Get comprehensive file statistics.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with file statistics, or None if file doesn't exist
    """
    try:
        path = Path(file_path)
        file_stat = path.stat()

        return {
            "size": file_stat.st_size,
            "mtime": file_stat.st_mtime,
            "ctime": file_stat.st_ctime,
            "atime": file_stat.st_atime,
            "mode": file_stat.st_mode,
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "is_symlink": path.is_symlink(),
        }
    except (OSError, FileNotFoundError):
        return None


def batch_get_file_stats(
    file_paths: list[str | Path],
) -> dict[str, dict[str, float | int] | None]:
    """
    Get file statistics for multiple files efficiently.

    Args:
        file_paths: List of file paths

    Returns:
        Dictionary mapping file paths to their statistics
    """
    results = {}

    for file_path in file_paths:
        str_path = str(file_path)
        results[str_path] = get_file_stats(file_path)

    return results


def compare_file_times(file_path1: str | Path, file_path2: str | Path) -> str | None:
    """
    Compare modification times of two files.

    Args:
        file_path1: First file path
        file_path2: Second file path

    Returns:
        'newer', 'older', 'same', or None if either file doesn't exist
    """
    mtime1 = get_file_mtime(file_path1)
    mtime2 = get_file_mtime(file_path2)

    if mtime1 is None or mtime2 is None:
        return None

    if abs(mtime1 - mtime2) < 0.001:  # Account for floating point precision
        return "same"
    elif mtime1 > mtime2:
        return "newer"
    else:
        return "older"


def find_files_newer_than(directory: str | Path, timestamp: float, patterns: list[str] | None = None) -> list[str]:
    """
    Find files in directory that are newer than given timestamp.

    Args:
        directory: Directory to search
        timestamp: Unix timestamp to compare against
        patterns: Optional list of file patterns to match (glob-style)

    Returns:
        List of file paths that are newer than timestamp
    """
    newer_files = []
    dir_path = Path(directory)

    if not dir_path.exists() or not dir_path.is_dir():
        return newer_files

    try:
        # If patterns specified, use them; otherwise find all files
        if patterns:
            files_to_check = []
            for pattern in patterns:
                files_to_check.extend(dir_path.rglob(pattern))
        else:
            files_to_check = [p for p in dir_path.rglob("*") if p.is_file()]

        for file_path in files_to_check:
            try:
                if file_path.stat().st_mtime > timestamp:
                    newer_files.append(str(file_path))
            except (OSError, FileNotFoundError):
                continue

    except Exception as e:
        logger.warning(f"Error scanning directory {directory}: {e}")

    return newer_files


def create_directory_if_not_exists(directory: str | Path) -> bool:
    """
    Create directory if it doesn't exist.

    Args:
        directory: Directory path to create

    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        return False


def is_file_binary(file_path: str | Path) -> bool | None:
    """
    Check if file appears to be binary based on content sampling.

    Args:
        file_path: Path to the file

    Returns:
        True if file appears binary, False if text, None if cannot determine
    """
    try:
        with open(file_path, "rb") as f:
            # Read first 8KB to check for binary content
            chunk = f.read(8192)

            if not chunk:
                return False  # Empty files are considered text

            # Check for null bytes (common in binary files)
            if b"\x00" in chunk:
                return True

            # Check ratio of printable characters
            try:
                chunk.decode("utf-8")
                return False  # Successfully decoded as UTF-8, likely text
            except UnicodeDecodeError:
                # Try other common encodings
                for encoding in ["latin-1", "cp1252", "ascii"]:
                    try:
                        chunk.decode(encoding)
                        return False  # Successfully decoded, likely text
                    except UnicodeDecodeError:
                        continue

                return True  # Could not decode with common encodings, likely binary

    except (OSError, FileNotFoundError, PermissionError):
        return None


def get_relative_path(file_path: str | Path, base_path: str | Path) -> str | None:
    """
    Get relative path from base path to file path.

    Args:
        file_path: Target file path
        base_path: Base path to calculate relative path from

    Returns:
        Relative path string, or None if file is outside base path
    """
    try:
        file_path = Path(file_path).resolve()
        base_path = Path(base_path).resolve()

        return str(file_path.relative_to(base_path))
    except ValueError:
        # File is outside base path
        return None
    except (OSError, FileNotFoundError):
        return None


def safe_file_copy_with_metadata(src: str | Path, dst: str | Path) -> bool:
    """
    Safely copy file preserving metadata.

    Args:
        src: Source file path
        dst: Destination file path

    Returns:
        True if copy was successful
    """
    try:
        import shutil

        # Ensure destination directory exists
        dst_path = Path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file with metadata
        shutil.copy2(src, dst)
        return True

    except (OSError, PermissionError, shutil.Error) as e:
        logger.error(f"Failed to copy {src} to {dst}: {e}")
        return False


def calculate_directory_size(directory: str | Path) -> tuple[int, int]:
    """
    Calculate total size and file count of directory.

    Args:
        directory: Directory path

    Returns:
        Tuple of (total_size_bytes, file_count)
    """
    total_size = 0
    file_count = 0

    try:
        for file_path in Path(directory).rglob("*"):
            if file_path.is_file():
                try:
                    total_size += file_path.stat().st_size
                    file_count += 1
                except (OSError, FileNotFoundError):
                    continue
    except (OSError, PermissionError) as e:
        logger.warning(f"Error calculating directory size for {directory}: {e}")

    return total_size, file_count


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(size_bytes)

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"


def format_timestamp(timestamp: float, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format Unix timestamp as human-readable string.

    Args:
        timestamp: Unix timestamp
        format_str: Format string for datetime formatting

    Returns:
        Formatted timestamp string
    """
    try:
        return datetime.fromtimestamp(timestamp).strftime(format_str)
    except (ValueError, OSError):
        return "Invalid timestamp"


class FileSystemWatcher:
    """
    Simple file system watcher for detecting changes.

    This is a basic implementation for tracking file modifications
    during incremental indexing operations.
    """

    def __init__(self):
        self.watched_files: dict[str, float] = {}
        self.logger = logging.getLogger(f"{__name__}.FileSystemWatcher")

    def add_files(self, file_paths: list[str | Path]) -> None:
        """
        Add files to watch list.

        Args:
            file_paths: List of file paths to watch
        """
        for file_path in file_paths:
            str_path = str(file_path)
            mtime = get_file_mtime(file_path)
            if mtime is not None:
                self.watched_files[str_path] = mtime

    def check_changes(self) -> dict[str, str]:
        """
        Check for changes in watched files.

        Returns:
            Dictionary mapping file paths to change types ('modified', 'deleted')
        """
        changes = {}

        for file_path, stored_mtime in self.watched_files.items():
            current_mtime = get_file_mtime(file_path)

            if current_mtime is None:
                changes[file_path] = "deleted"
            elif abs(current_mtime - stored_mtime) > 0.001:
                changes[file_path] = "modified"
                # Update stored mtime
                self.watched_files[file_path] = current_mtime

        return changes

    def remove_file(self, file_path: str | Path) -> None:
        """
        Remove file from watch list.

        Args:
            file_path: File path to stop watching
        """
        str_path = str(file_path)
        self.watched_files.pop(str_path, None)

    def clear(self) -> None:
        """Clear all watched files."""
        self.watched_files.clear()
