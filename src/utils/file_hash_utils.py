"""
File hashing utilities for incremental indexing.

This module provides additional hashing utilities beyond the basic
functionality in FileMetadata, including batch hashing and verification.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


logger = logging.getLogger(__name__)


def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> Optional[str]:
    """
    Calculate hash of a file using specified algorithm.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use ('sha256', 'md5', 'sha1')
        
    Returns:
        Hexadecimal hash string, or None if file cannot be read
    """
    try:
        # Get hash object
        if algorithm.lower() == 'sha256':
            hash_obj = hashlib.sha256()
        elif algorithm.lower() == 'md5':
            hash_obj = hashlib.md5()
        elif algorithm.lower() == 'sha1':
            hash_obj = hashlib.sha1()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        # Read file in chunks for memory efficiency
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
        
    except Exception as e:
        logger.warning(f"Failed to hash file {file_path}: {e}")
        return None


def batch_calculate_hashes(
    file_paths: List[str], 
    algorithm: str = 'sha256',
    max_workers: Optional[int] = None
) -> Dict[str, Optional[str]]:
    """
    Calculate hashes for multiple files in parallel.
    
    Args:
        file_paths: List of file paths to hash
        algorithm: Hash algorithm to use
        max_workers: Maximum number of worker threads
        
    Returns:
        Dictionary mapping file paths to their hashes
    """
    results = {}
    
    if not file_paths:
        return results
    
    # Use reasonable number of workers
    if max_workers is None:
        max_workers = min(len(file_paths), 4)  # Don't overwhelm the system
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_path = {
            executor.submit(calculate_file_hash, path, algorithm): path 
            for path in file_paths
        }
        
        # Collect results
        for future in as_completed(future_to_path):
            file_path = future_to_path[future]
            try:
                hash_value = future.result()
                results[file_path] = hash_value
            except Exception as e:
                logger.error(f"Error hashing file {file_path}: {e}")
                results[file_path] = None
    
    duration = time.time() - start_time
    success_count = sum(1 for h in results.values() if h is not None)
    
    logger.info(
        f"Batch hashed {success_count}/{len(file_paths)} files in {duration:.2f}s "
        f"({len(file_paths)/duration:.1f} files/sec)"
    )
    
    return results


def verify_file_hash(file_path: str, expected_hash: str, algorithm: str = 'sha256') -> bool:
    """
    Verify that a file's hash matches the expected value.
    
    Args:
        file_path: Path to the file
        expected_hash: Expected hash value
        algorithm: Hash algorithm to use
        
    Returns:
        True if hash matches, False otherwise
    """
    actual_hash = calculate_file_hash(file_path, algorithm)
    
    if actual_hash is None:
        return False
    
    return actual_hash.lower() == expected_hash.lower()


def compare_file_hashes(file_path1: str, file_path2: str, algorithm: str = 'sha256') -> bool:
    """
    Compare hashes of two files to check if they have identical content.
    
    Args:
        file_path1: Path to first file
        file_path2: Path to second file
        algorithm: Hash algorithm to use
        
    Returns:
        True if files have identical content, False otherwise
    """
    hash1 = calculate_file_hash(file_path1, algorithm)
    hash2 = calculate_file_hash(file_path2, algorithm)
    
    if hash1 is None or hash2 is None:
        return False
    
    return hash1 == hash2


def find_duplicate_files(file_paths: List[str], algorithm: str = 'sha256') -> Dict[str, List[str]]:
    """
    Find duplicate files based on content hash.
    
    Args:
        file_paths: List of file paths to check
        algorithm: Hash algorithm to use
        
    Returns:
        Dictionary mapping hashes to lists of files with that hash
    """
    # Calculate all hashes
    file_hashes = batch_calculate_hashes(file_paths, algorithm)
    
    # Group files by hash
    hash_to_files = {}
    for file_path, file_hash in file_hashes.items():
        if file_hash is not None:
            if file_hash not in hash_to_files:
                hash_to_files[file_hash] = []
            hash_to_files[file_hash].append(file_path)
    
    # Return only groups with multiple files (duplicates)
    duplicates = {
        file_hash: files 
        for file_hash, files in hash_to_files.items() 
        if len(files) > 1
    }
    
    return duplicates


def get_file_hash_info(file_path: str) -> Dict[str, any]:
    """
    Get comprehensive hash information for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with hash information and metadata
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            return {
                'file_path': file_path,
                'exists': False,
                'error': 'File not found'
            }
        
        stat = path.stat()
        
        # Calculate multiple hashes
        hashes = {}
        for algorithm in ['sha256', 'md5', 'sha1']:
            hashes[algorithm] = calculate_file_hash(file_path, algorithm)
        
        return {
            'file_path': file_path,
            'exists': True,
            'size': stat.st_size,
            'mtime': stat.st_mtime,
            'hashes': hashes,
            'primary_hash': hashes['sha256']  # Use SHA256 as primary
        }
        
    except Exception as e:
        return {
            'file_path': file_path,
            'exists': False,
            'error': str(e)
        }


class HashVerifier:
    """
    Utility class for batch hash verification operations.
    """
    
    def __init__(self, algorithm: str = 'sha256'):
        """
        Initialize hash verifier.
        
        Args:
            algorithm: Hash algorithm to use
        """
        self.algorithm = algorithm
        self.logger = logging.getLogger(f"{__name__}.HashVerifier")
    
    def verify_file_list(self, file_hash_pairs: List[Tuple[str, str]]) -> Dict[str, bool]:
        """
        Verify multiple files against their expected hashes.
        
        Args:
            file_hash_pairs: List of (file_path, expected_hash) tuples
            
        Returns:
            Dictionary mapping file paths to verification results
        """
        results = {}
        
        for file_path, expected_hash in file_hash_pairs:
            try:
                results[file_path] = verify_file_hash(file_path, expected_hash, self.algorithm)
            except Exception as e:
                self.logger.error(f"Error verifying {file_path}: {e}")
                results[file_path] = False
        
        return results
    
    def get_verification_summary(self, results: Dict[str, bool]) -> Dict[str, int]:
        """
        Get summary statistics for verification results.
        
        Args:
            results: Dictionary of verification results
            
        Returns:
            Summary statistics
        """
        passed = sum(1 for result in results.values() if result)
        failed = len(results) - passed
        
        return {
            'total': len(results),
            'passed': passed,
            'failed': failed,
            'success_rate': (passed / len(results) * 100) if results else 0
        }