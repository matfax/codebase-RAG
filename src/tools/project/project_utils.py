"""Project utility functions for project detection and management.

This module provides utility functions for detecting project boundaries, 
managing project configuration, and handling collection naming.
"""

import os
import logging
from typing import Dict, Optional, Set, Tuple, Any
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Global variables for caching
_current_project = None

# Configuration
PROJECT_MARKERS = ['.git', 'pyproject.toml', 'package.json', 'Cargo.toml', 'go.mod', 'pom.xml', 'build.gradle']


def get_current_project(client_directory: Optional[str] = None) -> Optional[Dict[str, str]]:
    """
    Detect the current project based on common project markers.
    
    This function traverses up the directory tree looking for project markers
    like .git, pyproject.toml, package.json, etc.
    
    Args:
        client_directory: Optional directory to start search from (default: current directory)
        
    Returns:
        Dictionary with project information or None if no project found
    """
    global _current_project
    
    if client_directory:
        cwd = Path(client_directory).resolve()
    else:
        cwd = Path.cwd()

    # Search for project markers in current directory and parent directories
    for parent in [cwd] + list(cwd.parents):
        for marker in PROJECT_MARKERS:
            if (parent / marker).exists():
                project_name = parent.name.replace(" ", "_").replace("-", "_")
                _current_project = {
                    "name": project_name,
                    "root": str(parent),
                    "collection_prefix": f"project_{project_name}"
                }
                logger.debug(f"Detected project: {project_name} at {parent}")
                return _current_project
    
    # Fallback to directory name if no project markers found
    project_name = cwd.name.replace(" ", "_").replace("-", "_")
    _current_project = {
        "name": project_name,
        "root": str(cwd),
        "collection_prefix": f"dir_{project_name}"
    }
    logger.debug(f"No project markers found, using directory name: {project_name}")
    return _current_project


def get_collection_name(file_path: str, file_type: str = "code") -> str:
    """
    Generate a collection name for a file based on project context and file type.
    
    Args:
        file_path: Path to the file
        file_type: Type of content (code, config, documentation)
        
    Returns:
        Collection name string
    """
    path = Path(file_path).resolve()
    
    # Try to use current project context
    current_project = get_current_project()
    if current_project:
        project_root = Path(current_project["root"])
        try:
            # Check if file is within project root
            path.relative_to(project_root)
            return f"{current_project['collection_prefix']}_{file_type}"
        except ValueError:
            # File is outside project root, continue with fallback logic
            pass
    
    # Fallback: search for project markers in file's directory tree
    for parent in path.parents:
        for marker in PROJECT_MARKERS:
            if (parent / marker).exists():
                project_name = parent.name.replace(" ", "_").replace("-", "_")
                return f"project_{project_name}_{file_type}"
    
    # Final fallback: use global collection
    return f"global_{file_type}"


def load_ragignore_patterns(directory: Path) -> Tuple[Set[str], Set[str]]:
    """
    Load .ragignore patterns for excluding files and directories from indexing.
    
    Args:
        directory: Directory to start search for .ragignore file
        
    Returns:
        Tuple of (excluded_directories, excluded_patterns)
    """
    exclude_dirs = set()
    exclude_patterns = set()
    
    # Default exclusions
    default_exclude_dirs = {
        'node_modules', '__pycache__', '.git', '.venv', 'venv', 'env', '.env',
        'dist', 'build', 'target', '.pytest_cache', '.mypy_cache', '.coverage', 
        'htmlcov', '.tox', 'data', 'logs', 'tmp', 'temp', '.idea', '.vscode', 
        '.vs', 'qdrant_storage', 'models', '.cache'
    }
    
    default_exclude_patterns = {
        '*.pyc', '*.pyo', '*.pyd', '.DS_Store', '*.so', '*.dylib', '*.dll', 
        '*.class', '*.log', '*.lock', '*.swp', '*.swo', '*.bak', '*.tmp', 
        '*.temp', '*.old', '*.orig', '*.rej', '.env*', '*.sqlite', '*.db', '*.pid'
    }
    
    # Search for .ragignore file in directory tree
    ragignore_path = None
    for parent in [directory] + list(directory.parents):
        potential_path = parent / '.ragignore'
        if potential_path.exists():
            ragignore_path = potential_path
            break
    
    if not ragignore_path:
        logger.debug("No .ragignore file found, using default exclusions")
        return default_exclude_dirs, default_exclude_patterns
    
    try:
        logger.debug(f"Loading .ragignore patterns from {ragignore_path}")
        with open(ragignore_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                if line.endswith('/'):
                    exclude_dirs.add(line.rstrip('/'))
                else:
                    exclude_patterns.add(line)
                    
        logger.debug(f"Loaded {len(exclude_dirs)} directory patterns and {len(exclude_patterns)} file patterns")
        
    except Exception as e:
        logger.warning(f"Error reading .ragignore: {e}, using defaults")
        return default_exclude_dirs, default_exclude_patterns
    
    # If no patterns loaded, use defaults
    if not exclude_dirs and not exclude_patterns:
        return default_exclude_dirs, default_exclude_patterns
    
    return exclude_dirs, exclude_patterns


def clear_project_collections() -> Dict[str, Any]:
    """
    Clear all collections for the current project.
    
    Returns:
        Dictionary with results of clearing operations
    """
    from ...tools.database.qdrant_utils import get_qdrant_client
    
    current_project = get_current_project()
    if not current_project:
        return {"error": "No project context found", "cleared": []}
    
    client = get_qdrant_client()
    cleared = []
    errors = []
    
    try:
        existing_collections = [c.name for c in client.get_collections().collections]
        
        for collection_type in ['code', 'config', 'documentation', 'file_metadata']:
            collection_name = f"{current_project['collection_prefix']}_{collection_type}"
            if collection_name in existing_collections:
                try:
                    client.delete_collection(collection_name)
                    cleared.append(collection_name)
                    logger.info(f"Cleared collection: {collection_name}")
                except Exception as e:
                    error_msg = f"Failed to clear {collection_name}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
        
        return {
            "project": current_project['name'],
            "cleared_collections": cleared,
            "errors": errors if errors else None
        }
        
    except Exception as e:
        error_msg = f"Failed to access collections: {str(e)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "project": current_project.get('name', 'unknown')
        }


def delete_file_chunks(file_path: str, collection_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Delete all chunks for a specific file from the vector database.
    
    Args:
        file_path: Path to the file whose chunks should be deleted
        collection_name: Optional specific collection name to delete from
        
    Returns:
        Dictionary with deletion results
    """
    from ...tools.database.qdrant_utils import get_qdrant_client
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue
    
    logger.info(f"Deleting chunks for file: {file_path}")
    
    try:
        if not file_path or not isinstance(file_path, str):
            return {"error": "Invalid file path"}
            
        abs_path = Path(file_path).resolve()
        qdrant_client = get_qdrant_client()
        
        # Determine collection name if not provided
        if collection_name is None:
            suffix = abs_path.suffix.lower()
            if suffix in ['.json', '.yaml', '.yml', '.xml', '.toml', '.ini', '.env']:
                file_type = "config"
            elif suffix in ['.md', '.markdown', '.rst', '.txt', '.mdx']:
                file_type = "documentation"
            else:
                file_type = "code"
            collection_name = get_collection_name(str(abs_path), file_type)
        
        # Check if collection exists
        try:
            collections = [c.name for c in qdrant_client.get_collections().collections]
            if collection_name not in collections:
                return {"error": f"Collection '{collection_name}' does not exist"}
        except Exception:
            return {"error": f"Could not access collection '{collection_name}'"}
        
        # Count existing points for the file
        filter_condition = Filter(
            must=[FieldCondition(key="file_path", match=MatchValue(value=str(abs_path)))]
        )
        
        count_response = qdrant_client.count(
            collection_name=collection_name,
            count_filter=filter_condition,
            exact=True
        )
        points_before = count_response.count
        
        # Delete the points
        delete_response = qdrant_client.delete(
            collection_name=collection_name,
            points_selector=filter_condition
        )
        
        logger.info(f"Deleted {points_before} points for {file_path} from {collection_name}")
        
        return {
            "file_path": str(abs_path),
            "collection": collection_name,
            "deleted_points": points_before,
        }
        
    except Exception as e:
        error_msg = f"Failed to delete chunks for {file_path}: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg, "file_path": file_path}