import os
import pathspec
import logging
from pathlib import Path
from typing import List, Set, Dict, Any
from git import Repo, InvalidGitRepositoryError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ProjectAnalysisService:
    """Service for analyzing project structure and identifying relevant files."""
    
    def __init__(self):
        # Load configuration from environment variables
        self.max_directory_depth = int(os.getenv('MAX_DIRECTORY_DEPTH', '20'))
        self.follow_symlinks = os.getenv('FOLLOW_SYMLINKS', 'false').lower() == 'true'
        
        self.default_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', '.c', '.cpp', '.h', '.hpp',
            '.php', '.rb', '.swift', '.kt', '.scala', '.clj', '.cs', '.vb', '.f90', '.r', '.m',
            '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
            '.sql', '.json', '.yaml', '.yml', '.xml', '.toml', '.ini', '.cfg', '.conf',
            '.md', '.rst', '.txt', '.adoc', '.tex',
            '.html', '.css', '.scss', '.sass', '.less', '.vue', '.svelte',
            '.dockerfile', '.dockerignore', '.gitignore', '.gitattributes',
        }
        
        self.exclude_dirs = {
            'node_modules', '__pycache__', '.git', '.venv', 'venv', 'env', '.env',
            'dist', 'build', 'target', '.pytest_cache', '.mypy_cache', '.coverage',
            'htmlcov', '.tox', 'data', 'logs', 'tmp', 'temp', '.idea', '.vscode',
            '.vs', 'qdrant_storage', 'models', '.cache', 'bin', 'obj', 'out',
        }
        
        self.exclude_files = {
            '*.pyc', '*.pyo', '*.pyd', '.DS_Store', '*.so', '*.dylib', '*.dll',
            '*.class', '*.log', '*.lock', '*.swp', '*.swo', '*.bak', '*.tmp',
            '*.temp', '*.old', '*.orig', '*.rej', '*.pid', '*.sqlite', '*.db'
        }

    def get_relevant_files(self, directory: str) -> List[str]:
        """Get list of relevant files to index from a directory."""
        directory_path = Path(directory).resolve()
        
        if not directory_path.exists():
            print(f"Directory does not exist: {directory}")
            return []
        
        # Load gitignore patterns if available
        gitignore_spec = self._load_gitignore(directory_path)
        
        relevant_files = []
        
        for root, dirs, files in os.walk(directory_path, followlinks=self.follow_symlinks):
            root_path = Path(root)
            
            # Check directory depth
            try:
                relative_path = root_path.relative_to(directory_path)
                depth = len(relative_path.parts)
                if depth > self.max_directory_depth:
                    print(f"Skipping deep directory (depth {depth} > {self.max_directory_depth}): {root_path}")
                    dirs.clear()  # Don't recurse deeper
                    continue
            except ValueError:
                # Handle cases where relative_to fails
                continue
            
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            # Skip if directory is in gitignore
            if gitignore_spec:
                relative_root = root_path.relative_to(directory_path)
                if gitignore_spec.match_file(str(relative_root)):
                    dirs.clear()  # Don't recurse into this directory
                    continue
            
            for file in files:
                file_path = root_path / file
                
                # Check file extension
                if file_path.suffix.lower() not in self.default_extensions:
                    # Also check for files without extensions that might be important
                    if not self._is_important_file_without_extension(file):
                        continue
                
                # Check if file matches exclude patterns
                if self._should_exclude_file(file):
                    continue
                
                # Check gitignore
                if gitignore_spec:
                    relative_file = file_path.relative_to(directory_path)
                    if gitignore_spec.match_file(str(relative_file)):
                        continue
                
                # Check file size (skip very large files)
                try:
                    if file_path.stat().st_size > 1024 * 1024:  # 1MB limit
                        print(f"Skipping large file: {file_path}")
                        continue
                except OSError:
                    continue
                
                relevant_files.append(str(file_path))
        
        print(f"Found {len(relevant_files)} relevant files in {directory}")
        return relevant_files
    
    def _load_gitignore(self, directory: Path) -> pathspec.PathSpec:
        """Load gitignore patterns from the directory."""
        gitignore_path = directory / '.gitignore'
        
        if not gitignore_path.exists():
            # Try to find gitignore in parent directories (for git repositories)
            try:
                repo = Repo(directory, search_parent_directories=True)
                gitignore_path = Path(repo.working_dir) / '.gitignore'
            except (InvalidGitRepositoryError, Exception):
                return None
        
        if not gitignore_path.exists():
            return None
        
        try:
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                patterns = f.readlines()
            
            # Filter out comments and empty lines
            patterns = [line.strip() for line in patterns if line.strip() and not line.startswith('#')]
            
            return pathspec.PathSpec.from_lines('gitwildmatch', patterns)
        except Exception as e:
            print(f"Error reading .gitignore: {e}")
            return None
    
    def _is_important_file_without_extension(self, filename: str) -> bool:
        """Check if a file without extension is important (like Dockerfile, Makefile, etc.)."""
        important_files = {
            'dockerfile', 'makefile', 'rakefile', 'gemfile', 'procfile',
            'readme', 'license', 'changelog', 'authors', 'contributors',
            'copyright', 'install', 'news', 'todo', 'manifest'
        }
        return filename.lower() in important_files
    
    def _should_exclude_file(self, filename: str) -> bool:
        """Check if file should be excluded based on patterns."""
        import fnmatch
        
        for pattern in self.exclude_files:
            if fnmatch.fnmatch(filename, pattern):
                return True
        return False
    
    def detect_project_type(self, directory: str) -> str:
        """Detect the type of project based on files present."""
        directory_path = Path(directory)
        
        # Check for specific project files
        if (directory_path / 'package.json').exists():
            return 'javascript/nodejs'
        elif (directory_path / 'requirements.txt').exists() or (directory_path / 'pyproject.toml').exists():
            return 'python'
        elif (directory_path / 'pom.xml').exists() or (directory_path / 'build.gradle').exists():
            return 'java'
        elif (directory_path / 'go.mod').exists():
            return 'go'
        elif (directory_path / 'Cargo.toml').exists():
            return 'rust'
        elif (directory_path / 'composer.json').exists():
            return 'php'
        elif (directory_path / 'Gemfile').exists():
            return 'ruby'
        else:
            return 'unknown'
    
    def analyze_repository(self, directory: str = ".") -> Dict[str, Any]:
        """
        Analyze repository structure and provide detailed statistics for indexing planning.
        
        Args:
            directory: Path to the directory to analyze
            
        Returns:
            Dictionary with comprehensive analysis including file counts, size distribution,
            language breakdown, complexity assessment, and indexing recommendations.
        """
        try:
            directory_path = Path(directory).resolve()
            
            if not directory_path.exists():
                return {"error": f"Directory not found: {directory}"}
            
            # Initialize analysis result
            analysis = {
                "directory": str(directory_path),
                "project_type": self.detect_project_type(str(directory_path)),
                "total_files": 0,
                "relevant_files": 0,
                "excluded_files": 0,
                "exclusion_rate": 0.0,
                "size_analysis": {},
                "language_breakdown": {},
                "indexing_complexity": {},
                "recommendations": []
            }
            
            # Track file statistics
            total_files = 0
            relevant_files_list = []
            excluded_count = 0
            size_stats = {"total_size_mb": 0, "small_files": 0, "medium_files": 0, "large_files": 0, "binary_files": 0}
            language_stats = {}
            
            # Load gitignore patterns
            gitignore_spec = self._load_gitignore(directory_path)
            
            # Walk through directory structure
            for root, dirs, files in os.walk(directory_path, followlinks=self.follow_symlinks):
                root_path = Path(root)
                
                # Filter out excluded directories early
                dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
                
                for file in files:
                    file_path = root_path / file
                    total_files += 1
                    
                    try:
                        file_size = file_path.stat().st_size
                        size_stats["total_size_mb"] += file_size / (1024 * 1024)
                        
                        # Categorize by size
                        if file_size < 10 * 1024:  # < 10KB
                            size_stats["small_files"] += 1
                        elif file_size < 100 * 1024:  # 10KB - 100KB
                            size_stats["medium_files"] += 1
                        else:  # > 100KB
                            size_stats["large_files"] += 1
                        
                        # Check if file would be included
                        should_include = True
                        exclusion_reason = ""
                        
                        # Check file extension
                        if file_path.suffix.lower() not in self.default_extensions:
                            if not self._is_important_file_without_extension(file):
                                should_include = False
                                exclusion_reason = "extension"
                        
                        # Check exclude patterns
                        if should_include and self._should_exclude_file(file):
                            should_include = False
                            exclusion_reason = "pattern"
                        
                        # Check gitignore
                        if should_include and gitignore_spec:
                            try:
                                relative_file = file_path.relative_to(directory_path)
                                if gitignore_spec.match_file(str(relative_file)):
                                    should_include = False
                                    exclusion_reason = "gitignore"
                            except ValueError:
                                pass
                        
                        # Check file size limit
                        if should_include and file_size > 5 * 1024 * 1024:  # 5MB limit
                            should_include = False
                            exclusion_reason = "size"
                        
                        # Check if binary
                        if self._is_binary_file(file_path):
                            size_stats["binary_files"] += 1
                            if should_include:
                                should_include = False
                                exclusion_reason = "binary"
                        
                        if should_include:
                            relevant_files_list.append(str(file_path))
                            
                            # Track language statistics
                            extension = file_path.suffix.lower()
                            language = self._extension_to_language(extension)
                            language_stats[language] = language_stats.get(language, 0) + 1
                        else:
                            excluded_count += 1
                            
                    except (OSError, PermissionError):
                        excluded_count += 1
                        continue
            
            # Update analysis with collected statistics
            analysis["total_files"] = total_files
            analysis["relevant_files"] = len(relevant_files_list)
            analysis["excluded_files"] = excluded_count
            analysis["exclusion_rate"] = round((excluded_count / total_files) * 100, 1) if total_files > 0 else 0
            analysis["size_analysis"] = {
                "total_size_mb": round(size_stats["total_size_mb"], 2),
                "small_files": size_stats["small_files"],
                "medium_files": size_stats["medium_files"], 
                "large_files": size_stats["large_files"],
                "binary_files": size_stats["binary_files"]
            }
            analysis["language_breakdown"] = language_stats
            
            # Assess indexing complexity
            complexity_level = "simple"
            if len(relevant_files_list) > 10000:
                complexity_level = "high"
            elif len(relevant_files_list) > 1000:
                complexity_level = "medium"
            
            analysis["indexing_complexity"] = {
                "level": complexity_level,
                "estimated_indexing_time_minutes": self._estimate_indexing_time(len(relevant_files_list)),
                "recommended_batch_size": self._recommend_batch_size(len(relevant_files_list)),
                "memory_considerations": self._get_memory_recommendations(len(relevant_files_list))
            }
            
            # Generate recommendations
            recommendations = []
            if analysis["exclusion_rate"] < 50:
                recommendations.append("Consider adding more exclusion patterns to .ragignore for better performance")
            if size_stats["large_files"] > 100:
                recommendations.append("Many large files detected - consider reducing MAX_FILE_SIZE_MB")
            if len(language_stats) > 10:
                recommendations.append("Multi-language project detected - consider language-specific indexing strategies")
            
            analysis["recommendations"] = recommendations
            
            return analysis
            
        except Exception as e:
            return {"error": f"Repository analysis failed: {str(e)}", "directory": directory}
    
    def get_file_filtering_stats(self, directory: str = ".") -> Dict[str, Any]:
        """
        Get detailed statistics about file filtering for debugging and optimization.
        
        Args:
            directory: Path to the directory to analyze
            
        Returns:
            Dictionary with detailed breakdown of file filtering statistics
        """
        try:
            directory_path = Path(directory).resolve()
            
            if not directory_path.exists():
                return {"error": f"Directory not found: {directory}"}
            
            stats = {
                "directory": str(directory_path),
                "total_examined": 0,
                "included": 0,
                "excluded_by_extension": 0,
                "excluded_by_pattern": 0,
                "excluded_by_gitignore": 0,
                "excluded_by_size": 0,
                "excluded_by_binary_extension": 0,
                "excluded_by_binary_header": 0,
                "excluded_by_ragignore": 0,
                "excluded_directories": 0,
                "configuration": {
                    "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE_MB", "5")),
                    "max_directory_depth": self.max_directory_depth,
                    "follow_symlinks": self.follow_symlinks,
                    "default_extensions_count": len(self.default_extensions),
                    "exclude_dirs_count": len(self.exclude_dirs),
                    "exclude_patterns_count": len(self.exclude_files)
                }
            }
            
            # Load filtering specs
            gitignore_spec = self._load_gitignore(directory_path)
            max_file_size = int(os.getenv("MAX_FILE_SIZE_MB", "5")) * 1024 * 1024
            
            # Walk through directory
            for root, dirs, files in os.walk(directory_path, followlinks=self.follow_symlinks):
                root_path = Path(root)
                
                # Count excluded directories
                original_dirs = dirs[:]
                dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
                stats["excluded_directories"] += len(original_dirs) - len(dirs)
                
                for file in files:
                    file_path = root_path / file
                    stats["total_examined"] += 1
                    
                    excluded = False
                    
                    try:
                        file_size = file_path.stat().st_size
                        
                        # Check extension
                        if file_path.suffix.lower() not in self.default_extensions:
                            if not self._is_important_file_without_extension(file):
                                stats["excluded_by_extension"] += 1
                                excluded = True
                        
                        # Check exclude patterns
                        if not excluded and self._should_exclude_file(file):
                            stats["excluded_by_pattern"] += 1
                            excluded = True
                        
                        # Check gitignore
                        if not excluded and gitignore_spec:
                            try:
                                relative_file = file_path.relative_to(directory_path)
                                if gitignore_spec.match_file(str(relative_file)):
                                    stats["excluded_by_gitignore"] += 1
                                    excluded = True
                            except ValueError:
                                pass
                        
                        # Check file size
                        if not excluded and file_size > max_file_size:
                            stats["excluded_by_size"] += 1
                            excluded = True
                        
                        # Check binary by extension
                        if not excluded and self._is_binary_extension(file_path):
                            stats["excluded_by_binary_extension"] += 1
                            excluded = True
                        
                        # Check binary by header (sample check)
                        if not excluded and self._is_binary_file(file_path):
                            stats["excluded_by_binary_header"] += 1
                            excluded = True
                        
                        if not excluded:
                            stats["included"] += 1
                            
                    except (OSError, PermissionError):
                        stats["excluded_by_pattern"] += 1  # Treat as excluded due to access issues
            
            return stats
            
        except Exception as e:
            return {"error": f"File filtering analysis failed: {str(e)}", "directory": directory}
    
    def _extension_to_language(self, extension: str) -> str:
        """Map file extension to programming language."""
        mapping = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript', '.jsx': 'javascript',
            '.tsx': 'typescript', '.java': 'java', '.go': 'go', '.rs': 'rust',
            '.c': 'c', '.cpp': 'cpp', '.h': 'c', '.hpp': 'cpp', '.php': 'php',
            '.rb': 'ruby', '.swift': 'swift', '.kt': 'kotlin', '.scala': 'scala',
            '.clj': 'clojure', '.cs': 'csharp', '.vb': 'vb', '.f90': 'fortran',
            '.r': 'r', '.m': 'matlab', '.sh': 'shell', '.bash': 'shell',
            '.zsh': 'shell', '.fish': 'shell', '.ps1': 'powershell',
            '.bat': 'batch', '.cmd': 'batch', '.sql': 'sql', '.json': 'json',
            '.yaml': 'yaml', '.yml': 'yaml', '.xml': 'xml', '.toml': 'toml',
            '.ini': 'ini', '.cfg': 'config', '.conf': 'config', '.md': 'markdown',
            '.rst': 'rst', '.txt': 'text', '.html': 'html', '.css': 'css',
            '.scss': 'scss', '.sass': 'sass', '.less': 'less', '.vue': 'vue',
            '.svelte': 'svelte'
        }
        return mapping.get(extension, 'other')
    
    def _is_binary_extension(self, file_path: Path) -> bool:
        """Check if file has a binary extension."""
        binary_extensions = {
            '.exe', '.dll', '.so', '.dylib', '.bin', '.obj', '.o', '.a', '.lib',
            '.zip', '.tar', '.gz', '.bz2', '.xz', '.rar', '.7z',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico', '.svg', '.webp',
            '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv', '.wav', '.ogg',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.ttf', '.otf', '.woff', '.woff2', '.eot',
            '.class', '.jar', '.war', '.ear', '.pyc', '.pyo', '.pyd'
        }
        return file_path.suffix.lower() in binary_extensions
    
    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary by reading first few bytes."""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                if not chunk:
                    return False
                # Check for null bytes which are common in binary files
                return b'\x00' in chunk
        except (OSError, PermissionError):
            return True  # Assume binary if can't read
    
    def _estimate_indexing_time(self, file_count: int) -> float:
        """Estimate indexing time in minutes based on file count."""
        # Rough estimate: ~0.1 seconds per file for processing + embedding
        estimated_seconds = file_count * 0.1
        return round(estimated_seconds / 60, 1)
    
    def _recommend_batch_size(self, file_count: int) -> int:
        """Recommend optimal batch size based on file count."""
        if file_count < 100:
            return 10
        elif file_count < 1000:
            return 20
        elif file_count < 10000:
            return 50
        else:
            return 100
    
    def analyze_directory_structure(self, directory: str) -> Dict[str, Any]:
        """Analyze directory structure - wrapper for analyze_repository for backward compatibility."""
        return self.analyze_repository(directory)
    
    def get_project_context(self, directory: str) -> Dict[str, Any]:
        """Get project context information including project name and type."""
        try:
            directory_path = Path(directory).resolve()
            # Use same naming convention as MCP tools (replace spaces and hyphens with underscores)
            project_name = directory_path.name.replace(" ", "_").replace("-", "_")
            project_type = self.detect_project_type(directory)
            
            return {
                "project_name": project_name,
                "project_type": project_type,
                "directory": str(directory_path)
            }
        except Exception as e:
            return {
                "project_name": "unknown",
                "project_type": "unknown",
                "directory": directory,
                "error": str(e)
            }
    
    def _get_memory_recommendations(self, file_count: int) -> List[str]:
        """Get memory usage recommendations based on file count."""
        recommendations = []
        
        if file_count > 10000:
            recommendations.append("Large repository detected - consider using streaming mode")
            recommendations.append("Increase MEMORY_WARNING_THRESHOLD_MB to 2000+")
        elif file_count > 1000:
            recommendations.append("Medium repository - default settings should work well")
        else:
            recommendations.append("Small repository - can use larger batch sizes for faster processing")
        
        return recommendations