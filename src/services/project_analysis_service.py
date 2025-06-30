import os
import pathspec
from pathlib import Path
from typing import List, Set
from git import Repo, InvalidGitRepositoryError

class ProjectAnalysisService:
    """Service for analyzing project structure and identifying relevant files."""
    
    def __init__(self):
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
        
        for root, dirs, files in os.walk(directory_path):
            root_path = Path(root)
            
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