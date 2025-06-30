import os
from pathspec import PathSpec
from pathspec.patterns import GitWildMatchPattern

class ProjectAnalysisService:
    def __init__(self):
        self.project_type_indicators = {
            "npm": ["package.json"],
            "python": ["requirements.txt", "pyproject.toml"],
            "java": ["pom.xml", "build.gradle"],
            "go": ["go.mod"],
            # Add more as needed
        }
        self.ignored_extensions = [
            '.pyc', '.o', '.so', '.dll', '.exe', '.bin', '.log', '.tmp', '.temp',
            '.zip', '.tar', '.gz', '.rar', '.7z', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.ico',
            '.mp3', '.mp4', '.avi', '.mov', '.flv', '.wmv', '.pdf', '.doc', '.docx', '.xls', '.xlsx',
            '.ppt', '.pptx', '.sqlite3', '.db', '.DS_Store', '.vscode', '.idea',
        ]
        self.ignored_directories = [
            '.git', '.venv', 'venv', 'node_modules', '__pycache__', 'build', 'dist',
        ]

    def _get_gitignore_patterns(self, directory_path: str) -> list[str]:
        gitignore_path = os.path.join(directory_path, '.gitignore')
        if os.path.exists(gitignore_path):
            with open(gitignore_path, 'r') as f:
                return f.readlines()
        return []

    def scan_directory(self, directory_path: str) -> list[str]:
        """Scans a directory and returns a list of all file paths within it."""
        filepaths = []
        for root, _, files in os.walk(directory_path):
            # Check if the current directory is in the ignored_directories list
            if any(ignored_dir in root for ignored_dir in self.ignored_directories):
                continue

            for file in files:
                # Check if the file extension is in the ignored_extensions list
                if any(file.endswith(ext) for ext in self.ignored_extensions):
                    continue
                filepaths.append(os.path.join(root, file))
        return filepaths

    def identify_project_type(self, directory_path: str) -> list[str]:
        """Identifies the project type based on indicator files."""
        found_types = []
        for root, _, files in os.walk(directory_path):
            for project_type, indicators in self.project_type_indicators.items():
                if any(indicator in files for indicator in indicators):
                    found_types.append(project_type)
            # Only check the top-level directory for project type indicators
            break
        return found_types

    def get_relevant_files(self, directory_path: str) -> list[str]:
        """Returns a list of relevant source code files to be indexed, ignoring .gitignore patterns and common non-source files."""
        all_files = self.scan_directory(directory_path)
        gitignore_patterns = self._get_gitignore_patterns(directory_path)

        spec = PathSpec.from_lines(GitWildMatchPattern, gitignore_patterns)

        relevant_files = []
        for filepath in all_files:
            relative_filepath = os.path.relpath(filepath, directory_path)
            if not spec.match_file(relative_filepath):
                relevant_files.append(filepath)
        return relevant_files
