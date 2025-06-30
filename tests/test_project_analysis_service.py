import os
import tempfile
import shutil
from src.services.project_analysis_service import ProjectAnalysisService

class TestProjectAnalysisService:

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.service = ProjectAnalysisService()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir)

    def _create_file(self, path, content=""):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)

    def test_scan_directory(self):
        self._create_file(os.path.join(self.temp_dir, "file1.txt"))
        self._create_file(os.path.join(self.temp_dir, "subdir/file2.py"))
        
        files = self.service.scan_directory(self.temp_dir)
        assert len(files) == 2
        assert os.path.join(self.temp_dir, "file1.txt") in files
        assert os.path.join(self.temp_dir, "subdir/file2.py") in files

    def test_identify_project_type(self):
        self._create_file(os.path.join(self.temp_dir, "package.json"))
        project_types = self.service.identify_project_type(self.temp_dir)
        assert "npm" in project_types

        self._create_file(os.path.join(self.temp_dir, "requirements.txt"))
        project_types = self.service.identify_project_type(self.temp_dir)
        assert "python" in project_types

    def test_get_relevant_files_with_gitignore(self):
        self._create_file(os.path.join(self.temp_dir, "main.py"))
        self._create_file(os.path.join(self.temp_dir, "temp.txt"))
        self._create_file(os.path.join(self.temp_dir, "build/output.log"))
        self._create_file(os.path.join(self.temp_dir, ".gitignore"), content="""
/temp.txt
build/
""")

        relevant_files = self.service.get_relevant_files(self.temp_dir)
        assert len(relevant_files) == 2 # main.py and .gitignore itself
        assert os.path.join(self.temp_dir, "main.py") in relevant_files
        assert os.path.join(self.temp_dir, ".gitignore") in relevant_files
        assert os.path.join(self.temp_dir, "temp.txt") not in relevant_files
        assert os.path.join(self.temp_dir, "build/output.log") not in relevant_files
