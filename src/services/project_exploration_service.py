"""
Project Exploration Service

This module provides comprehensive project exploration capabilities for the
explore_project MCP prompt, offering detailed analysis and intelligent insights.
"""

import os
import logging
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from dataclasses import dataclass, field

try:
    from services.project_analysis_service import ProjectAnalysisService
    from services.indexing_service import IndexingService  
    from services.embedding_service import EmbeddingService
    from services.rag_search_strategy import RAGSearchStrategy
except ImportError:
    # For testing without full service dependencies
    ProjectAnalysisService = None
    IndexingService = None
    EmbeddingService = None
    RAGSearchStrategy = None


logger = logging.getLogger(__name__)


@dataclass
class ProjectExplorationResult:
    """
    Comprehensive project exploration result.
    
    Contains all the insights and analysis results from project exploration.
    """
    
    # Basic project information
    project_name: str
    project_root: str
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    # Architecture insights
    architecture_pattern: str = "unknown"
    project_type: str = "unknown"
    framework_stack: List[str] = field(default_factory=list)
    
    # Structure analysis
    directory_structure: Dict[str, Any] = field(default_factory=dict)
    key_directories: List[str] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)
    
    # Component analysis
    core_modules: List[str] = field(default_factory=list)
    utility_modules: List[str] = field(default_factory=list)
    test_modules: List[str] = field(default_factory=list)
    config_files: List[str] = field(default_factory=list)
    
    # Dependencies and relationships
    external_dependencies: List[str] = field(default_factory=list)
    internal_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    circular_dependencies: List[Tuple[str, str]] = field(default_factory=list)
    
    # Complexity and quality metrics
    complexity_score: float = 0.0
    maintainability_score: float = 0.0
    test_coverage_estimate: float = 0.0
    documentation_level: str = "unknown"
    
    # Development insights
    coding_patterns: List[str] = field(default_factory=list)
    common_conventions: List[str] = field(default_factory=list)
    potential_issues: List[str] = field(default_factory=list)
    
    # Learning recommendations
    exploration_priorities: List[str] = field(default_factory=list)
    recommended_reading_order: List[str] = field(default_factory=list)
    key_concepts_to_understand: List[str] = field(default_factory=list)
    
    # RAG-based insights (NEW)
    rag_architecture_analysis: Dict[str, Any] = field(default_factory=dict)
    rag_entry_points: List[Dict[str, Any]] = field(default_factory=list)
    rag_component_relationships: Dict[str, Any] = field(default_factory=dict)
    rag_insights_enabled: bool = False
    
    # Metadata
    analysis_duration_seconds: float = 0.0
    files_analyzed: int = 0
    confidence_score: float = 0.0


class ProjectExplorationService:
    """
    Advanced project exploration service for deep codebase analysis.
    
    Provides comprehensive project analysis capabilities including architecture
    detection, component identification, dependency mapping, and learning guidance.
    """
    
    def __init__(self, enable_rag: bool = True):
        self.logger = logger
        self.analysis_service = ProjectAnalysisService() if ProjectAnalysisService else None
        self.indexing_service = IndexingService() if IndexingService else None
        self.embedding_service = EmbeddingService() if EmbeddingService else None
        
        # RAG Search Strategy (NEW)
        self.rag_search_strategy = RAGSearchStrategy() if RAGSearchStrategy and enable_rag else None
        self.enable_rag = enable_rag and RAGSearchStrategy is not None
        
        # Pattern databases for recognition
        self.architecture_patterns = self._load_architecture_patterns()
        self.framework_signatures = self._load_framework_signatures()
        self.project_type_indicators = self._load_project_type_indicators()
    
    def explore_project(
        self,
        project_path: str,
        focus_area: Optional[str] = None,
        detail_level: str = "overview",
        include_dependencies: bool = True,
        analyze_complexity: bool = True
    ) -> ProjectExplorationResult:
        """
        Perform comprehensive project exploration.
        
        Args:
            project_path: Path to the project root
            focus_area: Specific area to focus analysis on
            detail_level: Level of detail ("overview", "detailed", "comprehensive")
            include_dependencies: Whether to analyze dependencies
            analyze_complexity: Whether to perform complexity analysis
            
        Returns:
            ProjectExplorationResult with comprehensive insights
        """
        start_time = time.time()
        
        project_path = Path(project_path).resolve()
        project_name = project_path.name
        
        self.logger.info(f"Starting project exploration for: {project_name}")
        
        # Initialize result
        result = ProjectExplorationResult(
            project_name=project_name,
            project_root=str(project_path)
        )
        
        try:
            # Phase 1: Basic project analysis
            basic_analysis = self._perform_basic_analysis(project_path)
            self._update_result_with_basic_analysis(result, basic_analysis)
            
            # Phase 2: Architecture detection
            architecture_info = self._detect_architecture_pattern(project_path, basic_analysis)
            self._update_result_with_architecture(result, architecture_info)
            
            # Phase 3: Component identification
            components = self._identify_key_components(project_path, focus_area)
            self._update_result_with_components(result, components)
            
            # Phase 4: Dependency analysis (if requested)
            if include_dependencies:
                dependencies = self._analyze_dependencies(project_path)
                self._update_result_with_dependencies(result, dependencies)
            
            # Phase 5: Complexity analysis (if requested)
            if analyze_complexity:
                complexity = self._analyze_complexity(project_path, basic_analysis)
                self._update_result_with_complexity(result, complexity)
            
            # Phase 6: RAG-based analysis (NEW)
            if self.enable_rag:
                rag_results = self._perform_rag_analysis(project_path, focus_area, detail_level)
                self._integrate_rag_results(result, rag_results)
            
            # Phase 7: Generate learning recommendations (enhanced with RAG)
            learning_recs = self._generate_learning_recommendations(result, detail_level, focus_area)
            self._update_result_with_learning_recommendations(result, learning_recs)
            
            # Finalize result
            result.analysis_duration_seconds = time.time() - start_time
            result.confidence_score = self._calculate_confidence_score(result)
            
            self.logger.info(f"Project exploration completed in {result.analysis_duration_seconds:.2f}s")
            if self.enable_rag:
                self.logger.info(f"RAG-enhanced analysis included architecture detection, entry points, and component relationships")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during project exploration: {e}")
            result.potential_issues.append(f"Analysis error: {str(e)}")
            result.analysis_duration_seconds = time.time() - start_time
            return result
    
    def _perform_basic_analysis(self, project_path: Path) -> Dict[str, Any]:
        """Perform basic project analysis using existing services."""
        if self.analysis_service:
            try:
                return self.analysis_service.analyze_repository(str(project_path))
            except Exception as e:
                self.logger.warning(f"Failed to use analysis service: {e}")
        
        # Fallback to basic analysis
        return self._basic_project_scan(project_path)
    
    def _basic_project_scan(self, project_path: Path) -> Dict[str, Any]:
        """Perform basic project scanning when services are not available."""
        analysis = {
            "total_files": 0,
            "relevant_files": 0,
            "language_breakdown": {},
            "size_analysis": {"total_size_mb": 0},
            "directory_structure": {},
            "key_files": []
        }
        
        try:
            # Scan directories and files
            for root, dirs, files in os.walk(project_path):
                # Skip hidden and common ignore directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {
                    'node_modules', '__pycache__', 'venv', '.venv', 'dist', 'build'
                }]
                
                root_path = Path(root)
                relative_root = root_path.relative_to(project_path)
                
                for file in files:
                    if file.startswith('.'):
                        continue
                        
                    analysis["total_files"] += 1
                    file_path = root_path / file
                    
                    # Get file extension
                    ext = file_path.suffix.lower()
                    if ext in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', '.cpp', '.c']:
                        analysis["relevant_files"] += 1
                        
                        # Update language breakdown
                        lang = self._extension_to_language(ext)
                        analysis["language_breakdown"][lang] = analysis["language_breakdown"].get(lang, 0) + 1
                    
                    # Track key files
                    if file.lower() in ['main.py', 'app.py', 'index.js', 'main.js', 'package.json', 'requirements.txt', 'pyproject.toml']:
                        analysis["key_files"].append(str(file_path.relative_to(project_path)))
                    
                    # Update size
                    try:
                        size = file_path.stat().st_size
                        analysis["size_analysis"]["total_size_mb"] += size / (1024 * 1024)
                    except:
                        pass
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in basic project scan: {e}")
            return analysis
    
    def _extension_to_language(self, ext: str) -> str:
        """Map file extension to programming language."""
        mapping = {
            '.py': 'Python',
            '.js': 'JavaScript', 
            '.ts': 'TypeScript',
            '.jsx': 'React',
            '.tsx': 'React TypeScript',
            '.java': 'Java',
            '.go': 'Go',
            '.rs': 'Rust',
            '.cpp': 'C++',
            '.c': 'C'
        }
        return mapping.get(ext, 'Other')
    
    def _detect_architecture_pattern(self, project_path: Path, basic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Detect the project's architecture pattern."""
        architecture_info = {
            "pattern": "unknown",
            "confidence": 0.0,
            "framework_stack": [],
            "project_type": "unknown"
        }
        
        # Check for common architecture patterns based on directory structure and files
        patterns_found = []
        
        # Check for MVC pattern
        if self._has_mvc_structure(project_path):
            patterns_found.append(("MVC", 0.8))
        
        # Check for microservices
        if self._has_microservices_structure(project_path):
            patterns_found.append(("Microservices", 0.7))
        
        # Check for layered architecture
        if self._has_layered_structure(project_path):
            patterns_found.append(("Layered", 0.6))
        
        # Check for component-based architecture
        if self._has_component_structure(project_path):
            patterns_found.append(("Component-based", 0.7))
        
        # Select the most likely pattern
        if patterns_found:
            best_pattern = max(patterns_found, key=lambda x: x[1])
            architecture_info["pattern"] = best_pattern[0]
            architecture_info["confidence"] = best_pattern[1]
        
        # Detect frameworks
        frameworks = self._detect_frameworks(project_path, basic_analysis)
        architecture_info["framework_stack"] = frameworks
        
        # Determine project type
        project_type = self._determine_project_type(project_path, basic_analysis, frameworks)
        architecture_info["project_type"] = project_type
        
        return architecture_info
    
    def _has_mvc_structure(self, project_path: Path) -> bool:
        """Check if project has MVC structure."""
        mvc_indicators = ['models', 'views', 'controllers', 'model', 'view', 'controller']
        subdirs = [d.name.lower() for d in project_path.iterdir() if d.is_dir()]
        return any(indicator in subdirs for indicator in mvc_indicators)
    
    def _has_microservices_structure(self, project_path: Path) -> bool:
        """Check if project has microservices structure."""
        # Look for multiple service directories or docker-compose
        service_indicators = ['services', 'service', 'docker-compose.yml', 'docker-compose.yaml']
        has_docker_compose = any((project_path / indicator).exists() for indicator in service_indicators[-2:])
        
        subdirs = [d.name.lower() for d in project_path.iterdir() if d.is_dir()]
        has_services_dir = any('service' in subdir for subdir in subdirs)
        
        return has_docker_compose or has_services_dir
    
    def _has_layered_structure(self, project_path: Path) -> bool:
        """Check if project has layered architecture."""
        layer_indicators = ['data', 'business', 'presentation', 'api', 'core', 'infrastructure']
        subdirs = [d.name.lower() for d in project_path.iterdir() if d.is_dir()]
        return sum(1 for indicator in layer_indicators if indicator in subdirs) >= 2
    
    def _has_component_structure(self, project_path: Path) -> bool:
        """Check if project has component-based structure."""
        component_indicators = ['components', 'component', 'modules', 'packages']
        subdirs = [d.name.lower() for d in project_path.iterdir() if d.is_dir()]
        return any(indicator in subdirs for indicator in component_indicators)
    
    def _detect_frameworks(self, project_path: Path, basic_analysis: Dict[str, Any]) -> List[str]:
        """Detect frameworks used in the project."""
        frameworks = []
        
        # Check package files for framework dependencies
        package_files = ['package.json', 'requirements.txt', 'pyproject.toml', 'Cargo.toml', 'pom.xml']
        
        for package_file in package_files:
            package_path = project_path / package_file
            if package_path.exists():
                frameworks.extend(self._parse_package_file_for_frameworks(package_path))
        
        # Check for framework-specific files and directories
        framework_files = {
            'React': ['src/App.jsx', 'src/App.tsx', 'public/index.html'],
            'Vue': ['vue.config.js', 'src/main.js'],
            'Angular': ['angular.json', 'src/main.ts'],
            'Django': ['manage.py', 'settings.py'],
            'Flask': ['app.py', 'wsgi.py'],
            'FastAPI': ['main.py'],
            'Express': ['app.js', 'server.js'],
            'Spring': ['application.properties', 'pom.xml']
        }
        
        for framework, files in framework_files.items():
            if any((project_path / file).exists() for file in files):
                if framework not in frameworks:
                    frameworks.append(framework)
        
        return frameworks
    
    def _parse_package_file_for_frameworks(self, package_path: Path) -> List[str]:
        """Parse package file to identify frameworks."""
        frameworks = []
        
        try:
            content = package_path.read_text()
            
            # Framework keywords to look for
            framework_keywords = {
                'react': 'React',
                'vue': 'Vue',
                'angular': 'Angular',
                'django': 'Django',
                'flask': 'Flask',
                'fastapi': 'FastAPI',
                'express': 'Express',
                'spring': 'Spring',
                'bootstrap': 'Bootstrap',
                'tailwind': 'Tailwind CSS'
            }
            
            content_lower = content.lower()
            for keyword, framework in framework_keywords.items():
                if keyword in content_lower:
                    frameworks.append(framework)
                    
        except Exception as e:
            self.logger.debug(f"Error parsing package file {package_path}: {e}")
        
        return frameworks
    
    def _determine_project_type(self, project_path: Path, basic_analysis: Dict[str, Any], frameworks: List[str]) -> str:
        """Determine the type of project."""
        # Check based on frameworks
        if any(fw in frameworks for fw in ['React', 'Vue', 'Angular']):
            return "Frontend Web Application"
        if any(fw in frameworks for fw in ['Django', 'Flask', 'FastAPI', 'Express']):
            return "Backend Web Service"
        if 'React Native' in frameworks:
            return "Mobile Application"
        
        # Check based on file types
        languages = basic_analysis.get("language_breakdown", {})
        if 'Python' in languages:
            # Check for specific Python project types
            if (project_path / 'setup.py').exists() or (project_path / 'pyproject.toml').exists():
                return "Python Package/Library"
            if any((project_path / f).exists() for f in ['manage.py', 'wsgi.py']):
                return "Python Web Application"
            return "Python Application"
        
        if 'JavaScript' in languages or 'TypeScript' in languages:
            if (project_path / 'package.json').exists():
                return "Node.js Application"
            return "JavaScript Application"
        
        if 'Java' in languages:
            return "Java Application"
        
        if 'Go' in languages:
            return "Go Application"
        
        if 'Rust' in languages:
            return "Rust Application"
        
        return "Unknown"
    
    def _identify_key_components(self, project_path: Path, focus_area: Optional[str]) -> Dict[str, Any]:
        """Identify key components in the project."""
        components = {
            "core_modules": [],
            "utility_modules": [],
            "test_modules": [],
            "config_files": [],
            "entry_points": []
        }
        
        try:
            # Walk through the project directory
            for root, dirs, files in os.walk(project_path):
                # Skip hidden and ignore directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {
                    'node_modules', '__pycache__', 'venv', '.venv', 'dist', 'build', '.git'
                }]
                
                root_path = Path(root)
                relative_root = root_path.relative_to(project_path)
                
                for file in files:
                    if file.startswith('.'):
                        continue
                    
                    file_path = root_path / file
                    relative_path = file_path.relative_to(project_path)
                    
                    # Categorize files
                    if self._is_entry_point(file):
                        components["entry_points"].append(str(relative_path))
                    elif self._is_config_file(file):
                        components["config_files"].append(str(relative_path))
                    elif self._is_test_file(file):
                        components["test_modules"].append(str(relative_path))
                    elif self._is_utility_module(file, relative_path):
                        components["utility_modules"].append(str(relative_path))
                    elif self._is_core_module(file, relative_path):
                        components["core_modules"].append(str(relative_path))
            
            # Sort by importance/size
            for category in components:
                components[category] = sorted(components[category])[:10]  # Limit to top 10
                
        except Exception as e:
            self.logger.error(f"Error identifying components: {e}")
        
        return components
    
    def _is_entry_point(self, filename: str) -> bool:
        """Check if file is likely an entry point."""
        entry_patterns = [
            'main.py', 'app.py', 'server.py', 'index.js', 'main.js', 
            'server.js', 'app.js', 'main.go', 'main.rs', 'Main.java'
        ]
        return filename in entry_patterns
    
    def _is_config_file(self, filename: str) -> bool:
        """Check if file is a configuration file."""
        config_patterns = [
            'config.py', 'settings.py', 'config.js', 'config.json',
            'package.json', 'requirements.txt', 'pyproject.toml',
            'Dockerfile', 'docker-compose.yml', '.env', 'Makefile'
        ]
        return filename in config_patterns or filename.endswith('.config.js')
    
    def _is_test_file(self, filename: str) -> bool:
        """Check if file is a test file."""
        return ('test' in filename.lower() or 
                filename.endswith('_test.py') or 
                filename.endswith('.test.js') or
                filename.endswith('.test.ts') or
                filename.endswith('_spec.py') or
                filename.endswith('.spec.js'))
    
    def _is_utility_module(self, filename: str, relative_path: Path) -> bool:
        """Check if file is a utility module."""
        utility_indicators = ['util', 'helper', 'common', 'shared', 'lib']
        path_str = str(relative_path).lower()
        return any(indicator in path_str for indicator in utility_indicators)
    
    def _is_core_module(self, filename: str, relative_path: Path) -> bool:
        """Check if file is a core module."""
        # Heuristic: files in main source directories with substantial extensions
        if relative_path.suffix in ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs']:
            path_parts = relative_path.parts
            if len(path_parts) <= 3:  # Not too deeply nested
                return True
        return False
    
    def _analyze_dependencies(self, project_path: Path) -> Dict[str, Any]:
        """Analyze project dependencies."""
        dependencies = {
            "external_dependencies": [],
            "internal_dependencies": {},
            "circular_dependencies": []
        }
        
        # Parse package files for external dependencies
        dependencies["external_dependencies"] = self._extract_external_dependencies(project_path)
        
        # For now, skip complex internal dependency analysis
        # This would require parsing imports in source files
        
        return dependencies
    
    def _extract_external_dependencies(self, project_path: Path) -> List[str]:
        """Extract external dependencies from package files."""
        deps = []
        
        # Python dependencies
        requirements_file = project_path / 'requirements.txt'
        if requirements_file.exists():
            try:
                content = requirements_file.read_text()
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        dep = line.split('==')[0].split('>=')[0].split('<=')[0]
                        deps.append(dep)
            except Exception:
                pass
        
        # Node.js dependencies
        package_json = project_path / 'package.json'
        if package_json.exists():
            try:
                import json
                with open(package_json) as f:
                    data = json.load(f)
                    deps.extend(data.get('dependencies', {}).keys())
                    deps.extend(data.get('devDependencies', {}).keys())
            except Exception:
                pass
        
        return list(set(deps))[:20]  # Limit and deduplicate
    
    def _analyze_complexity(self, project_path: Path, basic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze project complexity metrics."""
        complexity = {
            "complexity_score": 0.0,
            "maintainability_score": 0.0,
            "test_coverage_estimate": 0.0,
            "documentation_level": "unknown"
        }
        
        total_files = basic_analysis.get("total_files", 1)
        relevant_files = basic_analysis.get("relevant_files", 0)
        
        # Simple complexity heuristics
        if total_files > 1000:
            complexity["complexity_score"] = 0.9
        elif total_files > 100:
            complexity["complexity_score"] = 0.6
        elif total_files > 20:
            complexity["complexity_score"] = 0.3
        else:
            complexity["complexity_score"] = 0.1
        
        # Estimate maintainability (inverse of complexity with some factors)
        complexity["maintainability_score"] = max(0.1, 1.0 - complexity["complexity_score"] * 0.8)
        
        # Estimate test coverage based on test files
        test_files = len([f for f in Path(project_path).rglob('*') 
                         if f.is_file() and self._is_test_file(f.name)])
        if relevant_files > 0:
            test_ratio = test_files / relevant_files
            complexity["test_coverage_estimate"] = min(0.9, test_ratio * 0.7)
        
        # Documentation level
        doc_files = len([f for f in Path(project_path).rglob('*.md')])
        if doc_files >= 5:
            complexity["documentation_level"] = "comprehensive"
        elif doc_files >= 2:
            complexity["documentation_level"] = "moderate"
        elif doc_files >= 1:
            complexity["documentation_level"] = "basic"
        else:
            complexity["documentation_level"] = "minimal"
        
        return complexity
    
    def _generate_learning_recommendations(
        self, 
        result: ProjectExplorationResult, 
        detail_level: str, 
        focus_area: Optional[str]
    ) -> Dict[str, Any]:
        """Generate learning and exploration recommendations."""
        recommendations = {
            "exploration_priorities": [],
            "recommended_reading_order": [],
            "key_concepts_to_understand": []
        }
        
        # Base recommendations on project type and architecture
        if result.project_type == "Python Web Application":
            recommendations["key_concepts_to_understand"].extend([
                "Web framework architecture",
                "Request/response cycle",
                "Database models and ORM",
                "URL routing and views",
                "Authentication and security"
            ])
        elif result.project_type == "Frontend Web Application":
            recommendations["key_concepts_to_understand"].extend([
                "Component architecture",
                "State management",
                "Routing and navigation",
                "API integration",
                "Styling and theming"
            ])
        
        # Prioritize exploration based on entry points
        if result.entry_points:
            recommendations["exploration_priorities"].append("Start with main entry points")
            recommendations["recommended_reading_order"].extend(result.entry_points[:3])
        
        # Add core modules to reading order
        if result.core_modules:
            recommendations["recommended_reading_order"].extend(result.core_modules[:5])
        
        # Focus area specific recommendations
        if focus_area:
            recommendations["exploration_priorities"].append(f"Focus on {focus_area} related components")
        
        # Detail level adjustments
        if detail_level == "comprehensive":
            recommendations["exploration_priorities"].extend([
                "Analyze all major components",
                "Understand data flow and dependencies",
                "Review configuration and environment setup",
                "Study test patterns and examples"
            ])
        elif detail_level == "detailed":
            recommendations["exploration_priorities"].extend([
                "Understand core architecture",
                "Review key components and their interactions",
                "Check configuration files"
            ])
        else:  # overview
            recommendations["exploration_priorities"].extend([
                "Get high-level architecture understanding",
                "Identify main entry points",
                "Understand project structure"
            ])
        
        return recommendations
    
    def _update_result_with_basic_analysis(self, result: ProjectExplorationResult, analysis: Dict[str, Any]):
        """Update result with basic analysis data."""
        result.files_analyzed = analysis.get("relevant_files", 0)
        
        # Extract key directories from analysis if available
        if "directory_structure" in analysis:
            result.directory_structure = analysis["directory_structure"]
    
    def _update_result_with_architecture(self, result: ProjectExplorationResult, architecture: Dict[str, Any]):
        """Update result with architecture information."""
        result.architecture_pattern = architecture.get("pattern", "unknown")
        result.project_type = architecture.get("project_type", "unknown")
        result.framework_stack = architecture.get("framework_stack", [])
    
    def _update_result_with_components(self, result: ProjectExplorationResult, components: Dict[str, Any]):
        """Update result with component information."""
        result.core_modules = components.get("core_modules", [])
        result.utility_modules = components.get("utility_modules", [])
        result.test_modules = components.get("test_modules", [])
        result.config_files = components.get("config_files", [])
        result.entry_points = components.get("entry_points", [])
    
    def _update_result_with_dependencies(self, result: ProjectExplorationResult, dependencies: Dict[str, Any]):
        """Update result with dependency information."""
        result.external_dependencies = dependencies.get("external_dependencies", [])
        result.internal_dependencies = dependencies.get("internal_dependencies", {})
        result.circular_dependencies = dependencies.get("circular_dependencies", [])
    
    def _update_result_with_complexity(self, result: ProjectExplorationResult, complexity: Dict[str, Any]):
        """Update result with complexity metrics."""
        result.complexity_score = complexity.get("complexity_score", 0.0)
        result.maintainability_score = complexity.get("maintainability_score", 0.0)
        result.test_coverage_estimate = complexity.get("test_coverage_estimate", 0.0)
        result.documentation_level = complexity.get("documentation_level", "unknown")
    
    def _update_result_with_learning_recommendations(self, result: ProjectExplorationResult, recommendations: Dict[str, Any]):
        """Update result with learning recommendations."""
        result.exploration_priorities = recommendations.get("exploration_priorities", [])
        result.recommended_reading_order = recommendations.get("recommended_reading_order", [])
        result.key_concepts_to_understand = recommendations.get("key_concepts_to_understand", [])
    
    def _calculate_confidence_score(self, result: ProjectExplorationResult) -> float:
        """Calculate overall confidence score for the analysis."""
        score = 0.0
        factors = 0
        
        # Architecture detection confidence
        if result.architecture_pattern != "unknown":
            score += 0.3
        factors += 1
        
        # Component identification confidence
        if result.entry_points:
            score += 0.3
        factors += 1
        
        # Framework detection confidence
        if result.framework_stack:
            score += 0.2
        factors += 1
        
        # Project type detection confidence
        if result.project_type != "unknown":
            score += 0.2
        factors += 1
        
        return score / factors if factors > 0 else 0.0
    
    def _load_architecture_patterns(self) -> Dict[str, Any]:
        """Load architecture pattern definitions."""
        # This would normally load from a configuration file
        return {}
    
    def _load_framework_signatures(self) -> Dict[str, Any]:
        """Load framework signature definitions."""
        # This would normally load from a configuration file
        return {}
    
    def _load_project_type_indicators(self) -> Dict[str, Any]:
        """Load project type indicator definitions."""
        # This would normally load from a configuration file
        return {}
    
    # NEW: RAG-based analysis methods
    
    def _perform_rag_analysis(
        self, 
        project_path: Path, 
        focus_area: Optional[str], 
        detail_level: str
    ) -> Dict[str, Any]:
        """
        Perform RAG-based project analysis.
        
        Args:
            project_path: Path to the project
            focus_area: Specific focus area
            detail_level: Level of detail required
            
        Returns:
            Dictionary containing RAG analysis results
        """
        if not self.rag_search_strategy:
            return {}
        
        self.logger.info("Performing RAG-based project analysis")
        rag_start_time = time.time()
        
        try:
            rag_results = {
                "architecture_analysis": {},
                "entry_points": [],
                "component_relationships": {},
                "analysis_duration": 0.0
            }
            
            # Architecture pattern detection using RAG
            if detail_level in ["detailed", "comprehensive"]:
                self.logger.debug("Running RAG architecture pattern detection")
                arch_results = self.rag_search_strategy.detect_architecture_patterns(
                    project_path=str(project_path),
                    max_results=10
                )
                rag_results["architecture_analysis"] = arch_results
            
            # Entry point discovery using RAG
            self.logger.debug("Running RAG entry point discovery")
            entry_results = self.rag_search_strategy.discover_entry_points(
                project_path=str(project_path),
                max_results=15
            )
            rag_results["entry_points"] = entry_results
            
            # Component relationship analysis (for comprehensive mode)
            if detail_level == "comprehensive":
                self.logger.debug("Running RAG component relationship analysis")
                relationship_results = self.rag_search_strategy.analyze_component_relationships(
                    project_path=str(project_path),
                    similarity_threshold=0.7,
                    max_components=20
                )
                rag_results["component_relationships"] = relationship_results
            
            rag_results["analysis_duration"] = time.time() - rag_start_time
            
            self.logger.info(f"RAG analysis completed in {rag_results['analysis_duration']:.2f}s")
            return rag_results
            
        except Exception as e:
            self.logger.warning(f"RAG analysis failed: {e}")
            return {
                "architecture_analysis": {},
                "entry_points": [],
                "component_relationships": {},
                "analysis_duration": time.time() - rag_start_time,
                "error": str(e)
            }
    
    def _integrate_rag_results(self, result: ProjectExplorationResult, rag_results: Dict[str, Any]):
        """
        Integrate RAG analysis results into the project exploration result.
        
        Args:
            result: ProjectExplorationResult to enhance
            rag_results: RAG analysis results to integrate
        """
        if not rag_results:
            return
        
        result.rag_insights_enabled = True
        
        # Store raw RAG results
        result.rag_architecture_analysis = rag_results.get("architecture_analysis", {})
        result.rag_component_relationships = rag_results.get("component_relationships", {})
        
        # Process entry points
        entry_point_data = rag_results.get("entry_points", {})
        if entry_point_data:
            result.rag_entry_points = entry_point_data.get("primary_entry_points", [])
            
            # Enhance existing entry points with RAG findings
            rag_entry_point_files = [
                ep.get("file_path", "") for ep in result.rag_entry_points
            ]
            
            for entry_file in rag_entry_point_files:
                if entry_file and entry_file not in result.entry_points:
                    result.entry_points.append(entry_file)
        
        # Enhance architecture pattern with RAG findings
        arch_analysis = result.rag_architecture_analysis
        if arch_analysis and arch_analysis.get("primary_pattern"):
            primary_pattern = arch_analysis["primary_pattern"]
            rag_pattern = primary_pattern.get("pattern_name", "")
            rag_confidence = primary_pattern.get("confidence", 0.0)
            
            # If RAG found a high-confidence pattern, use it
            if rag_confidence > 0.7 and result.architecture_pattern == "unknown":
                result.architecture_pattern = rag_pattern
                self.logger.info(f"Architecture pattern enhanced by RAG: {rag_pattern} (confidence: {rag_confidence:.2f})")
        
        # Enhance core modules with component relationship findings
        component_analysis = result.rag_component_relationships
        if component_analysis and component_analysis.get("core_components"):
            rag_components = component_analysis["core_components"]
            
            for component in rag_components[:5]:  # Top 5 components
                component_file = component.get("file_path", "")
                if component_file and component_file not in result.core_modules:
                    result.core_modules.append(component_file)
        
        # Add RAG-derived insights to learning recommendations
        self._enhance_learning_recommendations_with_rag(result, rag_results)
    
    def _enhance_learning_recommendations_with_rag(
        self, 
        result: ProjectExplorationResult, 
        rag_results: Dict[str, Any]
    ):
        """
        Enhance learning recommendations using RAG insights.
        
        Args:
            result: ProjectExplorationResult to enhance
            rag_results: RAG analysis results
        """
        # Add architecture-specific insights
        arch_analysis = rag_results.get("architecture_analysis", {})
        if arch_analysis and arch_analysis.get("primary_pattern"):
            pattern_name = arch_analysis["primary_pattern"].get("pattern_name", "")
            if pattern_name:
                pattern_recommendation = f"Study {pattern_name} architecture pattern implementation"
                if pattern_recommendation not in result.key_concepts_to_understand:
                    result.key_concepts_to_understand.insert(0, pattern_recommendation)
        
        # Add entry point insights
        entry_analysis = rag_results.get("entry_points", {})
        if entry_analysis and entry_analysis.get("primary_entry_points"):
            primary_entries = entry_analysis["primary_entry_points"]
            
            # Add entry points to reading order
            for entry_point in primary_entries[:3]:  # Top 3
                entry_file = entry_point.get("file_path", "")
                entry_name = entry_point.get("function_name", "")
                
                if entry_file:
                    entry_description = f"{entry_file}"
                    if entry_name:
                        entry_description += f" ({entry_name})"
                    
                    if entry_description not in result.recommended_reading_order:
                        result.recommended_reading_order.insert(0, entry_description)
        
        # Add component relationship insights
        relationship_analysis = rag_results.get("component_relationships", {})
        if relationship_analysis and relationship_analysis.get("architectural_insights"):
            insights = relationship_analysis["architectural_insights"]
            for insight in insights[:2]:  # Top 2 insights
                if insight not in result.coding_patterns:
                    result.coding_patterns.append(insight)

    def format_exploration_summary(self, result: ProjectExplorationResult, detail_level: str = "overview") -> str:
        """Format exploration result into a human-readable summary."""
        summary_parts = []
        
        # Header
        summary_parts.append(f"# 📊 Project Exploration: {result.project_name}")
        summary_parts.append("")
        
        # Basic Information
        summary_parts.append("## 🏗️ **Project Overview**")
        summary_parts.append(f"- **Type**: {result.project_type}")
        summary_parts.append(f"- **Architecture**: {result.architecture_pattern}")
        if result.framework_stack:
            summary_parts.append(f"- **Frameworks**: {', '.join(result.framework_stack)}")
        summary_parts.append(f"- **Files Analyzed**: {result.files_analyzed}")
        summary_parts.append("")
        
        # Entry Points
        if result.entry_points:
            summary_parts.append("## 🚀 **Entry Points**")
            for entry in result.entry_points[:5]:
                summary_parts.append(f"- `{entry}`")
            summary_parts.append("")
        
        # Core Modules
        if result.core_modules and detail_level in ["detailed", "comprehensive"]:
            summary_parts.append("## 🔧 **Core Modules**")
            for module in result.core_modules[:5]:
                summary_parts.append(f"- `{module}`")
            summary_parts.append("")
        
        # Exploration Priorities
        if result.exploration_priorities:
            summary_parts.append("## 📋 **Exploration Priorities**")
            for i, priority in enumerate(result.exploration_priorities[:5], 1):
                summary_parts.append(f"{i}. {priority}")
            summary_parts.append("")
        
        # Learning Path
        if result.recommended_reading_order:
            summary_parts.append("## 📚 **Recommended Reading Order**")
            for i, item in enumerate(result.recommended_reading_order[:5], 1):
                summary_parts.append(f"{i}. `{item}`")
            summary_parts.append("")
        
        # Key Concepts
        if result.key_concepts_to_understand:
            summary_parts.append("## 💡 **Key Concepts to Understand**")
            for concept in result.key_concepts_to_understand:
                summary_parts.append(f"- {concept}")
            summary_parts.append("")
        
        # Complexity Metrics (for detailed/comprehensive)
        if detail_level in ["detailed", "comprehensive"]:
            summary_parts.append("## 📈 **Project Metrics**")
            summary_parts.append(f"- **Complexity Score**: {result.complexity_score:.1f}/1.0")
            summary_parts.append(f"- **Maintainability**: {result.maintainability_score:.1f}/1.0")
            summary_parts.append(f"- **Documentation Level**: {result.documentation_level}")
            if result.test_coverage_estimate > 0:
                summary_parts.append(f"- **Test Coverage Estimate**: {result.test_coverage_estimate:.1%}")
            summary_parts.append("")
        
        # External Dependencies (for comprehensive)
        if detail_level == "comprehensive" and result.external_dependencies:
            summary_parts.append("## 📦 **Key Dependencies**")
            for dep in result.external_dependencies[:10]:
                summary_parts.append(f"- {dep}")
            summary_parts.append("")
        
        # RAG-Enhanced Insights (NEW)
        if result.rag_insights_enabled:
            summary_parts.append("## 🔍 **RAG-Enhanced Insights**")
            
            # RAG Architecture Analysis
            if result.rag_architecture_analysis and result.rag_architecture_analysis.get("primary_pattern"):
                primary_pattern = result.rag_architecture_analysis["primary_pattern"]
                pattern_name = primary_pattern.get("pattern_name", "Unknown")
                confidence = primary_pattern.get("confidence", 0.0)
                summary_parts.append(f"- **Architecture Pattern Detected**: {pattern_name} (confidence: {confidence:.1%})")
            
            # RAG Entry Points
            if result.rag_entry_points:
                summary_parts.append(f"- **Function-Level Entry Points Found**: {len(result.rag_entry_points)}")
                for entry in result.rag_entry_points[:3]:  # Top 3
                    entry_type = entry.get("entry_type", "Entry Point")
                    entry_name = entry.get("function_name", "")
                    entry_file = entry.get("file_path", "").split("/")[-1]  # Just filename
                    summary_parts.append(f"  - {entry_type}: `{entry_name}` in `{entry_file}`")
            
            # RAG Component Relationships
            if result.rag_component_relationships and result.rag_component_relationships.get("analysis_summary"):
                rel_summary = result.rag_component_relationships["analysis_summary"]
                components_count = rel_summary.get("components_analyzed", 0)
                relationships_count = rel_summary.get("relationships_found", 0)
                if components_count > 0:
                    summary_parts.append(f"- **Component Analysis**: {components_count} components, {relationships_count} relationships")
            
            summary_parts.append("")
        
        # Analysis Info
        summary_parts.append("## ⏱️ **Analysis Info**")
        summary_parts.append(f"- **Duration**: {result.analysis_duration_seconds:.2f} seconds")
        summary_parts.append(f"- **Confidence**: {result.confidence_score:.1%}")
        summary_parts.append(f"- **Timestamp**: {result.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        if result.rag_insights_enabled:
            summary_parts.append(f"- **RAG Analysis**: Enabled (function-level insights included)")
        
        return "\n".join(summary_parts)