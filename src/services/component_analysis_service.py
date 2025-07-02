"""
Component Analysis Service

This module provides deep component analysis capabilities for the understand_component
MCP prompt, offering detailed insights into component structure, dependencies, and usage patterns.
"""

import logging
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from services.code_parser_service import CodeParserService
    from services.embedding_service import EmbeddingService
except ImportError:
    # For testing without full service dependencies
    EmbeddingService = None
    CodeParserService = None


logger = logging.getLogger(__name__)


@dataclass
class ComponentMatch:
    """Represents a matched component in the codebase."""

    file_path: str
    name: str
    component_type: str  # class, function, method, variable, etc.
    start_line: int
    end_line: int
    signature: str = ""
    docstring: str = ""
    content: str = ""
    language: str = ""

    # Context information
    parent_class: str | None = None
    module_path: str = ""
    imports: list[str] = field(default_factory=list)

    # Analysis metadata
    complexity_score: float = 0.0
    usage_count: int = 0
    confidence_score: float = 0.0


@dataclass
class DependencyAnalysis:
    """Analysis of component dependencies."""

    # Direct dependencies
    imports_from: list[str] = field(default_factory=list)
    calls_to: list[str] = field(default_factory=list)
    inherits_from: list[str] = field(default_factory=list)

    # Reverse dependencies (what depends on this component)
    imported_by: list[str] = field(default_factory=list)
    called_by: list[str] = field(default_factory=list)
    inherited_by: list[str] = field(default_factory=list)

    # Dependency metrics
    coupling_score: float = 0.0  # 0 = low coupling, 1 = high coupling
    fan_in: int = 0  # Number of components depending on this one
    fan_out: int = 0  # Number of components this one depends on


@dataclass
class UsagePattern:
    """Pattern of how a component is used."""

    usage_type: str  # instantiation, method_call, inheritance, import, etc.
    file_path: str
    line_number: int
    context: str  # Surrounding code context
    example_usage: str
    pattern_frequency: int = 1


@dataclass
class ComponentAnalysisResult:
    """Comprehensive analysis result for a component."""

    # Component identification
    component_name: str
    analysis_timestamp: datetime = field(default_factory=datetime.now)

    # Matched components
    exact_matches: list[ComponentMatch] = field(default_factory=list)
    partial_matches: list[ComponentMatch] = field(default_factory=list)
    related_components: list[ComponentMatch] = field(default_factory=list)

    # Component details
    primary_component: ComponentMatch | None = None
    component_summary: str = ""
    purpose_description: str = ""

    # Interface analysis
    public_methods: list[dict[str, Any]] = field(default_factory=list)
    parameters: list[dict[str, Any]] = field(default_factory=list)
    return_types: list[str] = field(default_factory=list)

    # Dependencies and relationships
    dependency_analysis: DependencyAnalysis = field(default_factory=DependencyAnalysis)
    related_concepts: list[str] = field(default_factory=list)

    # Usage analysis
    usage_patterns: list[UsagePattern] = field(default_factory=list)
    common_use_cases: list[str] = field(default_factory=list)
    best_practices: list[str] = field(default_factory=list)

    # Quality metrics
    complexity_assessment: str = "unknown"
    maintainability_score: float = 0.0
    test_coverage_info: dict[str, Any] = field(default_factory=dict)

    # Documentation and examples
    documentation_quality: str = "unknown"
    code_examples: list[str] = field(default_factory=list)
    external_references: list[str] = field(default_factory=list)

    # Analysis metadata
    analysis_duration_seconds: float = 0.0
    confidence_score: float = 0.0
    components_analyzed: int = 0


class ComponentAnalysisService:
    """
    Advanced component analysis service for deep codebase component understanding.

    Provides comprehensive component analysis including interface detection,
    dependency mapping, usage pattern analysis, and quality assessment.
    """

    def __init__(self):
        self.logger = logger
        self.embedding_service = EmbeddingService() if EmbeddingService else None
        self.code_parser = CodeParserService() if CodeParserService else None

        # Pattern databases for analysis
        self.language_patterns = self._load_language_patterns()
        self.complexity_indicators = self._load_complexity_indicators()

    def analyze_component(
        self,
        component_name: str,
        project_path: str = ".",
        component_type: str = "auto",
        include_dependencies: bool = True,
        include_usage_examples: bool = True,
        analyze_quality: bool = True,
    ) -> ComponentAnalysisResult:
        """
        Perform comprehensive component analysis.

        Args:
            component_name: Name of component to analyze
            project_path: Path to search for the component
            component_type: Type hint for component (class, function, auto, etc.)
            include_dependencies: Whether to analyze dependencies
            include_usage_examples: Whether to find usage examples
            analyze_quality: Whether to perform quality analysis

        Returns:
            ComponentAnalysisResult with comprehensive insights
        """
        start_time = time.time()

        project_path = Path(project_path).resolve()

        self.logger.info(f"Starting component analysis for: {component_name}")

        # Initialize result
        result = ComponentAnalysisResult(component_name=component_name)

        try:
            # Phase 1: Component discovery and matching
            matches = self._find_component_matches(component_name, project_path, component_type)
            self._update_result_with_matches(result, matches)

            # Phase 2: Interface analysis
            if result.primary_component:
                interface_info = self._analyze_component_interface(result.primary_component)
                self._update_result_with_interface(result, interface_info)

            # Phase 3: Dependency analysis
            if include_dependencies and result.primary_component:
                dependencies = self._analyze_dependencies(result.primary_component, project_path)
                result.dependency_analysis = dependencies

            # Phase 4: Usage pattern analysis
            if include_usage_examples:
                usage_patterns = self._analyze_usage_patterns(component_name, project_path, result.exact_matches)
                result.usage_patterns = usage_patterns
                self._generate_use_cases_and_best_practices(result)

            # Phase 5: Quality analysis
            if analyze_quality and result.primary_component:
                quality_info = self._analyze_component_quality(result.primary_component, project_path)
                self._update_result_with_quality(result, quality_info)

            # Phase 6: Generate summary and insights
            self._generate_component_summary(result)

            # Finalize result
            result.analysis_duration_seconds = time.time() - start_time
            result.confidence_score = self._calculate_confidence_score(result)
            result.components_analyzed = len(result.exact_matches) + len(result.partial_matches)

            self.logger.info(f"Component analysis completed in {result.analysis_duration_seconds:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error during component analysis: {e}")
            result.component_summary = f"Analysis error: {str(e)}"
            result.analysis_duration_seconds = time.time() - start_time
            return result

    def _find_component_matches(self, component_name: str, project_path: Path, component_type: str) -> dict[str, list[ComponentMatch]]:
        """Find all matches for the component in the codebase."""
        matches = {"exact": [], "partial": [], "related": []}

        # Different search strategies based on component type
        if component_type == "auto":
            search_patterns = [
                self._create_class_pattern(component_name),
                self._create_function_pattern(component_name),
                self._create_variable_pattern(component_name),
                self._create_module_pattern(component_name),
            ]
        else:
            search_patterns = [self._create_typed_pattern(component_name, component_type)]

        # Search through source files
        for root, dirs, files in os.walk(project_path):
            # Skip common ignore directories
            dirs[:] = [
                d for d in dirs if not d.startswith(".") and d not in {"node_modules", "__pycache__", "venv", ".venv", "dist", "build"}
            ]

            for file in files:
                if self._is_source_file(file):
                    file_path = Path(root) / file
                    file_matches = self._search_file_for_component(file_path, component_name, search_patterns)

                    for match_type, match_list in file_matches.items():
                        matches[match_type].extend(match_list)

        return matches

    def _create_class_pattern(self, name: str) -> dict[str, Any]:
        """Create search pattern for class definitions."""
        return {
            "type": "class",
            "patterns": [
                rf"class\s+{re.escape(name)}\s*\(",
                rf"class\s+{re.escape(name)}\s*:",
                rf"class\s+{re.escape(name)}\s*\{{",  # JavaScript/TypeScript
                rf"interface\s+{re.escape(name)}\s*\{{",  # TypeScript interface
            ],
        }

    def _create_function_pattern(self, name: str) -> dict[str, Any]:
        """Create search pattern for function definitions."""
        return {
            "type": "function",
            "patterns": [
                rf"def\s+{re.escape(name)}\s*\(",  # Python
                rf"function\s+{re.escape(name)}\s*\(",  # JavaScript
                rf"const\s+{re.escape(name)}\s*=\s*\(",  # Arrow function
                rf"let\s+{re.escape(name)}\s*=\s*\(",
                rf"var\s+{re.escape(name)}\s*=\s*\(",
                rf"func\s+{re.escape(name)}\s*\(",  # Go
                rf"fn\s+{re.escape(name)}\s*\(",  # Rust
            ],
        }

    def _create_variable_pattern(self, name: str) -> dict[str, Any]:
        """Create search pattern for variable definitions."""
        return {
            "type": "variable",
            "patterns": [
                rf"^{re.escape(name)}\s*=",  # Assignment
                rf"const\s+{re.escape(name)}\s*=",
                rf"let\s+{re.escape(name)}\s*=",
                rf"var\s+{re.escape(name)}\s*=",
            ],
        }

    def _create_module_pattern(self, name: str) -> dict[str, Any]:
        """Create search pattern for module/file references."""
        return {
            "type": "module",
            "patterns": [
                rf"from\s+.*{re.escape(name)}.*\s+import",  # Python import
                rf"import\s+.*{re.escape(name)}",
                rf"require\s*\(\s*['\"].*{re.escape(name)}.*['\"]\s*\)",  # Node.js
                rf"import\s+.*from\s+['\"].*{re.escape(name)}.*['\"]",  # ES6
            ],
        }

    def _create_typed_pattern(self, name: str, component_type: str) -> dict[str, Any]:
        """Create search pattern for specific component type."""
        if component_type == "class":
            return self._create_class_pattern(name)
        elif component_type == "function":
            return self._create_function_pattern(name)
        elif component_type == "variable":
            return self._create_variable_pattern(name)
        elif component_type == "module":
            return self._create_module_pattern(name)
        else:
            return self._create_function_pattern(name)  # Default fallback

    def _is_source_file(self, filename: str) -> bool:
        """Check if file is a source code file."""
        source_extensions = {
            ".py",
            ".js",
            ".ts",
            ".jsx",
            ".tsx",
            ".java",
            ".go",
            ".rs",
            ".cpp",
            ".c",
            ".h",
        }
        return Path(filename).suffix.lower() in source_extensions

    def _search_file_for_component(
        self, file_path: Path, component_name: str, patterns: list[dict[str, Any]]
    ) -> dict[str, list[ComponentMatch]]:
        """Search a single file for component matches."""
        matches = {"exact": [], "partial": [], "related": []}

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")

            for pattern_info in patterns:
                component_type = pattern_info["type"]

                for pattern in pattern_info["patterns"]:
                    for line_num, line in enumerate(lines, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            match = self._create_component_match(
                                file_path,
                                component_name,
                                component_type,
                                line_num,
                                line,
                                content,
                                lines,
                            )

                            # Classify match quality
                            if self._is_exact_match(match, component_name):
                                matches["exact"].append(match)
                            elif self._is_partial_match(match, component_name):
                                matches["partial"].append(match)
                            else:
                                matches["related"].append(match)

        except Exception as e:
            self.logger.debug(f"Error searching file {file_path}: {e}")

        return matches

    def _create_component_match(
        self,
        file_path: Path,
        component_name: str,
        component_type: str,
        line_num: int,
        line: str,
        content: str,
        lines: list[str],
    ) -> ComponentMatch:
        """Create a ComponentMatch object from search results."""
        # Extract more context around the match
        start_line = max(0, line_num - 1)
        end_line = min(len(lines), line_num + 10)  # Get some context

        # Try to find the actual component boundaries
        if component_type == "class":
            end_line = self._find_class_end(lines, start_line)
        elif component_type == "function":
            end_line = self._find_function_end(lines, start_line)

        context_content = "\n".join(lines[start_line:end_line])

        # Extract signature and docstring
        signature = self._extract_signature(line, component_type)
        docstring = self._extract_docstring(lines, line_num)

        return ComponentMatch(
            file_path=str(file_path),
            name=component_name,
            component_type=component_type,
            start_line=line_num,
            end_line=end_line,
            signature=signature,
            docstring=docstring,
            content=context_content,
            language=self._detect_language(file_path),
            module_path=self._get_module_path(file_path),
            confidence_score=self._calculate_match_confidence(line, component_name),
        )

    def _find_class_end(self, lines: list[str], start_idx: int) -> int:
        """Find the end line of a class definition."""
        indent_level = len(lines[start_idx]) - len(lines[start_idx].lstrip())

        for i in range(start_idx + 1, min(len(lines), start_idx + 100)):
            line = lines[i]
            if line.strip() and (len(line) - len(line.lstrip())) <= indent_level:
                if not line.strip().startswith(('"""', "'''", "#")):
                    return i

        return min(len(lines), start_idx + 50)  # Default fallback

    def _find_function_end(self, lines: list[str], start_idx: int) -> int:
        """Find the end line of a function definition."""
        # Simple heuristic - find next function/class definition or end of indentation
        return self._find_class_end(lines, start_idx)

    def _extract_signature(self, line: str, component_type: str) -> str:
        """Extract the signature from a definition line."""
        line = line.strip()

        if component_type == "class":
            if ":" in line:
                return line[: line.index(":")]
            elif "{" in line:
                return line[: line.index("{")]
        elif component_type == "function":
            if ":" in line:
                return line[: line.index(":")]
            elif "{" in line:
                return line[: line.index("{")]

        return line

    def _extract_docstring(self, lines: list[str], start_line: int) -> str:
        """Extract docstring/comment for a component."""
        docstring_lines = []

        # Look for docstring after the definition
        for i in range(start_line, min(len(lines), start_line + 10)):
            line = lines[i].strip()
            if line.startswith('"""') or line.startswith("'''"):
                # Multi-line docstring
                quote_type = line[:3]
                if line.endswith(quote_type) and len(line) > 6:
                    # Single line docstring
                    return line[3:-3].strip()
                else:
                    # Multi-line docstring
                    for j in range(i + 1, min(len(lines), i + 20)):
                        docstring_lines.append(lines[j].strip())
                        if lines[j].strip().endswith(quote_type):
                            break
                    return "\n".join(docstring_lines[:-1])
            elif line.startswith("//") or line.startswith("#"):
                # Single line comment
                docstring_lines.append(line[2:].strip())
            elif line.startswith("/*"):
                # Multi-line comment
                comment_lines = [line[2:]]
                for j in range(i + 1, min(len(lines), i + 10)):
                    comment_lines.append(lines[j].strip())
                    if lines[j].strip().endswith("*/"):
                        comment_lines[-1] = comment_lines[-1][:-2]
                        break
                return "\n".join(comment_lines).strip()
            elif line and not line.startswith(("def ", "class ", "function ", "const ", "let ", "var ")):
                continue
            else:
                break

        return "\n".join(docstring_lines) if docstring_lines else ""

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        ext_map = {
            ".py": "Python",
            ".js": "JavaScript",
            ".ts": "TypeScript",
            ".jsx": "React",
            ".tsx": "React TypeScript",
            ".java": "Java",
            ".go": "Go",
            ".rs": "Rust",
            ".cpp": "C++",
            ".c": "C",
        }
        return ext_map.get(file_path.suffix.lower(), "Unknown")

    def _get_module_path(self, file_path: Path) -> str:
        """Get the module path for the component."""
        # Convert file path to module path (simplified)
        path_parts = file_path.parts
        if len(path_parts) > 1:
            return ".".join(path_parts[-2:]).replace(".py", "").replace(".js", "")
        return file_path.stem

    def _calculate_match_confidence(self, line: str, component_name: str) -> float:
        """Calculate confidence score for a match."""
        # Simple heuristic based on exact name match and context
        if component_name.lower() in line.lower():
            if f" {component_name} " in line or f"({component_name}" in line:
                return 0.9
            elif component_name in line:
                return 0.7
        return 0.5

    def _is_exact_match(self, match: ComponentMatch, component_name: str) -> bool:
        """Check if this is an exact match for the component."""
        return match.confidence_score >= 0.8 and component_name.lower() in match.signature.lower()

    def _is_partial_match(self, match: ComponentMatch, component_name: str) -> bool:
        """Check if this is a partial match for the component."""
        return match.confidence_score >= 0.6

    def _update_result_with_matches(self, result: ComponentAnalysisResult, matches: dict[str, list[ComponentMatch]]):
        """Update result with component matches."""
        result.exact_matches = matches["exact"]
        result.partial_matches = matches["partial"]
        result.related_components = matches["related"]

        # Select primary component (best exact match)
        if result.exact_matches:
            result.primary_component = max(result.exact_matches, key=lambda m: m.confidence_score)
        elif result.partial_matches:
            result.primary_component = max(result.partial_matches, key=lambda m: m.confidence_score)

    def _analyze_component_interface(self, component: ComponentMatch) -> dict[str, Any]:
        """Analyze the interface of a component."""
        interface_info = {"public_methods": [], "parameters": [], "return_types": []}

        # Parse the component content for interface details
        if component.component_type == "class":
            interface_info["public_methods"] = self._extract_class_methods(component.content)
        elif component.component_type == "function":
            interface_info["parameters"] = self._extract_function_parameters(component.signature)
            interface_info["return_types"] = self._extract_return_type(component.signature)

        return interface_info

    def _extract_class_methods(self, content: str) -> list[dict[str, Any]]:
        """Extract methods from class content."""
        methods = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith("def ") and not line.startswith("def __"):
                # Public method
                method_name = line.split("(")[0].replace("def ", "").strip()
                method_sig = line.split(":")[0] if ":" in line else line

                methods.append(
                    {
                        "name": method_name,
                        "signature": method_sig,
                        "line_number": i + 1,
                        "is_public": not method_name.startswith("_"),
                    }
                )

        return methods

    def _extract_function_parameters(self, signature: str) -> list[dict[str, Any]]:
        """Extract parameters from function signature."""
        parameters = []

        # Simple parameter extraction
        if "(" in signature and ")" in signature:
            param_part = signature[signature.index("(") + 1 : signature.rindex(")")]
            if param_part.strip():
                params = [p.strip() for p in param_part.split(",")]
                for param in params:
                    if param and param != "self":
                        param_info = {"name": param.split("=")[0].strip()}
                        if "=" in param:
                            param_info["default_value"] = param.split("=")[1].strip()
                        if ":" in param:
                            parts = param.split(":")
                            param_info["name"] = parts[0].strip()
                            param_info["type_hint"] = parts[1].split("=")[0].strip()

                        parameters.append(param_info)

        return parameters

    def _extract_return_type(self, signature: str) -> list[str]:
        """Extract return type from function signature."""
        return_types = []

        if "->" in signature:
            return_part = signature.split("->")[1].split(":")[0].strip()
            return_types.append(return_part)

        return return_types

    def _analyze_dependencies(self, component: ComponentMatch, project_path: Path) -> DependencyAnalysis:
        """Analyze component dependencies."""
        deps = DependencyAnalysis()

        # Extract imports from the file
        try:
            file_content = Path(component.file_path).read_text(encoding="utf-8", errors="ignore")
            deps.imports_from = self._extract_imports(file_content, component.language)

            # Simple call analysis within the component
            deps.calls_to = self._extract_function_calls(component.content)

            # Inheritance analysis for classes
            if component.component_type == "class":
                deps.inherits_from = self._extract_inheritance(component.signature)

        except Exception as e:
            self.logger.debug(f"Error analyzing dependencies: {e}")

        return deps

    def _extract_imports(self, content: str, language: str) -> list[str]:
        """Extract imports from file content."""
        imports = []
        lines = content.split("\n")

        for line in lines:
            line = line.strip()
            if language.lower() == "python":
                if line.startswith("import ") or line.startswith("from "):
                    imports.append(line)
            elif language.lower() in ["javascript", "typescript"]:
                if line.startswith("import ") or line.startswith("const ") and "require(" in line:
                    imports.append(line)

        return imports[:10]  # Limit to avoid overwhelming output

    def _extract_function_calls(self, content: str) -> list[str]:
        """Extract function calls from component content."""

        # Simple regex to find function calls
        call_pattern = r"(\w+)\s*\("
        matches = re.findall(call_pattern, content)

        # Filter out common keywords and duplicates
        common_keywords = {
            "if",
            "for",
            "while",
            "def",
            "class",
            "return",
            "print",
            "len",
            "str",
            "int",
        }
        unique_calls = set(matches) - common_keywords

        return list(unique_calls)[:15]  # Limit output

    def _extract_inheritance(self, signature: str) -> list[str]:
        """Extract inheritance information from class signature."""
        inheritance = []

        if "(" in signature and ")" in signature:
            parents_part = signature[signature.index("(") + 1 : signature.rindex(")")]
            if parents_part.strip():
                parents = [p.strip() for p in parents_part.split(",")]
                inheritance.extend(parents)

        return inheritance

    def _analyze_usage_patterns(self, component_name: str, project_path: Path, matches: list[ComponentMatch]) -> list[UsagePattern]:
        """Analyze how the component is used throughout the codebase."""
        patterns = []

        if not matches:
            return patterns

        # Search for usage patterns in other files
        usage_files = self._find_usage_files(component_name, project_path, matches)

        for file_path, usages in usage_files.items():
            for usage in usages:
                pattern = UsagePattern(
                    usage_type=usage["type"],
                    file_path=file_path,
                    line_number=usage["line"],
                    context=usage["context"],
                    example_usage=usage["example"],
                )
                patterns.append(pattern)

        return patterns[:10]  # Limit to most relevant patterns

    def _find_usage_files(self, component_name: str, project_path: Path, matches: list[ComponentMatch]) -> dict[str, list[dict]]:
        """Find files that use the component."""
        usage_files = defaultdict(list)
        component_files = {match.file_path for match in matches}

        # Search for component usage in other files
        for root, dirs, files in os.walk(project_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {"node_modules", "__pycache__", "venv", ".venv"}]

            for file in files:
                if self._is_source_file(file):
                    file_path = str(Path(root) / file)

                    # Skip the files where the component is defined
                    if file_path in component_files:
                        continue

                    usages = self._find_component_usages_in_file(file_path, component_name)
                    if usages:
                        usage_files[file_path].extend(usages)

        return dict(usage_files)

    def _find_component_usages_in_file(self, file_path: str, component_name: str) -> list[dict]:
        """Find usages of a component in a specific file."""
        usages = []

        try:
            content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                if component_name in line:
                    # Determine usage type
                    usage_type = self._classify_usage_type(line, component_name)

                    if usage_type:
                        # Get surrounding context
                        start_idx = max(0, i - 3)
                        end_idx = min(len(lines), i + 2)
                        context = "\n".join(lines[start_idx:end_idx])

                        usages.append(
                            {
                                "type": usage_type,
                                "line": i,
                                "context": context,
                                "example": line.strip(),
                            }
                        )

        except Exception as e:
            self.logger.debug(f"Error analyzing usage in {file_path}: {e}")

        return usages

    def _classify_usage_type(self, line: str, component_name: str) -> str | None:
        """Classify the type of component usage in a line."""
        line_lower = line.lower()
        name_lower = component_name.lower()

        if f"import {name_lower}" in line_lower or "from " in line_lower and name_lower in line_lower:
            return "import"
        elif f"{name_lower}(" in line_lower:
            return "instantiation"
        elif f".{name_lower}(" in line_lower:
            return "method_call"
        elif "class " in line_lower and name_lower in line_lower:
            return "inheritance"
        elif f"{name_lower}." in line_lower:
            return "attribute_access"
        elif name_lower in line_lower:
            return "reference"

        return None

    def _analyze_component_quality(self, component: ComponentMatch, project_path: Path) -> dict[str, Any]:
        """Analyze component quality metrics."""
        quality = {
            "complexity_assessment": "unknown",
            "maintainability_score": 0.0,
            "documentation_quality": "unknown",
            "test_coverage_info": {},
        }

        # Simple complexity analysis based on content
        complexity_score = self._calculate_complexity_score(component.content)

        if complexity_score < 0.3:
            quality["complexity_assessment"] = "low"
        elif complexity_score < 0.7:
            quality["complexity_assessment"] = "moderate"
        else:
            quality["complexity_assessment"] = "high"

        # Maintainability (inverse of complexity with documentation boost)
        doc_boost = 0.2 if component.docstring else 0.0
        quality["maintainability_score"] = max(0.1, 1.0 - complexity_score + doc_boost)

        # Documentation quality
        if component.docstring:
            if len(component.docstring) > 100:
                quality["documentation_quality"] = "comprehensive"
            elif len(component.docstring) > 30:
                quality["documentation_quality"] = "adequate"
            else:
                quality["documentation_quality"] = "minimal"
        else:
            quality["documentation_quality"] = "none"

        # Look for test files
        test_info = self._find_test_files(component, project_path)
        quality["test_coverage_info"] = test_info

        return quality

    def _calculate_complexity_score(self, content: str) -> float:
        """Calculate complexity score based on content analysis."""
        if not content:
            return 0.0

        lines = content.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        # Simple heuristics
        line_count = len(non_empty_lines)
        complexity_keywords = ["if", "for", "while", "try", "except", "switch", "case"]

        complexity_count = sum(1 for line in non_empty_lines for keyword in complexity_keywords if keyword in line.lower())

        # Normalize to 0-1 scale
        base_score = min(1.0, line_count / 100.0)  # Lines contribute to complexity
        complexity_boost = min(0.5, complexity_count / 10.0)  # Control structures add complexity

        return min(1.0, base_score + complexity_boost)

    def _find_test_files(self, component: ComponentMatch, project_path: Path) -> dict[str, Any]:
        """Find test files related to the component."""
        test_info = {
            "has_tests": False,
            "test_files": [],
            "estimated_coverage": "unknown",
        }

        component_file = Path(component.file_path)
        component_name = component.name.lower()

        # Look for test files
        test_patterns = [
            f"test_{component_file.stem}.py",
            f"{component_file.stem}_test.py",
            f"test_{component_name}.py",
            f"{component_name}_test.py",
        ]

        for root, _dirs, files in os.walk(project_path):
            for file in files:
                if any(pattern in file.lower() for pattern in test_patterns):
                    test_info["test_files"].append(str(Path(root) / file))
                    test_info["has_tests"] = True

        if test_info["has_tests"]:
            test_info["estimated_coverage"] = "partial"

        return test_info

    def _generate_use_cases_and_best_practices(self, result: ComponentAnalysisResult):
        """Generate common use cases and best practices from usage patterns."""
        if not result.usage_patterns:
            return

        # Analyze usage patterns to generate insights
        usage_types = Counter(pattern.usage_type for pattern in result.usage_patterns)

        # Generate use cases based on most common usage patterns
        common_use_cases = []
        if "instantiation" in usage_types:
            common_use_cases.append("Creating instances of the component")
        if "method_call" in usage_types:
            common_use_cases.append("Calling component methods")
        if "inheritance" in usage_types:
            common_use_cases.append("Extending the component through inheritance")
        if "import" in usage_types:
            common_use_cases.append("Importing and using in other modules")

        result.common_use_cases = common_use_cases

        # Generate best practices
        best_practices = []
        if result.primary_component and result.primary_component.docstring:
            best_practices.append("Component is well-documented with docstrings")
        if result.dependency_analysis.fan_out < 5:
            best_practices.append("Component has low coupling with other modules")
        if any("test" in tf for tf in result.test_coverage_info.get("test_files", [])):
            best_practices.append("Component has associated test files")

        result.best_practices = best_practices

    def _generate_component_summary(self, result: ComponentAnalysisResult):
        """Generate a summary description of the component."""
        if not result.primary_component:
            result.component_summary = f"No primary component found for '{result.component_name}'"
            return

        primary = result.primary_component
        summary_parts = []

        # Basic description
        summary_parts.append(f"'{primary.name}' is a {primary.component_type} in {primary.language}")

        # Location
        rel_path = Path(primary.file_path).name
        summary_parts.append(f"defined in {rel_path}")

        # Purpose from docstring
        if primary.docstring:
            purpose = primary.docstring.split("\n")[0][:100]
            summary_parts.append(f"Purpose: {purpose}")

        # Interface info
        if result.public_methods:
            method_count = len(result.public_methods)
            summary_parts.append(f"Provides {method_count} public methods")

        if result.parameters:
            param_count = len(result.parameters)
            summary_parts.append(f"Takes {param_count} parameters")

        # Usage info
        if result.usage_patterns:
            usage_count = len(result.usage_patterns)
            summary_parts.append(f"Found {usage_count} usage examples")

        result.component_summary = ". ".join(summary_parts) + "."

    def _calculate_confidence_score(self, result: ComponentAnalysisResult) -> float:
        """Calculate overall confidence score for the analysis."""
        score = 0.0
        factors = 0

        # Primary component confidence
        if result.primary_component:
            score += result.primary_component.confidence_score * 0.4
        factors += 1

        # Match quality
        total_matches = len(result.exact_matches) + len(result.partial_matches)
        if total_matches > 0:
            match_score = len(result.exact_matches) / total_matches
            score += match_score * 0.3
        factors += 1

        # Documentation quality
        if result.primary_component and result.primary_component.docstring:
            score += 0.2
        factors += 1

        # Usage pattern availability
        if result.usage_patterns:
            score += 0.1
        factors += 1

        return score / factors if factors > 0 else 0.0

    def _load_language_patterns(self) -> dict[str, Any]:
        """Load language-specific patterns."""
        return {}

    def _load_complexity_indicators(self) -> dict[str, Any]:
        """Load complexity indicator patterns."""
        return {}

    def format_analysis_summary(self, result: ComponentAnalysisResult, detail_level: str = "overview") -> str:
        """Format analysis result into a human-readable summary."""
        summary_parts = []

        # Header
        summary_parts.append(f"# 🔍 Component Analysis: {result.component_name}")
        summary_parts.append("")

        # Basic Information
        summary_parts.append("## 📋 **Component Overview**")
        summary_parts.append(f"- **Summary**: {result.component_summary}")
        if result.primary_component:
            summary_parts.append(f"- **Type**: {result.primary_component.component_type.title()}")
            summary_parts.append(f"- **Language**: {result.primary_component.language}")
            summary_parts.append(f"- **Location**: `{Path(result.primary_component.file_path).name}:{result.primary_component.start_line}`")
        summary_parts.append("")

        # Signature and Interface
        if result.primary_component and result.primary_component.signature:
            summary_parts.append("## 🔧 **Component Interface**")
            summary_parts.append(f"```{result.primary_component.language.lower()}")
            summary_parts.append(result.primary_component.signature)
            summary_parts.append("```")
            summary_parts.append("")

        # Documentation
        if result.primary_component and result.primary_component.docstring:
            summary_parts.append("## 📚 **Documentation**")
            summary_parts.append(f"> {result.primary_component.docstring}")
            summary_parts.append("")

        # Public Methods (for classes)
        if result.public_methods and detail_level in ["detailed", "comprehensive"]:
            summary_parts.append("## 🛠️ **Public Methods**")
            for method in result.public_methods[:5]:
                summary_parts.append(f"- `{method['signature']}`")
            summary_parts.append("")

        # Usage Patterns
        if result.usage_patterns:
            summary_parts.append("## 💡 **Usage Patterns**")
            usage_counts = Counter(p.usage_type for p in result.usage_patterns)
            for usage_type, count in usage_counts.most_common(5):
                summary_parts.append(f"- **{usage_type.replace('_', ' ').title()}**: {count} examples")

            if detail_level in ["detailed", "comprehensive"] and result.usage_patterns:
                summary_parts.append("")
                summary_parts.append("### Example Usage:")
                summary_parts.append(f"```{result.primary_component.language.lower() if result.primary_component else ''}")
                summary_parts.append(result.usage_patterns[0].example_usage)
                summary_parts.append("```")
            summary_parts.append("")

        # Dependencies (for detailed/comprehensive)
        if detail_level in ["detailed", "comprehensive"] and result.dependency_analysis:
            deps = result.dependency_analysis
            summary_parts.append("## 🔗 **Dependencies**")
            if deps.imports_from:
                summary_parts.append("### Imports:")
                for imp in deps.imports_from[:5]:
                    summary_parts.append(f"- `{imp}`")

            if deps.calls_to:
                summary_parts.append("### Function Calls:")
                for call in deps.calls_to[:5]:
                    summary_parts.append(f"- `{call}()`")
            summary_parts.append("")

        # Quality Metrics (for comprehensive)
        if detail_level == "comprehensive":
            summary_parts.append("## 📊 **Quality Metrics**")
            summary_parts.append(f"- **Complexity**: {result.complexity_assessment}")
            summary_parts.append(f"- **Maintainability**: {result.maintainability_score:.1f}/1.0")
            summary_parts.append(f"- **Documentation**: {result.documentation_quality}")
            if result.test_coverage_info.get("has_tests"):
                summary_parts.append(f"- **Test Coverage**: {result.test_coverage_info.get('estimated_coverage', 'unknown')}")
            summary_parts.append("")

        # Best Practices
        if result.best_practices:
            summary_parts.append("## ✅ **Best Practices Observed**")
            for practice in result.best_practices:
                summary_parts.append(f"- {practice}")
            summary_parts.append("")

        # Analysis Info
        summary_parts.append("## ⏱️ **Analysis Info**")
        summary_parts.append(f"- **Components Found**: {result.components_analyzed}")
        summary_parts.append(f"- **Duration**: {result.analysis_duration_seconds:.2f} seconds")
        summary_parts.append(f"- **Confidence**: {result.confidence_score:.1%}")
        summary_parts.append(f"- **Timestamp**: {result.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(summary_parts)
