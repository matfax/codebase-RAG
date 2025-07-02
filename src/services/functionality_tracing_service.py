"""
Functionality Tracing Service

This module provides comprehensive functionality tracing capabilities for the
trace_functionality MCP prompt, offering detailed call flow analysis and execution mapping.
"""

import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    from services.component_analysis_service import ComponentAnalysisService
except ImportError:
    ComponentAnalysisService = None


logger = logging.getLogger(__name__)


@dataclass
class CallNode:
    """Represents a node in the call chain."""

    name: str
    file_path: str
    line_number: int
    node_type: str  # function, method, class_init, api_endpoint, etc.
    signature: str = ""
    context: str = ""  # Surrounding code context

    # Call chain information
    caller: Optional["CallNode"] = None
    callees: list["CallNode"] = field(default_factory=list)
    call_depth: int = 0

    # Additional metadata
    language: str = ""
    module_path: str = ""
    confidence_score: float = 0.0
    execution_probability: float = 1.0  # Probability this path is taken


@dataclass
class DataFlowStep:
    """Represents a step in data transformation."""

    step_name: str
    input_data: str
    output_data: str
    transformation: str
    location: str

    # Flow metadata
    data_type: str = "unknown"
    validation_rules: list[str] = field(default_factory=list)
    error_handling: list[str] = field(default_factory=list)


@dataclass
class ConfigurationDependency:
    """Represents a configuration dependency."""

    config_name: str
    config_type: str  # env_var, file, database, api_config
    default_value: str | None = None
    required: bool = True
    file_path: str = ""

    # Usage information
    used_in_functions: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class FunctionalityTrace:
    """Complete trace of functionality implementation."""

    # Trace identification
    functionality_name: str
    trace_type: str
    analysis_timestamp: datetime = field(default_factory=datetime.now)

    # Entry points and triggers
    entry_points: list[CallNode] = field(default_factory=list)
    api_endpoints: list[dict[str, Any]] = field(default_factory=list)
    event_handlers: list[CallNode] = field(default_factory=list)

    # Call chain analysis
    call_tree: CallNode | None = None
    execution_paths: list[list[CallNode]] = field(default_factory=list)
    critical_path: list[CallNode] = field(default_factory=list)

    # Data flow analysis
    data_flow_steps: list[DataFlowStep] = field(default_factory=list)
    input_sources: list[str] = field(default_factory=list)
    output_destinations: list[str] = field(default_factory=list)

    # Dependencies and configuration
    configuration_deps: list[ConfigurationDependency] = field(default_factory=list)
    external_services: list[str] = field(default_factory=list)
    database_operations: list[dict[str, Any]] = field(default_factory=list)

    # Error handling and edge cases
    error_handlers: list[CallNode] = field(default_factory=list)
    validation_points: list[str] = field(default_factory=list)
    edge_cases: list[str] = field(default_factory=list)

    # Quality and complexity
    complexity_score: float = 0.0
    reliability_score: float = 0.0
    test_coverage: dict[str, Any] = field(default_factory=dict)

    # Analysis metadata
    analysis_duration_seconds: float = 0.0
    confidence_score: float = 0.0
    files_analyzed: int = 0
    functions_traced: int = 0


class FunctionalityTracingService:
    """
    Advanced functionality tracing service for comprehensive execution flow analysis.

    Provides detailed tracing of functionality from entry points through the complete
    execution chain, including data flow, configuration dependencies, and error handling.
    """

    def __init__(self):
        self.logger = logger
        self.component_analyzer = ComponentAnalysisService() if ComponentAnalysisService else None

        # Pattern databases for tracing
        self.entry_point_patterns = self._load_entry_point_patterns()
        self.api_patterns = self._load_api_patterns()
        self.data_flow_patterns = self._load_data_flow_patterns()
        self.config_patterns = self._load_config_patterns()

    def trace_functionality(
        self,
        functionality_description: str,
        project_path: str = ".",
        trace_type: str = "full_flow",
        include_config: bool = True,
        include_data_flow: bool = True,
        max_depth: int = 10,
    ) -> FunctionalityTrace:
        """
        Perform comprehensive functionality tracing.

        Args:
            functionality_description: Description of functionality to trace
            project_path: Path to search for the functionality
            trace_type: Type of trace ("full_flow", "api_to_db", "user_journey", "data_pipeline")
            include_config: Whether to analyze configuration dependencies
            include_data_flow: Whether to trace data transformation flow
            max_depth: Maximum depth for call chain traversal

        Returns:
            FunctionalityTrace with comprehensive execution analysis
        """
        start_time = time.time()

        project_path = Path(project_path).resolve()

        self.logger.info(f"Starting functionality tracing for: {functionality_description}")

        # Initialize trace result
        trace = FunctionalityTrace(functionality_name=functionality_description, trace_type=trace_type)

        try:
            # Phase 1: Identify entry points
            entry_points = self._identify_entry_points(functionality_description, project_path)
            trace.entry_points = entry_points

            # Phase 2: Build call tree and execution paths
            if entry_points:
                call_tree, execution_paths = self._build_call_tree(entry_points, project_path, max_depth)
                trace.call_tree = call_tree
                trace.execution_paths = execution_paths
                trace.critical_path = self._identify_critical_path(execution_paths)

            # Phase 3: Analyze data flow
            if include_data_flow:
                data_flow = self._analyze_data_flow(trace.execution_paths, project_path)
                trace.data_flow_steps = data_flow["steps"]
                trace.input_sources = data_flow["inputs"]
                trace.output_destinations = data_flow["outputs"]

            # Phase 4: Identify configuration dependencies
            if include_config:
                config_deps = self._analyze_configuration_dependencies(trace.execution_paths, project_path)
                trace.configuration_deps = config_deps

            # Phase 5: Find API endpoints and external integrations
            api_info = self._analyze_api_endpoints(functionality_description, project_path)
            trace.api_endpoints = api_info["endpoints"]
            trace.external_services = api_info["external_services"]

            # Phase 6: Analyze error handling and validation
            error_info = self._analyze_error_handling(trace.execution_paths, project_path)
            trace.error_handlers = error_info["handlers"]
            trace.validation_points = error_info["validation"]
            trace.edge_cases = error_info["edge_cases"]

            # Phase 7: Quality assessment
            quality_info = self._assess_trace_quality(trace, project_path)
            trace.complexity_score = quality_info["complexity"]
            trace.reliability_score = quality_info["reliability"]
            trace.test_coverage = quality_info["test_coverage"]

            # Finalize trace
            trace.analysis_duration_seconds = time.time() - start_time
            trace.confidence_score = self._calculate_trace_confidence(trace)
            trace.functions_traced = self._count_unique_functions(trace)

            self.logger.info(f"Functionality tracing completed in {trace.analysis_duration_seconds:.2f}s")
            return trace

        except Exception as e:
            self.logger.error(f"Error during functionality tracing: {e}")
            trace.analysis_duration_seconds = time.time() - start_time
            return trace

    def _identify_entry_points(self, functionality_description: str, project_path: Path) -> list[CallNode]:
        """Identify possible entry points for the functionality."""
        entry_points = []

        # Extract keywords from functionality description
        keywords = self._extract_keywords(functionality_description)

        # Search for entry points in different categories
        entry_points.extend(self._find_api_entry_points(keywords, project_path))
        entry_points.extend(self._find_function_entry_points(keywords, project_path))
        entry_points.extend(self._find_main_entry_points(keywords, project_path))
        entry_points.extend(self._find_event_handlers(keywords, project_path))

        # Score and sort entry points by relevance
        scored_points = []
        for entry in entry_points:
            relevance = self._calculate_entry_point_relevance(entry, keywords)
            entry.confidence_score = relevance
            scored_points.append(entry)

        # Return top candidates
        scored_points.sort(key=lambda x: x.confidence_score, reverse=True)
        return scored_points[:5]

    def _extract_keywords(self, description: str) -> list[str]:
        """Extract relevant keywords from functionality description."""
        # Remove common words and extract meaningful terms
        common_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        words = re.findall(r"\b\w+\b", description.lower())
        keywords = [word for word in words if word not in common_words and len(word) > 2]

        # Add some variations
        variations = []
        for keyword in keywords:
            variations.extend(
                [
                    f"{keyword}s",  # plural
                    f"{keyword}_",  # with underscore
                    f"_{keyword}",  # with underscore prefix
                    keyword.replace("_", ""),  # remove underscores
                ]
            )

        return list(set(keywords + variations))

    def _find_api_entry_points(self, keywords: list[str], project_path: Path) -> list[CallNode]:
        """Find API endpoints that might be entry points."""
        api_points = []

        # Common API patterns to look for
        api_patterns = [
            r'@app\.route\s*\(\s*[\'"]([^\'"]+)[\'"]',  # Flask
            r'@router\.\w+\s*\(\s*[\'"]([^\'"]+)[\'"]',  # FastAPI
            r'app\.\w+\s*\(\s*[\'"]([^\'"]+)[\'"]',  # Express.js
            r'@RequestMapping\s*\([^)]*[\'"]([^\'"]+)[\'"]',  # Spring
        ]

        for root, dirs, files in os.walk(project_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {"node_modules", "__pycache__", "venv", ".venv"}]

            for file in files:
                if self._is_source_file(file):
                    file_path = Path(root) / file
                    api_points.extend(self._search_file_for_api_patterns(file_path, keywords, api_patterns))

        return api_points

    def _find_function_entry_points(self, keywords: list[str], project_path: Path) -> list[CallNode]:
        """Find function definitions that might be entry points."""
        function_points = []

        for root, dirs, files in os.walk(project_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {"node_modules", "__pycache__", "venv", ".venv"}]

            for file in files:
                if self._is_source_file(file):
                    file_path = Path(root) / file
                    function_points.extend(self._search_file_for_functions(file_path, keywords))

        return function_points

    def _find_main_entry_points(self, keywords: list[str], project_path: Path) -> list[CallNode]:
        """Find main function or application entry points."""
        main_points = []

        # Look for main functions and application startup
        main_patterns = [
            r"def\s+main\s*\(",
            r'if\s+__name__\s*==\s*[\'"]__main__[\'"]',
            r"app\s*=\s*\w+\(",
            r"server\s*=\s*\w+\(",
        ]

        for root, dirs, files in os.walk(project_path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                if file in ["main.py", "app.py", "server.py", "index.js", "main.js"]:
                    file_path = Path(root) / file
                    main_points.extend(self._search_file_for_patterns(file_path, main_patterns, "main"))

        return main_points

    def _find_event_handlers(self, keywords: list[str], project_path: Path) -> list[CallNode]:
        """Find event handlers and callback functions."""
        event_points = []

        # Event handler patterns
        event_patterns = [
            r'@\w+\.on\s*\(\s*[\'"]([^\'"]+)[\'"]',  # Event decorators
            r'addEventListener\s*\(\s*[\'"]([^\'"]+)[\'"]',  # JavaScript events
            r"on\w+\s*=\s*\w+",  # Inline event handlers
        ]

        for root, dirs, files in os.walk(project_path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                if self._is_source_file(file):
                    file_path = Path(root) / file
                    event_points.extend(self._search_file_for_patterns(file_path, event_patterns, "event"))

        return event_points

    def _search_file_for_api_patterns(self, file_path: Path, keywords: list[str], patterns: list[str]) -> list[CallNode]:
        """Search a file for API endpoint patterns."""
        api_nodes = []

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                for pattern in patterns:
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        # Check if any keywords match the endpoint or nearby function
                        context_lines = lines[max(0, i - 3) : min(len(lines), i + 3)]
                        context = "\n".join(context_lines)

                        if any(keyword in context.lower() for keyword in keywords):
                            node = CallNode(
                                name=(match.group(1) if match.groups() else f"endpoint_line_{i}"),
                                file_path=str(file_path),
                                line_number=i,
                                node_type="api_endpoint",
                                signature=line.strip(),
                                context=context,
                                language=self._detect_language(file_path),
                                module_path=self._get_module_path(file_path),
                            )
                            api_nodes.append(node)

        except Exception as e:
            self.logger.debug(f"Error searching API patterns in {file_path}: {e}")

        return api_nodes

    def _search_file_for_functions(self, file_path: Path, keywords: list[str]) -> list[CallNode]:
        """Search a file for function definitions matching keywords."""
        function_nodes = []

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")

            # Function definition patterns for different languages
            function_patterns = [
                r"def\s+(\w+)\s*\(",  # Python
                r"function\s+(\w+)\s*\(",  # JavaScript
                r"const\s+(\w+)\s*=\s*\(",  # Arrow functions
                r"func\s+(\w+)\s*\(",  # Go
                r"fn\s+(\w+)\s*\(",  # Rust
            ]

            for i, line in enumerate(lines, 1):
                for pattern in function_patterns:
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        function_name = match.group(1)

                        # Check if function name or surrounding context matches keywords
                        if any(keyword in function_name.lower() for keyword in keywords):
                            # Get function context
                            context_start = max(0, i - 2)
                            context_end = min(len(lines), i + 10)
                            context = "\n".join(lines[context_start:context_end])

                            node = CallNode(
                                name=function_name,
                                file_path=str(file_path),
                                line_number=i,
                                node_type="function",
                                signature=line.strip(),
                                context=context,
                                language=self._detect_language(file_path),
                                module_path=self._get_module_path(file_path),
                            )
                            function_nodes.append(node)

        except Exception as e:
            self.logger.debug(f"Error searching functions in {file_path}: {e}")

        return function_nodes

    def _search_file_for_patterns(self, file_path: Path, patterns: list[str], node_type: str) -> list[CallNode]:
        """Search a file for general patterns."""
        nodes = []

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                for pattern in patterns:
                    if re.search(pattern, line):
                        context_start = max(0, i - 2)
                        context_end = min(len(lines), i + 5)
                        context = "\n".join(lines[context_start:context_end])

                        node = CallNode(
                            name=f"{node_type}_line_{i}",
                            file_path=str(file_path),
                            line_number=i,
                            node_type=node_type,
                            signature=line.strip(),
                            context=context,
                            language=self._detect_language(file_path),
                            module_path=self._get_module_path(file_path),
                        )
                        nodes.append(node)

        except Exception as e:
            self.logger.debug(f"Error searching patterns in {file_path}: {e}")

        return nodes

    def _calculate_entry_point_relevance(self, entry: CallNode, keywords: list[str]) -> float:
        """Calculate relevance score for an entry point."""
        score = 0.0

        # Name matching
        name_matches = sum(1 for keyword in keywords if keyword in entry.name.lower())
        score += name_matches * 0.3

        # Context matching
        context_matches = sum(1 for keyword in keywords if keyword in entry.context.lower())
        score += context_matches * 0.1

        # Entry point type bonus
        if entry.node_type == "api_endpoint":
            score += 0.3
        elif entry.node_type == "main":
            score += 0.2
        elif entry.node_type == "function":
            score += 0.1

        return min(1.0, score)

    def _build_call_tree(
        self, entry_points: list[CallNode], project_path: Path, max_depth: int
    ) -> tuple[CallNode | None, list[list[CallNode]]]:
        """Build call tree and identify execution paths."""
        if not entry_points:
            return None, []

        # Start with the highest confidence entry point
        root = entry_points[0]
        execution_paths = []

        # Build call tree using BFS
        visited = set()
        queue = deque([(root, 0)])

        while queue:
            current_node, depth = queue.popleft()

            if depth >= max_depth:
                continue

            node_key = f"{current_node.file_path}:{current_node.line_number}"
            if node_key in visited:
                continue

            visited.add(node_key)

            # Find functions called by this node
            callees = self._find_function_calls(current_node, project_path)
            current_node.callees = callees

            # Add to queue for further exploration
            for callee in callees:
                callee.caller = current_node
                callee.call_depth = depth + 1
                queue.append((callee, depth + 1))

        # Extract execution paths
        execution_paths = self._extract_execution_paths(root, max_depth)

        return root, execution_paths

    def _find_function_calls(self, node: CallNode, project_path: Path) -> list[CallNode]:
        """Find functions called by a given node."""
        callees = []

        try:
            # Get the full function/method content
            file_path = Path(node.file_path)
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")

            # Get function body (simplified approach)
            start_line = node.line_number - 1
            function_lines = self._extract_function_body(lines, start_line)
            function_content = "\n".join(function_lines)

            # Find function calls in the content
            call_patterns = [
                r"(\w+)\s*\(",  # Simple function calls
                r"self\.(\w+)\s*\(",  # Method calls
                r"(\w+)\.(\w+)\s*\(",  # Object method calls
            ]

            for pattern in call_patterns:
                matches = re.finditer(pattern, function_content)
                for match in matches:
                    function_name = match.group(1)

                    # Try to find the definition of this function
                    callee_node = self._find_function_definition(function_name, project_path, file_path)
                    if callee_node:
                        callees.append(callee_node)

        except Exception as e:
            self.logger.debug(f"Error finding function calls for {node.name}: {e}")

        return callees[:5]  # Limit to avoid explosion

    def _extract_function_body(self, lines: list[str], start_line: int) -> list[str]:
        """Extract the body of a function from lines."""
        if start_line >= len(lines):
            return []

        # Find indentation level
        start_indent = len(lines[start_line]) - len(lines[start_line].lstrip())

        function_lines = [lines[start_line]]

        for i in range(start_line + 1, min(len(lines), start_line + 50)):
            line = lines[i]
            if line.strip():  # Non-empty line
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= start_indent and not line.strip().startswith(('"""', "'''", "#")):
                    break
            function_lines.append(line)

        return function_lines

    def _find_function_definition(self, function_name: str, project_path: Path, current_file: Path) -> CallNode | None:
        """Find the definition of a function."""
        # First check in the current file
        try:
            content = current_file.read_text(encoding="utf-8", errors="ignore")
            lines = content.split("\n")

            for i, line in enumerate(lines, 1):
                if re.search(rf"def\s+{re.escape(function_name)}\s*\(", line):
                    return CallNode(
                        name=function_name,
                        file_path=str(current_file),
                        line_number=i,
                        node_type="function",
                        signature=line.strip(),
                        language=self._detect_language(current_file),
                        module_path=self._get_module_path(current_file),
                    )
        except Exception:
            pass

        # TODO: Search in other files (simplified for now)
        return None

    def _extract_execution_paths(self, root: CallNode, max_depth: int) -> list[list[CallNode]]:
        """Extract all execution paths from the call tree."""
        paths = []

        def dfs(node: CallNode, current_path: list[CallNode], depth: int):
            if depth >= max_depth:
                if current_path:
                    paths.append(current_path.copy())
                return

            current_path.append(node)

            if not node.callees:
                # Leaf node - add the complete path
                paths.append(current_path.copy())
            else:
                # Continue with each callee
                for callee in node.callees:
                    dfs(callee, current_path, depth + 1)

            current_path.pop()

        if root:
            dfs(root, [], 0)

        return paths[:10]  # Limit number of paths

    def _identify_critical_path(self, execution_paths: list[list[CallNode]]) -> list[CallNode]:
        """Identify the most critical execution path."""
        if not execution_paths:
            return []

        # For now, return the longest path as critical
        return max(execution_paths, key=len, default=[])

    def _analyze_data_flow(self, execution_paths: list[list[CallNode]], project_path: Path) -> dict[str, Any]:
        """Analyze data flow through the execution paths."""
        data_flow = {"steps": [], "inputs": [], "outputs": []}

        # Simplified data flow analysis
        for path in execution_paths[:3]:  # Analyze top 3 paths
            for i, node in enumerate(path):
                if i < len(path) - 1:
                    next_node = path[i + 1]

                    step = DataFlowStep(
                        step_name=f"{node.name} -> {next_node.name}",
                        input_data=f"output from {node.name}",
                        output_data=f"input to {next_node.name}",
                        transformation="function call",
                        location=f"{Path(node.file_path).name}:{node.line_number}",
                    )
                    data_flow["steps"].append(step)

        # Identify inputs and outputs (simplified)
        if execution_paths:
            first_path = execution_paths[0]
            if first_path:
                data_flow["inputs"].append(f"Parameters to {first_path[0].name}")
                data_flow["outputs"].append(f"Return value from {first_path[-1].name}")

        return data_flow

    def _analyze_configuration_dependencies(
        self, execution_paths: list[list[CallNode]], project_path: Path
    ) -> list[ConfigurationDependency]:
        """Analyze configuration dependencies."""
        config_deps = []

        # Look for common configuration patterns in the execution paths
        config_patterns = [
            r"os\.environ\[\'([^\']+)\'\]",  # Environment variables
            r'os\.getenv\([\'"]([^\'"]+)[\'"]',  # Environment variables
            r"config\[\'([^\']+)\'\]",  # Config dictionary access
            r"settings\.(\w+)",  # Settings attributes
        ]

        processed_files = set()

        for path in execution_paths:
            for node in path:
                if node.file_path in processed_files:
                    continue

                processed_files.add(node.file_path)

                try:
                    content = Path(node.file_path).read_text(encoding="utf-8", errors="ignore")

                    for pattern in config_patterns:
                        matches = re.finditer(pattern, content)
                        for match in matches:
                            config_name = match.group(1)

                            config_dep = ConfigurationDependency(
                                config_name=config_name,
                                config_type=("env_var" if "environ" in pattern or "getenv" in pattern else "config"),
                                file_path=node.file_path,
                                used_in_functions=[node.name],
                            )
                            config_deps.append(config_dep)

                except Exception as e:
                    self.logger.debug(f"Error analyzing config in {node.file_path}: {e}")

        # Deduplicate and limit
        unique_configs = {}
        for config in config_deps:
            if config.config_name not in unique_configs:
                unique_configs[config.config_name] = config
            else:
                unique_configs[config.config_name].used_in_functions.extend(config.used_in_functions)

        return list(unique_configs.values())[:10]

    def _analyze_api_endpoints(self, functionality_description: str, project_path: Path) -> dict[str, Any]:
        """Analyze API endpoints and external service integrations."""
        api_info = {"endpoints": [], "external_services": []}

        # This would be implemented more thoroughly in a real system
        # For now, return basic structure

        return api_info

    def _analyze_error_handling(self, execution_paths: list[list[CallNode]], project_path: Path) -> dict[str, Any]:
        """Analyze error handling and validation points."""
        error_info = {"handlers": [], "validation": [], "edge_cases": []}

        # Look for error handling patterns in execution paths
        error_patterns = [
            r"try\s*:",
            r"except\s+\w+",
            r"raise\s+\w+",
            r"if\s+.*\s+is\s+None",
            r"assert\s+",
        ]

        processed_files = set()

        for path in execution_paths:
            for node in path:
                if node.file_path in processed_files:
                    continue

                processed_files.add(node.file_path)

                try:
                    content = Path(node.file_path).read_text(encoding="utf-8", errors="ignore")
                    lines = content.split("\n")

                    for i, line in enumerate(lines, 1):
                        for pattern in error_patterns:
                            if re.search(pattern, line):
                                if "try" in pattern or "except" in pattern:
                                    error_handler = CallNode(
                                        name=f"error_handler_line_{i}",
                                        file_path=node.file_path,
                                        line_number=i,
                                        node_type="error_handler",
                                        signature=line.strip(),
                                    )
                                    error_info["handlers"].append(error_handler)
                                elif "assert" in pattern or "is None" in pattern:
                                    error_info["validation"].append(f"{Path(node.file_path).name}:{i}: {line.strip()}")

                except Exception as e:
                    self.logger.debug(f"Error analyzing error handling in {node.file_path}: {e}")

        # Limit results
        error_info["handlers"] = error_info["handlers"][:5]
        error_info["validation"] = error_info["validation"][:5]

        return error_info

    def _assess_trace_quality(self, trace: FunctionalityTrace, project_path: Path) -> dict[str, Any]:
        """Assess the quality of the traced functionality."""
        quality = {"complexity": 0.0, "reliability": 0.0, "test_coverage": {}}

        # Calculate complexity based on execution paths
        if trace.execution_paths:
            avg_path_length = sum(len(path) for path in trace.execution_paths) / len(trace.execution_paths)
            num_paths = len(trace.execution_paths)

            # Normalize complexity (0-1 scale)
            quality["complexity"] = min(1.0, (avg_path_length * num_paths) / 50.0)

        # Calculate reliability based on error handling
        total_nodes = sum(len(path) for path in trace.execution_paths)
        error_handler_count = len(trace.error_handlers)

        if total_nodes > 0:
            error_coverage = error_handler_count / total_nodes
            quality["reliability"] = min(1.0, error_coverage * 2.0)  # Scale up

        # Basic test coverage info
        quality["test_coverage"] = {
            "has_tests": False,
            "test_files": [],
            "estimated_coverage": "unknown",
        }

        return quality

    def _calculate_trace_confidence(self, trace: FunctionalityTrace) -> float:
        """Calculate overall confidence in the trace results."""
        score = 0.0
        factors = 0

        # Entry point confidence
        if trace.entry_points:
            avg_entry_confidence = sum(ep.confidence_score for ep in trace.entry_points) / len(trace.entry_points)
            score += avg_entry_confidence * 0.4
        factors += 1

        # Execution path completeness
        if trace.execution_paths:
            path_completeness = min(1.0, len(trace.execution_paths) / 5.0)
            score += path_completeness * 0.3
        factors += 1

        # Configuration analysis
        if trace.configuration_deps:
            score += 0.2
        factors += 1

        # Error handling analysis
        if trace.error_handlers or trace.validation_points:
            score += 0.1
        factors += 1

        return score / factors if factors > 0 else 0.0

    def _count_unique_functions(self, trace: FunctionalityTrace) -> int:
        """Count unique functions traced."""
        unique_functions = set()

        for path in trace.execution_paths:
            for node in path:
                unique_functions.add(f"{node.file_path}:{node.name}")

        return len(unique_functions)

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
        """Get the module path for the file."""
        # Simplified module path
        return str(file_path.stem)

    def _load_entry_point_patterns(self) -> dict[str, Any]:
        """Load entry point pattern definitions."""
        return {}

    def _load_api_patterns(self) -> dict[str, Any]:
        """Load API pattern definitions."""
        return {}

    def _load_data_flow_patterns(self) -> dict[str, Any]:
        """Load data flow pattern definitions."""
        return {}

    def _load_config_patterns(self) -> dict[str, Any]:
        """Load configuration pattern definitions."""
        return {}

    def format_trace_summary(self, trace: FunctionalityTrace, detail_level: str = "overview") -> str:
        """Format trace result into a human-readable summary."""
        summary_parts = []

        # Header
        summary_parts.append(f"# 🔍 Functionality Trace: {trace.functionality_name}")
        summary_parts.append("")

        # Basic Information
        summary_parts.append("## 📋 **Trace Overview**")
        summary_parts.append(f"- **Trace Type**: {trace.trace_type}")
        summary_parts.append(f"- **Functions Traced**: {trace.functions_traced}")
        summary_parts.append(f"- **Execution Paths**: {len(trace.execution_paths)}")
        summary_parts.append(f"- **Entry Points Found**: {len(trace.entry_points)}")
        summary_parts.append("")

        # Entry Points
        if trace.entry_points:
            summary_parts.append("## 🚀 **Entry Points**")
            for i, entry in enumerate(trace.entry_points[:3], 1):
                confidence_pct = f"{entry.confidence_score:.0%}"
                summary_parts.append(f"{i}. **{entry.name}** ({entry.node_type}) - Confidence: {confidence_pct}")
                summary_parts.append(f"   - Location: `{Path(entry.file_path).name}:{entry.line_number}`")
                summary_parts.append(f"   - Signature: `{entry.signature}`")
            summary_parts.append("")

        # Critical Path
        if trace.critical_path:
            summary_parts.append("## 🎯 **Critical Execution Path**")
            for i, node in enumerate(trace.critical_path):
                arrow = " → " if i > 0 else ""
                summary_parts.append(f"{arrow}`{node.name}` ({Path(node.file_path).name}:{node.line_number})")
            summary_parts.append("")

        # Data Flow (for detailed/comprehensive)
        if detail_level in ["detailed", "comprehensive"] and trace.data_flow_steps:
            summary_parts.append("## 📊 **Data Flow**")
            for step in trace.data_flow_steps[:5]:
                summary_parts.append(f"- **{step.step_name}**")
                summary_parts.append(f"  - Input: {step.input_data}")
                summary_parts.append(f"  - Output: {step.output_data}")
                summary_parts.append(f"  - Location: {step.location}")
            summary_parts.append("")

        # Configuration Dependencies
        if trace.configuration_deps:
            summary_parts.append("## ⚙️ **Configuration Dependencies**")
            for config in trace.configuration_deps[:5]:
                summary_parts.append(f"- **{config.config_name}** ({config.config_type})")
                if config.used_in_functions:
                    functions = ", ".join(config.used_in_functions[:3])
                    summary_parts.append(f"  - Used in: {functions}")
            summary_parts.append("")

        # Error Handling (for comprehensive)
        if detail_level == "comprehensive" and (trace.error_handlers or trace.validation_points):
            summary_parts.append("## 🛡️ **Error Handling**")
            if trace.error_handlers:
                summary_parts.append("### Exception Handlers:")
                for handler in trace.error_handlers[:3]:
                    summary_parts.append(f"- `{handler.signature}` at {Path(handler.file_path).name}:{handler.line_number}")

            if trace.validation_points:
                summary_parts.append("### Validation Points:")
                for validation in trace.validation_points[:3]:
                    summary_parts.append(f"- {validation}")
            summary_parts.append("")

        # Quality Metrics (for comprehensive)
        if detail_level == "comprehensive":
            summary_parts.append("## 📈 **Quality Assessment**")
            summary_parts.append(f"- **Complexity Score**: {trace.complexity_score:.1f}/1.0")
            summary_parts.append(f"- **Reliability Score**: {trace.reliability_score:.1f}/1.0")
            if trace.error_handlers:
                summary_parts.append(f"- **Error Handlers**: {len(trace.error_handlers)} found")
            if trace.validation_points:
                summary_parts.append(f"- **Validation Points**: {len(trace.validation_points)} found")
            summary_parts.append("")

        # Analysis Info
        summary_parts.append("## ⏱️ **Analysis Info**")
        summary_parts.append(f"- **Duration**: {trace.analysis_duration_seconds:.2f} seconds")
        summary_parts.append(f"- **Confidence**: {trace.confidence_score:.1%}")
        summary_parts.append(f"- **Files Analyzed**: {trace.files_analyzed}")
        summary_parts.append(f"- **Timestamp**: {trace.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        return "\n".join(summary_parts)
