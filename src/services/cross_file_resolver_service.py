"""
Cross-File Function Resolution Service.

This service handles the resolution of function calls that span across multiple files
by leveraging the existing project indexing infrastructure to find function definitions
in different modules, packages, and files.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from src.models.function_call import FunctionCall
from src.services.breadcrumb_resolver_service import BreadcrumbCandidate, BreadcrumbResolutionResult
from src.tools.indexing.search_tools import search_async_cached


@dataclass
class CrossFileResolutionContext:
    """Context information for cross-file function resolution."""

    source_file_path: str
    source_module: str
    import_statements: list[str]
    package_structure: dict[str, str]
    project_name: str


@dataclass
class ImportInfo:
    """Information about an import statement."""

    module_name: str
    imported_names: list[str]
    alias: str | None
    is_from_import: bool
    line_number: int


class CrossFileResolver:
    """
    Service for resolving function calls across file boundaries.

    This service specializes in finding function definitions that exist in different
    files from where they are called, using import analysis and project indexing.
    """

    def __init__(self):
        """Initialize the cross-file resolver."""
        self.logger = logging.getLogger(__name__)

        # Cache for import analysis results
        self._import_cache: dict[str, list[ImportInfo]] = {}
        self._module_structure_cache: dict[str, dict[str, str]] = {}

        # Configuration
        self.max_search_depth = 3  # Maximum levels to search in module hierarchy
        self.enable_fuzzy_module_matching = True
        self.cross_project_search = False  # Whether to search across projects

        self.logger.info("CrossFileResolver initialized")

    async def resolve_cross_file_call(
        self, function_call: FunctionCall, resolution_context: CrossFileResolutionContext, target_projects: list[str] | None = None
    ) -> BreadcrumbResolutionResult:
        """
        Resolve a function call that may be defined in a different file.

        Args:
            function_call: FunctionCall object to resolve
            resolution_context: Context information about the source file
            target_projects: Optional list of projects to search in

        Returns:
            BreadcrumbResolutionResult with cross-file resolution details
        """
        start_time = time.time()

        try:
            # Analyze the call to determine if it's likely cross-file
            if not self._is_likely_cross_file_call(function_call):
                return BreadcrumbResolutionResult(
                    query=function_call.call_expression,
                    success=False,
                    error_message="Call does not appear to be cross-file",
                    resolution_time_ms=(time.time() - start_time) * 1000,
                )

            # Extract import information from the source file
            imports = await self._extract_imports_from_file(resolution_context.source_file_path)

            # Determine possible target modules based on the call expression
            target_modules = self._determine_target_modules(function_call, imports, resolution_context)

            if not target_modules:
                return BreadcrumbResolutionResult(
                    query=function_call.call_expression,
                    success=False,
                    error_message="No target modules identified for cross-file call",
                    resolution_time_ms=(time.time() - start_time) * 1000,
                )

            # Search for function definitions in the target modules
            candidates = await self._search_in_target_modules(function_call, target_modules, resolution_context, target_projects)

            if not candidates:
                return BreadcrumbResolutionResult(
                    query=function_call.call_expression,
                    success=False,
                    error_message="No function definitions found in target modules",
                    resolution_time_ms=(time.time() - start_time) * 1000,
                )

            # Rank candidates by cross-file resolution confidence
            ranked_candidates = self._rank_cross_file_candidates(candidates, function_call, resolution_context)

            # Return the best result
            primary_candidate = ranked_candidates[0]
            alternative_candidates = ranked_candidates[1:5]  # Top 5 alternatives

            return BreadcrumbResolutionResult(
                query=function_call.call_expression,
                success=True,
                primary_candidate=primary_candidate,
                alternative_candidates=alternative_candidates,
                resolution_time_ms=(time.time() - start_time) * 1000,
                search_results_count=len(candidates),
            )

        except Exception as e:
            self.logger.error(f"Error in cross-file resolution: {e}")
            return BreadcrumbResolutionResult(
                query=function_call.call_expression,
                success=False,
                error_message=f"Cross-file resolution error: {str(e)}",
                resolution_time_ms=(time.time() - start_time) * 1000,
            )

    async def batch_resolve_cross_file_calls(
        self, function_calls: list[FunctionCall], resolution_context: CrossFileResolutionContext, target_projects: list[str] | None = None
    ) -> dict[str, BreadcrumbResolutionResult]:
        """
        Resolve multiple cross-file function calls efficiently.

        Args:
            function_calls: List of function calls to resolve
            resolution_context: Context information about the source file
            target_projects: Optional list of projects to search in

        Returns:
            Dictionary mapping call expressions to resolution results
        """
        results = {}

        # Pre-load import information once for all calls
        imports = await self._extract_imports_from_file(resolution_context.source_file_path)

        for function_call in function_calls:
            try:
                # Create a modified context with pre-loaded imports
                enhanced_context = CrossFileResolutionContext(
                    source_file_path=resolution_context.source_file_path,
                    source_module=resolution_context.source_module,
                    import_statements=resolution_context.import_statements,
                    package_structure=resolution_context.package_structure,
                    project_name=resolution_context.project_name,
                )

                result = await self.resolve_cross_file_call(function_call, enhanced_context, target_projects)
                results[function_call.call_expression] = result

            except Exception as e:
                self.logger.error(f"Error resolving cross-file call {function_call.call_expression}: {e}")
                results[function_call.call_expression] = BreadcrumbResolutionResult(
                    query=function_call.call_expression, success=False, error_message=f"Batch resolution failed: {str(e)}"
                )

        return results

    def _is_likely_cross_file_call(self, function_call: FunctionCall) -> bool:
        """
        Determine if a function call is likely to be cross-file.

        Args:
            function_call: FunctionCall to analyze

        Returns:
            True if the call is likely cross-file
        """
        call_expr = function_call.call_expression
        call_type = function_call.call_type

        # Direct indicators of cross-file calls
        if call_type.value == "module_function":
            return True

        # Calls with module prefixes (e.g., utils.helper(), os.path.join())
        if "." in call_expr and not call_expr.startswith("self."):
            # Check if it looks like a module call
            parts = call_expr.split("(")[0].split(".")
            if len(parts) >= 2:
                # If first part doesn't look like a variable (lowercase module name)
                first_part = parts[0]
                if first_part.islower() and "_" not in first_part:
                    return True

        # Calls to common library functions
        common_modules = [
            "os",
            "sys",
            "json",
            "re",
            "time",
            "datetime",
            "math",
            "random",
            "asyncio",
            "logging",
            "pathlib",
            "collections",
            "itertools",
            "functools",
            "operator",
            "typing",
            "dataclasses",
        ]

        for module in common_modules:
            if call_expr.startswith(f"{module}."):
                return True

        # Import-based calls (this would require import analysis)
        # For now, use heuristics
        if any(keyword in call_expr for keyword in ["utils.", "helpers.", "service.", "manager."]):
            return True

        return False

    async def _extract_imports_from_file(self, file_path: str) -> list[ImportInfo]:
        """
        Extract import statements from a Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            List of ImportInfo objects
        """
        # Check cache first
        if file_path in self._import_cache:
            return self._import_cache[file_path]

        try:
            # Search for import statements in the file
            import_query = "import OR from OR import as"
            search_results = await search_async_cached(
                query=import_query,
                n_results=20,
                search_mode="keyword",
                include_context=True,
                # Note: We would ideally search only in the specific file
                # For now, we'll use a broader search and filter
            )

            imports = []
            if search_results.get("results"):
                for result in search_results["results"]:
                    # Filter for the specific file
                    if result.get("file_path") == file_path:
                        import_info = self._parse_import_from_result(result)
                        if import_info:
                            imports.append(import_info)

            # Cache the results
            self._import_cache[file_path] = imports
            return imports

        except Exception as e:
            self.logger.debug(f"Error extracting imports from {file_path}: {e}")
            return []

    def _parse_import_from_result(self, search_result: dict[str, Any]) -> ImportInfo | None:
        """
        Parse import information from a search result.

        Args:
            search_result: Search result containing import statement

        Returns:
            ImportInfo object or None if parsing fails
        """
        try:
            content = search_result.get("content", "").strip()
            line_number = search_result.get("line_start", 0)

            # Simple import parsing (could be enhanced with AST parsing)
            if content.startswith("import "):
                # import module [as alias]
                parts = content[7:].strip().split(" as ")
                module_name = parts[0].strip()
                alias = parts[1].strip() if len(parts) > 1 else None

                return ImportInfo(
                    module_name=module_name, imported_names=[module_name], alias=alias, is_from_import=False, line_number=line_number
                )

            elif content.startswith("from "):
                # from module import name1, name2 [as alias]
                parts = content[5:].split(" import ")
                if len(parts) == 2:
                    module_name = parts[0].strip()
                    imports_part = parts[1].strip()

                    # Handle "as" aliases (simplified)
                    imported_names = []
                    for name in imports_part.split(","):
                        name = name.strip()
                        if " as " in name:
                            imported_names.append(name.split(" as ")[1].strip())
                        else:
                            imported_names.append(name)

                    return ImportInfo(
                        module_name=module_name, imported_names=imported_names, alias=None, is_from_import=True, line_number=line_number
                    )

        except Exception as e:
            self.logger.debug(f"Error parsing import statement: {e}")

        return None

    def _determine_target_modules(
        self, function_call: FunctionCall, imports: list[ImportInfo], context: CrossFileResolutionContext
    ) -> list[str]:
        """
        Determine possible target modules for a function call.

        Args:
            function_call: FunctionCall to analyze
            imports: Import statements from the source file
            context: Resolution context

        Returns:
            List of possible module names where the function might be defined
        """
        call_expr = function_call.call_expression
        target_modules = []

        # Extract the module/object part of the call
        if "(" in call_expr:
            func_part = call_expr.split("(")[0].strip()
        else:
            func_part = call_expr.strip()

        # For module.function() calls
        if "." in func_part:
            parts = func_part.split(".")
            potential_module = parts[0]

            # Check if the potential module matches an import
            for import_info in imports:
                if import_info.alias and import_info.alias == potential_module:
                    # This is an aliased import
                    target_modules.append(import_info.module_name)
                elif potential_module in import_info.imported_names:
                    # This is a direct import
                    if import_info.is_from_import:
                        target_modules.append(import_info.module_name)
                    else:
                        target_modules.append(potential_module)
                elif import_info.module_name == potential_module:
                    # Direct module import
                    target_modules.append(potential_module)

        # Add common module patterns
        common_patterns = self._get_common_module_patterns(func_part)
        target_modules.extend(common_patterns)

        # Add relative module search within the same package
        if context.source_module:
            package_modules = self._get_package_relative_modules(context)
            target_modules.extend(package_modules)

        # Remove duplicates and return
        return list(set(target_modules))

    def _get_common_module_patterns(self, func_part: str) -> list[str]:
        """
        Get common module patterns based on function call patterns.

        Args:
            func_part: Function part of the call expression

        Returns:
            List of likely module names
        """
        patterns = []

        # Standard library modules
        stdlib_modules = [
            "os",
            "sys",
            "json",
            "re",
            "time",
            "datetime",
            "math",
            "random",
            "asyncio",
            "logging",
            "pathlib",
            "collections",
            "itertools",
            "functools",
            "operator",
            "typing",
            "dataclasses",
            "urllib",
            "requests",
            "sqlite3",
            "subprocess",
            "threading",
            "multiprocessing",
        ]

        for module in stdlib_modules:
            if func_part.startswith(f"{module}."):
                patterns.append(module)

        # Common third-party patterns
        if func_part.startswith("np."):
            patterns.append("numpy")
        elif func_part.startswith("pd."):
            patterns.append("pandas")
        elif func_part.startswith("plt."):
            patterns.append("matplotlib.pyplot")

        # Project-specific patterns
        if any(keyword in func_part for keyword in ["utils.", "helpers.", "service.", "manager."]):
            # Try to extract the module name
            if "." in func_part:
                potential_module = func_part.split(".")[0]
                patterns.append(potential_module)

        return patterns

    def _get_package_relative_modules(self, context: CrossFileResolutionContext) -> list[str]:
        """
        Get modules that are likely to be in the same package.

        Args:
            context: Resolution context

        Returns:
            List of relative module names
        """
        relative_modules = []

        if context.source_module:
            # Get the package path
            module_parts = context.source_module.split(".")

            # Add sibling modules
            for i in range(len(module_parts)):
                package_path = ".".join(module_parts[: i + 1])
                relative_modules.extend(
                    [
                        f"{package_path}.utils",
                        f"{package_path}.helpers",
                        f"{package_path}.services",
                        f"{package_path}.models",
                        f"{package_path}.core",
                    ]
                )

        return relative_modules

    async def _search_in_target_modules(
        self, function_call: FunctionCall, target_modules: list[str], context: CrossFileResolutionContext, target_projects: list[str] | None
    ) -> list[BreadcrumbCandidate]:
        """
        Search for function definitions in target modules.

        Args:
            function_call: FunctionCall to search for
            target_modules: List of modules to search in
            context: Resolution context
            target_projects: Optional list of projects to search in

        Returns:
            List of BreadcrumbCandidate objects
        """
        candidates = []

        # Extract the function name from the call
        call_expr = function_call.call_expression
        if "(" in call_expr:
            func_part = call_expr.split("(")[0].strip()
        else:
            func_part = call_expr.strip()

        # Get the target function name
        if "." in func_part:
            target_function = func_part.split(".")[-1]
        else:
            target_function = func_part

        # Search in each target module
        for module in target_modules:
            try:
                # Create module-specific search query
                search_query = f"{module} {target_function} OR def {target_function}"

                search_results = await search_async_cached(
                    query=search_query,
                    n_results=10,
                    search_mode="hybrid",
                    include_context=True,
                    context_chunks=1,
                    target_projects=target_projects,
                )

                if search_results.get("results"):
                    for result in search_results["results"]:
                        candidate = self._create_cross_file_candidate(result, function_call, module, target_function)
                        if candidate:
                            candidates.append(candidate)

            except Exception as e:
                self.logger.debug(f"Error searching in module {module}: {e}")
                continue

        return candidates

    def _create_cross_file_candidate(
        self, search_result: dict[str, Any], function_call: FunctionCall, target_module: str, target_function: str
    ) -> BreadcrumbCandidate | None:
        """
        Create a BreadcrumbCandidate from a cross-file search result.

        Args:
            search_result: Search result from module search
            function_call: Original function call
            target_module: Target module being searched
            target_function: Target function name

        Returns:
            BreadcrumbCandidate or None if creation fails
        """
        try:
            # Only consider function/method definitions
            chunk_type = search_result.get("chunk_type", "")
            if chunk_type not in ["function", "method", "class"]:
                return None

            # Check if the result is actually from the target module
            file_path = search_result.get("file_path", "")
            if not self._is_from_target_module(file_path, target_module):
                return None

            # Extract breadcrumb
            breadcrumb = search_result.get("breadcrumb", "")
            result_name = search_result.get("name", "")

            # Validate function name match
            if result_name and result_name != target_function:
                # Allow partial matches for now
                if target_function.lower() not in result_name.lower():
                    return None

            # Calculate cross-file specific confidence
            confidence = self._calculate_cross_file_confidence(search_result, function_call, target_module, target_function)

            # Create the candidate
            return BreadcrumbCandidate(
                breadcrumb=breadcrumb or f"{target_module}.{target_function}",
                confidence_score=confidence,
                source_chunk=None,  # Would create CodeChunk here if needed
                reasoning=self._generate_cross_file_reasoning(search_result, target_module, target_function),
                match_type="cross_file",
                file_path=file_path,
                line_start=search_result.get("line_start", 0),
                line_end=search_result.get("line_end", 0),
                chunk_type=chunk_type,
                language=search_result.get("language", ""),
            )

        except Exception as e:
            self.logger.debug(f"Error creating cross-file candidate: {e}")
            return None

    def _is_from_target_module(self, file_path: str, target_module: str) -> bool:
        """
        Check if a file path corresponds to the target module.

        Args:
            file_path: File path from search result
            target_module: Target module name

        Returns:
            True if the file is likely from the target module
        """
        if not file_path or not target_module:
            return False

        # Convert module name to file path patterns
        module_path_patterns = [
            target_module.replace(".", "/"),
            target_module.replace(".", "\\"),
            target_module.split(".")[-1],  # Just the module name
        ]

        # Check if any pattern matches the file path
        for pattern in module_path_patterns:
            if pattern in file_path:
                return True

        return False

    def _calculate_cross_file_confidence(
        self, search_result: dict[str, Any], function_call: FunctionCall, target_module: str, target_function: str
    ) -> float:
        """
        Calculate confidence score for cross-file resolution.

        Args:
            search_result: Search result
            function_call: Original function call
            target_module: Target module
            target_function: Target function name

        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0

        # Base search relevance (30%)
        search_score = search_result.get("score", 0.0)
        score += search_score * 0.3

        # Function name matching (40%)
        result_name = search_result.get("name", "")
        if result_name:
            if result_name == target_function:
                score += 0.4
            elif target_function.lower() in result_name.lower():
                score += 0.3
            elif result_name.lower() in target_function.lower():
                score += 0.25

        # Module path matching (20%)
        file_path = search_result.get("file_path", "")
        if self._is_from_target_module(file_path, target_module):
            score += 0.2

        # Chunk type bonus (10%)
        chunk_type = search_result.get("chunk_type", "")
        if chunk_type == "function":
            score += 0.1
        elif chunk_type == "method":
            score += 0.08

        return min(score, 1.0)

    def _generate_cross_file_reasoning(self, search_result: dict[str, Any], target_module: str, target_function: str) -> str:
        """
        Generate reasoning for cross-file candidate selection.

        Args:
            search_result: Search result
            target_module: Target module
            target_function: Target function name

        Returns:
            Human-readable reasoning string
        """
        reasons = []

        result_name = search_result.get("name", "")
        if result_name == target_function:
            reasons.append(f"exact function name match '{target_function}'")
        elif result_name:
            reasons.append(f"function name '{result_name}' matches target '{target_function}'")

        file_path = search_result.get("file_path", "")
        if self._is_from_target_module(file_path, target_module):
            reasons.append(f"found in target module '{target_module}'")

        chunk_type = search_result.get("chunk_type", "")
        if chunk_type:
            reasons.append(f"is a {chunk_type} definition")

        if file_path:
            reasons.append(f"located in {file_path}")

        return "Cross-file resolution: " + ", ".join(reasons) if reasons else "Cross-file candidate"

    def _rank_cross_file_candidates(
        self, candidates: list[BreadcrumbCandidate], function_call: FunctionCall, context: CrossFileResolutionContext
    ) -> list[BreadcrumbCandidate]:
        """
        Rank cross-file candidates by resolution confidence.

        Args:
            candidates: List of candidates to rank
            function_call: Original function call
            context: Resolution context

        Returns:
            Ranked list of candidates
        """
        # Sort by confidence score with tiebreakers
        candidates.sort(
            key=lambda c: (
                c.confidence_score,
                1.0 if c.match_type == "cross_file" else 0.0,
                -len(c.file_path),  # Prefer shorter paths (closer to root)
            ),
            reverse=True,
        )

        # Filter out low-confidence candidates
        min_threshold = 0.3
        filtered = [c for c in candidates if c.confidence_score >= min_threshold]

        # Remove duplicates by breadcrumb
        seen_breadcrumbs = set()
        unique_candidates = []
        for candidate in filtered:
            if candidate.breadcrumb not in seen_breadcrumbs:
                seen_breadcrumbs.add(candidate.breadcrumb)
                unique_candidates.append(candidate)

        return unique_candidates

    def clear_caches(self):
        """Clear all internal caches."""
        self._import_cache.clear()
        self._module_structure_cache.clear()
        self.logger.info("CrossFileResolver caches cleared")

    def get_statistics(self) -> dict[str, Any]:
        """Get resolver statistics."""
        return {
            "import_cache_size": len(self._import_cache),
            "module_structure_cache_size": len(self._module_structure_cache),
            "configuration": {
                "max_search_depth": self.max_search_depth,
                "enable_fuzzy_module_matching": self.enable_fuzzy_module_matching,
                "cross_project_search": self.cross_project_search,
            },
        }
