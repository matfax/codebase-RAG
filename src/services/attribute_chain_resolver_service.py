"""
Attribute Call Chain Resolution Service.

This service handles the resolution of complex attribute call chains like
`self.progress_tracker.set_total_items()` by analyzing object relationships,
class hierarchies, and attribute assignments to build complete call breadcrumbs.
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Union

from src.models.function_call import FunctionCall
from src.services.breadcrumb_resolver_service import BreadcrumbCandidate, BreadcrumbResolutionResult
from src.tools.indexing.search_tools import search_async_cached


@dataclass
class AttributeChainInfo:
    """Information about an attribute chain call."""

    base_object: str  # e.g., "self"
    attribute_chain: list[str]  # e.g., ["progress_tracker", "set_total_items"]
    method_name: str  # Final method name
    full_chain: str  # Complete chain: "self.progress_tracker.set_total_items"
    call_arguments: str  # Arguments part of the call


@dataclass
class AttributeDefinition:
    """Information about an attribute definition."""

    attribute_name: str
    attribute_type: str
    definition_context: str  # Context where attribute is defined
    file_path: str
    line_number: int
    confidence: float


class AttributeChainResolver:
    """
    Service for resolving complex attribute call chains.

    This service analyzes attribute chains to determine the actual target
    of method calls through object attribute traversal and type resolution.
    """

    def __init__(self):
        """Initialize the attribute chain resolver."""
        self.logger = logging.getLogger(__name__)

        # Cache for attribute resolution results
        self._attribute_cache: dict[str, list[AttributeDefinition]] = {}
        self._class_hierarchy_cache: dict[str, dict[str, Any]] = {}

        # Configuration
        self.max_chain_depth = 5  # Maximum depth for attribute chain resolution
        self.enable_type_inference = True
        self.search_inheritance_hierarchy = True

        self.logger.info("AttributeChainResolver initialized")

    async def resolve_attribute_chain_call(
        self, function_call: FunctionCall, source_context: dict[str, Any], target_projects: list[str] | None = None
    ) -> BreadcrumbResolutionResult:
        """
        Resolve an attribute chain function call to its target breadcrumb.

        Args:
            function_call: FunctionCall with attribute chain
            source_context: Context about the source class/function
            target_projects: Optional list of projects to search in

        Returns:
            BreadcrumbResolutionResult with resolved attribute chain target
        """
        start_time = time.time()

        try:
            # Parse the attribute chain from the call expression
            chain_info = self._parse_attribute_chain(function_call.call_expression)
            if not chain_info:
                return BreadcrumbResolutionResult(
                    query=function_call.call_expression,
                    success=False,
                    error_message="Unable to parse attribute chain",
                    resolution_time_ms=(time.time() - start_time) * 1000,
                )

            # Resolve the attribute chain step by step
            target_breadcrumb = await self._resolve_chain_step_by_step(chain_info, source_context, target_projects)

            if not target_breadcrumb:
                return BreadcrumbResolutionResult(
                    query=function_call.call_expression,
                    success=False,
                    error_message="Could not resolve attribute chain target",
                    resolution_time_ms=(time.time() - start_time) * 1000,
                )

            # Create the result with high confidence since we resolved the full chain
            candidate = BreadcrumbCandidate(
                breadcrumb=target_breadcrumb,
                confidence_score=0.9,  # High confidence for resolved chains
                source_chunk=None,
                reasoning=f"Resolved attribute chain: {chain_info.full_chain} -> {target_breadcrumb}",
                match_type="attribute_chain",
                file_path=function_call.file_path,
                line_start=function_call.line_number,
                line_end=function_call.line_number,
                chunk_type="method",
                language="python",
            )

            return BreadcrumbResolutionResult(
                query=function_call.call_expression,
                success=True,
                primary_candidate=candidate,
                resolution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            self.logger.error(f"Error resolving attribute chain: {e}")
            return BreadcrumbResolutionResult(
                query=function_call.call_expression,
                success=False,
                error_message=f"Attribute chain resolution error: {str(e)}",
                resolution_time_ms=(time.time() - start_time) * 1000,
            )

    def _parse_attribute_chain(self, call_expression: str) -> AttributeChainInfo | None:
        """
        Parse an attribute chain call expression.

        Args:
            call_expression: Call expression like "self.progress_tracker.set_total_items(100)"

        Returns:
            AttributeChainInfo object or None if parsing fails
        """
        try:
            # Split call into function part and arguments
            if "(" in call_expression:
                func_part = call_expression.split("(", 1)[0].strip()
                args_part = call_expression.split("(", 1)[1].rstrip(")")
            else:
                func_part = call_expression.strip()
                args_part = ""

            # Split the function part by dots
            parts = func_part.split(".")
            if len(parts) < 2:
                return None  # Not an attribute chain

            # Extract components
            base_object = parts[0]
            attribute_chain = parts[1:]
            method_name = attribute_chain[-1]
            full_chain = func_part

            return AttributeChainInfo(
                base_object=base_object,
                attribute_chain=attribute_chain,
                method_name=method_name,
                full_chain=full_chain,
                call_arguments=args_part,
            )

        except Exception as e:
            self.logger.debug(f"Error parsing attribute chain: {e}")
            return None

    async def _resolve_chain_step_by_step(
        self, chain_info: AttributeChainInfo, source_context: dict[str, Any], target_projects: list[str] | None
    ) -> str | None:
        """
        Resolve an attribute chain step by step to find the final target.

        Args:
            chain_info: Parsed attribute chain information
            source_context: Context about the source
            target_projects: Projects to search in

        Returns:
            Final breadcrumb target or None if resolution fails
        """
        current_type = None
        current_breadcrumb = source_context.get("breadcrumb", "")

        # Start with the base object
        if chain_info.base_object == "self":
            # For self, start with the current class
            current_type = source_context.get("class_name")
            if current_type:
                current_breadcrumb = current_type
        else:
            # Look for the base object definition
            base_definition = await self._find_attribute_definition(chain_info.base_object, source_context, target_projects)
            if base_definition:
                current_type = base_definition.attribute_type
                current_breadcrumb = base_definition.attribute_type

        if not current_type:
            self.logger.debug(f"Could not resolve base object: {chain_info.base_object}")
            return None

        # Traverse the attribute chain
        for i, attribute in enumerate(chain_info.attribute_chain[:-1]):  # Exclude the final method
            attribute_definition = await self._find_attribute_in_type(attribute, current_type, target_projects)

            if attribute_definition:
                current_type = attribute_definition.attribute_type
                current_breadcrumb = f"{current_breadcrumb}.{attribute}"
            else:
                # Try to infer the type from common patterns
                inferred_type = self._infer_attribute_type(attribute, current_type)
                if inferred_type:
                    current_type = inferred_type
                    current_breadcrumb = f"{current_breadcrumb}.{attribute}"
                else:
                    self.logger.debug(f"Could not resolve attribute: {attribute} in type: {current_type}")
                    return None

        # Finally, resolve the method call
        method_name = chain_info.method_name
        final_breadcrumb = await self._find_method_in_type(method_name, current_type, target_projects)

        if final_breadcrumb:
            return final_breadcrumb
        else:
            # Fallback: construct breadcrumb from resolved chain
            return f"{current_breadcrumb}.{method_name}"

    async def _find_attribute_definition(
        self, attribute_name: str, source_context: dict[str, Any], target_projects: list[str] | None
    ) -> AttributeDefinition | None:
        """
        Find the definition of an attribute in the source context.

        Args:
            attribute_name: Name of the attribute to find
            source_context: Context of the source
            target_projects: Projects to search in

        Returns:
            AttributeDefinition or None if not found
        """
        try:
            # Search for attribute assignments in the current context
            search_query = f"self.{attribute_name} = OR {attribute_name} ="

            search_results = await search_async_cached(
                query=search_query, n_results=10, search_mode="keyword", include_context=True, target_projects=target_projects
            )

            if search_results.get("results"):
                for result in search_results["results"]:
                    # Check if this result is in the same class context
                    if self._is_in_same_context(result, source_context):
                        attr_def = self._extract_attribute_definition(result, attribute_name)
                        if attr_def:
                            return attr_def

            return None

        except Exception as e:
            self.logger.debug(f"Error finding attribute definition for {attribute_name}: {e}")
            return None

    async def _find_attribute_in_type(
        self, attribute_name: str, type_name: str, target_projects: list[str] | None
    ) -> AttributeDefinition | None:
        """
        Find an attribute definition within a specific type/class.

        Args:
            attribute_name: Name of the attribute
            type_name: Name of the type/class
            target_projects: Projects to search in

        Returns:
            AttributeDefinition or None if not found
        """
        try:
            # Search for the attribute within the specific class
            search_query = f"class {type_name} AND {attribute_name}"

            search_results = await search_async_cached(
                query=search_query, n_results=15, search_mode="hybrid", include_context=True, target_projects=target_projects
            )

            if search_results.get("results"):
                for result in search_results["results"]:
                    if result.get("chunk_type") in ["class", "method", "function"]:
                        attr_def = self._extract_attribute_from_class_result(result, attribute_name, type_name)
                        if attr_def:
                            return attr_def

            return None

        except Exception as e:
            self.logger.debug(f"Error finding attribute {attribute_name} in type {type_name}: {e}")
            return None

    async def _find_method_in_type(self, method_name: str, type_name: str, target_projects: list[str] | None) -> str | None:
        """
        Find a method definition within a specific type/class.

        Args:
            method_name: Name of the method
            type_name: Name of the type/class
            target_projects: Projects to search in

        Returns:
            Complete breadcrumb to the method or None if not found
        """
        try:
            # Search for the method within the specific class
            search_query = f"class {type_name} AND def {method_name}"

            search_results = await search_async_cached(
                query=search_query, n_results=10, search_mode="hybrid", include_context=True, target_projects=target_projects
            )

            if search_results.get("results"):
                for result in search_results["results"]:
                    if result.get("chunk_type") == "method" and result.get("name") == method_name:
                        # Check if this method is in the right class
                        parent_class = result.get("parent_name") or result.get("parent_class")
                        if parent_class == type_name:
                            breadcrumb = result.get("breadcrumb")
                            if breadcrumb:
                                return breadcrumb

            # Fallback: construct breadcrumb
            return f"{type_name}.{method_name}"

        except Exception as e:
            self.logger.debug(f"Error finding method {method_name} in type {type_name}: {e}")
            return f"{type_name}.{method_name}"

    def _is_in_same_context(self, search_result: dict[str, Any], source_context: dict[str, Any]) -> bool:
        """
        Check if a search result is in the same context as the source.

        Args:
            search_result: Search result to check
            source_context: Source context

        Returns:
            True if in same context
        """
        # Check file path
        result_file = search_result.get("file_path", "")
        source_file = source_context.get("file_path", "")
        if result_file == source_file:
            return True

        # Check class context
        result_class = search_result.get("parent_class") or search_result.get("parent_name")
        source_class = source_context.get("class_name")
        if result_class and source_class and result_class == source_class:
            return True

        return False

    def _extract_attribute_definition(self, search_result: dict[str, Any], attribute_name: str) -> AttributeDefinition | None:
        """
        Extract attribute definition information from a search result.

        Args:
            search_result: Search result containing attribute assignment
            attribute_name: Name of the attribute

        Returns:
            AttributeDefinition or None if extraction fails
        """
        try:
            content = search_result.get("content", "")

            # Look for attribute assignments
            patterns = [
                rf"self\.{attribute_name}\s*=\s*([^=\n]+)",
                rf"{attribute_name}\s*=\s*([^=\n]+)",
                rf"self\.{attribute_name}\s*:\s*([^=\n]+)\s*=",
            ]

            for pattern in patterns:
                match = re.search(pattern, content)
                if match:
                    assignment_value = match.group(1).strip()

                    # Try to infer the type from the assignment
                    inferred_type = self._infer_type_from_assignment(assignment_value)

                    return AttributeDefinition(
                        attribute_name=attribute_name,
                        attribute_type=inferred_type,
                        definition_context=content,
                        file_path=search_result.get("file_path", ""),
                        line_number=search_result.get("line_start", 0),
                        confidence=0.8,
                    )

            return None

        except Exception as e:
            self.logger.debug(f"Error extracting attribute definition: {e}")
            return None

    def _extract_attribute_from_class_result(
        self, search_result: dict[str, Any], attribute_name: str, type_name: str
    ) -> AttributeDefinition | None:
        """
        Extract attribute definition from a class search result.

        Args:
            search_result: Search result from class
            attribute_name: Attribute name
            type_name: Type name

        Returns:
            AttributeDefinition or None
        """
        try:
            content = search_result.get("content", "")

            # Look for attribute in class definition
            patterns = [
                rf"self\.{attribute_name}\s*=\s*([^=\n]+)",
                rf"{attribute_name}\s*:\s*([^=\n]+)",
                rf"def\s+__init__.*self\.{attribute_name}\s*=\s*([^=\n]+)",
            ]

            for pattern in patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    assignment_value = match.group(1).strip()
                    inferred_type = self._infer_type_from_assignment(assignment_value)

                    return AttributeDefinition(
                        attribute_name=attribute_name,
                        attribute_type=inferred_type,
                        definition_context=content,
                        file_path=search_result.get("file_path", ""),
                        line_number=search_result.get("line_start", 0),
                        confidence=0.7,
                    )

            return None

        except Exception as e:
            self.logger.debug(f"Error extracting attribute from class result: {e}")
            return None

    def _infer_type_from_assignment(self, assignment_value: str) -> str:
        """
        Infer the type from an assignment expression.

        Args:
            assignment_value: Right side of assignment

        Returns:
            Inferred type name
        """
        assignment_value = assignment_value.strip()

        # Direct class instantiation: SomeClass()
        class_instantiation = re.match(r"([A-Z]\w*)\s*\(", assignment_value)
        if class_instantiation:
            return class_instantiation.group(1)

        # Module.Class instantiation: module.SomeClass()
        module_class = re.match(r"(\w+)\.([A-Z]\w*)\s*\(", assignment_value)
        if module_class:
            return f"{module_class.group(1)}.{module_class.group(2)}"

        # Common patterns
        if assignment_value.startswith("[]"):
            return "list"
        elif assignment_value.startswith("{}"):
            return "dict"
        elif assignment_value.startswith('"') or assignment_value.startswith("'"):
            return "str"
        elif assignment_value.isdigit():
            return "int"
        elif re.match(r"\d+\.\d+", assignment_value):
            return "float"
        elif assignment_value in ["True", "False"]:
            return "bool"
        elif assignment_value == "None":
            return "None"

        # Service/Manager patterns
        if "service" in assignment_value.lower():
            return "Service"
        elif "manager" in assignment_value.lower():
            return "Manager"
        elif "tracker" in assignment_value.lower():
            return "Tracker"

        # Fallback to the assignment value itself
        return assignment_value.split("(")[0].strip()

    def _infer_attribute_type(self, attribute_name: str, current_type: str) -> str | None:
        """
        Infer attribute type from naming patterns.

        Args:
            attribute_name: Name of the attribute
            current_type: Current type context

        Returns:
            Inferred type or None
        """
        # Common naming patterns
        if attribute_name.endswith("_service"):
            return attribute_name.replace("_", "").replace("service", "Service")
        elif attribute_name.endswith("_manager"):
            return attribute_name.replace("_", "").replace("manager", "Manager")
        elif attribute_name.endswith("_tracker"):
            return attribute_name.replace("_", "").replace("tracker", "Tracker")
        elif attribute_name.endswith("_client"):
            return attribute_name.replace("_", "").replace("client", "Client")
        elif attribute_name.endswith("_handler"):
            return attribute_name.replace("_", "").replace("handler", "Handler")

        # Convert snake_case to PascalCase
        if "_" in attribute_name:
            words = attribute_name.split("_")
            pascal_case = "".join(word.capitalize() for word in words)
            return pascal_case

        return None

    async def resolve_multiple_attribute_chains(
        self, function_calls: list[FunctionCall], source_context: dict[str, Any], target_projects: list[str] | None = None
    ) -> dict[str, BreadcrumbResolutionResult]:
        """
        Resolve multiple attribute chain calls efficiently.

        Args:
            function_calls: List of function calls to resolve
            source_context: Source context
            target_projects: Projects to search in

        Returns:
            Dictionary mapping call expressions to results
        """
        results = {}

        for function_call in function_calls:
            try:
                result = await self.resolve_attribute_chain_call(function_call, source_context, target_projects)
                results[function_call.call_expression] = result

            except Exception as e:
                self.logger.error(f"Error resolving attribute chain {function_call.call_expression}: {e}")
                results[function_call.call_expression] = BreadcrumbResolutionResult(
                    query=function_call.call_expression, success=False, error_message=f"Batch resolution failed: {str(e)}"
                )

        return results

    def is_attribute_chain_call(self, call_expression: str) -> bool:
        """
        Determine if a call expression is an attribute chain call.

        Args:
            call_expression: Call expression to check

        Returns:
            True if it's an attribute chain call
        """
        chain_info = self._parse_attribute_chain(call_expression)
        return chain_info is not None and len(chain_info.attribute_chain) > 1

    def clear_caches(self):
        """Clear all internal caches."""
        self._attribute_cache.clear()
        self._class_hierarchy_cache.clear()
        self.logger.info("AttributeChainResolver caches cleared")

    def get_statistics(self) -> dict[str, Any]:
        """Get resolver statistics."""
        return {
            "attribute_cache_size": len(self._attribute_cache),
            "class_hierarchy_cache_size": len(self._class_hierarchy_cache),
            "configuration": {
                "max_chain_depth": self.max_chain_depth,
                "enable_type_inference": self.enable_type_inference,
                "search_inheritance_hierarchy": self.search_inheritance_hierarchy,
            },
        }
