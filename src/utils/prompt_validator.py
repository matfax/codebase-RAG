"""
Prompt Parameter Validation Framework

This module provides comprehensive validation for MCP prompt parameters,
ensuring security, type safety, and proper formatting of inputs.
"""

import re
import os
import logging
from typing import Any, Dict, List, Optional, Union, Callable, Type
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from models.prompt_context import UserRole, TaskType, DifficultyLevel


logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Result of parameter validation."""
    valid: bool
    value: Any = None
    errors: List[str] = None
    warnings: List[str] = None
    info: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.info is None:
            self.info = []
    
    def add_issue(self, severity: ValidationSeverity, message: str):
        """Add a validation issue."""
        if severity == ValidationSeverity.ERROR:
            self.errors.append(message)
            self.valid = False
        elif severity == ValidationSeverity.WARNING:
            self.warnings.append(message)
        elif severity == ValidationSeverity.INFO:
            self.info.append(message)
    
    def has_issues(self) -> bool:
        """Check if there are any validation issues."""
        return bool(self.errors or self.warnings)
    
    def get_all_messages(self) -> List[str]:
        """Get all validation messages."""
        messages = []
        for error in self.errors:
            messages.append(f"ERROR: {error}")
        for warning in self.warnings:
            messages.append(f"WARNING: {warning}")
        for info in self.info:
            messages.append(f"INFO: {info}")
        return messages


class PromptValidator:
    """
    Comprehensive parameter validator for MCP prompts.
    
    Provides type validation, security checks, and business logic
    validation for all prompt parameters.
    """
    
    # Security patterns to block
    DANGEROUS_PATTERNS = [
        r'\.\./',  # Path traversal
        r'~/\.',   # Hidden file access
        r'/etc/',  # System file access
        r'/root/', # Root directory access
        r'rm\s+', r'del\s+',  # Deletion commands
        r'sudo\s+', r'su\s+',  # Privilege escalation
        r'eval\s*\(',  # Code evaluation
        r'exec\s*\(',  # Code execution
        r'import\s+os',  # OS module import
        r'__import__',  # Dynamic imports
        r'subprocess',  # Process execution
        r'shell=True',  # Shell execution
    ]
    
    # Valid file extensions for directory scanning
    VALID_EXTENSIONS = {
        'code': ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', '.cpp', '.c', '.h'],
        'config': ['.json', '.yaml', '.yml', '.toml', '.ini', '.env', '.xml'],
        'docs': ['.md', '.rst', '.txt', '.doc', '.docx'],
        'all': []  # Empty means all extensions allowed
    }
    
    def __init__(self):
        self.logger = logger
    
    def validate_directory_path(self, path: str, must_exist: bool = True) -> ValidationResult:
        """
        Validate directory path parameter.
        
        Args:
            path: Directory path to validate
            must_exist: Whether the directory must exist
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(valid=True, value=path)
        
        if not path:
            result.add_issue(ValidationSeverity.ERROR, "Directory path cannot be empty")
            return result
        
        if not isinstance(path, str):
            result.add_issue(ValidationSeverity.ERROR, "Directory path must be a string")
            return result
        
        # Security checks
        if self._contains_dangerous_patterns(path):
            result.add_issue(ValidationSeverity.ERROR, "Directory path contains potentially dangerous patterns")
            return result
        
        # Path normalization and validation
        try:
            normalized_path = Path(path).resolve()
            result.value = str(normalized_path)
        except Exception as e:
            result.add_issue(ValidationSeverity.ERROR, f"Invalid directory path format: {e}")
            return result
        
        # Existence check
        if must_exist and not normalized_path.exists():
            result.add_issue(ValidationSeverity.ERROR, f"Directory does not exist: {normalized_path}")
            return result
        
        if must_exist and not normalized_path.is_dir():
            result.add_issue(ValidationSeverity.ERROR, f"Path is not a directory: {normalized_path}")
            return result
        
        # Size and depth warnings
        if must_exist:
            try:
                # Quick check for very large directories
                item_count = sum(1 for _ in normalized_path.iterdir())
                if item_count > 10000:
                    result.add_issue(ValidationSeverity.WARNING, 
                                   f"Directory contains many items ({item_count}), processing may be slow")
                
                # Check depth
                depth = len(normalized_path.parts)
                if depth > 10:
                    result.add_issue(ValidationSeverity.WARNING, 
                                   f"Directory path is very deep (depth {depth})")
                    
            except PermissionError:
                result.add_issue(ValidationSeverity.WARNING, "Permission denied accessing directory")
            except Exception as e:
                result.add_issue(ValidationSeverity.INFO, f"Could not analyze directory: {e}")
        
        return result
    
    def validate_component_name(self, name: str) -> ValidationResult:
        """
        Validate component name parameter.
        
        Args:
            name: Component name to validate
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(valid=True, value=name)
        
        if not name:
            result.add_issue(ValidationSeverity.ERROR, "Component name cannot be empty")
            return result
        
        if not isinstance(name, str):
            result.add_issue(ValidationSeverity.ERROR, "Component name must be a string")
            return result
        
        # Security checks
        if self._contains_dangerous_patterns(name):
            result.add_issue(ValidationSeverity.ERROR, "Component name contains potentially dangerous patterns")
            return result
        
        # Length checks
        if len(name) > 200:
            result.add_issue(ValidationSeverity.WARNING, "Component name is very long, may not match effectively")
        
        if len(name) < 2:
            result.add_issue(ValidationSeverity.WARNING, "Component name is very short, may match too broadly")
        
        # Format validation
        name_clean = name.strip()
        result.value = name_clean
        
        # Check for valid identifier patterns
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name_clean):
            result.add_issue(ValidationSeverity.INFO, "Component name appears to be a valid identifier")
        elif re.match(r'^[a-zA-Z0-9_\-\.\/]+$', name_clean):
            result.add_issue(ValidationSeverity.INFO, "Component name appears to be a file/module path")
        else:
            result.add_issue(ValidationSeverity.INFO, "Component name contains special characters")
        
        return result
    
    def validate_component_type(self, component_type: str) -> ValidationResult:
        """
        Validate component type parameter.
        
        Args:
            component_type: Type of component
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(valid=True, value=component_type)
        
        valid_types = ["auto", "class", "function", "method", "module", "file", "variable", "constant"]
        
        if not component_type:
            result.value = "auto"
            result.add_issue(ValidationSeverity.INFO, "Component type defaulted to 'auto'")
            return result
        
        if not isinstance(component_type, str):
            result.add_issue(ValidationSeverity.ERROR, "Component type must be a string")
            return result
        
        component_type_clean = component_type.lower().strip()
        result.value = component_type_clean
        
        if component_type_clean not in valid_types:
            result.add_issue(ValidationSeverity.WARNING, 
                           f"Unknown component type '{component_type_clean}', using 'auto'. "
                           f"Valid types: {', '.join(valid_types)}")
            result.value = "auto"
        
        return result
    
    def validate_detail_level(self, detail_level: str) -> ValidationResult:
        """
        Validate detail level parameter.
        
        Args:
            detail_level: Level of detail requested
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(valid=True, value=detail_level)
        
        valid_levels = ["overview", "detailed", "comprehensive", "summary", "full"]
        
        if not detail_level:
            result.value = "overview"
            result.add_issue(ValidationSeverity.INFO, "Detail level defaulted to 'overview'")
            return result
        
        if not isinstance(detail_level, str):
            result.add_issue(ValidationSeverity.ERROR, "Detail level must be a string")
            return result
        
        detail_level_clean = detail_level.lower().strip()
        result.value = detail_level_clean
        
        if detail_level_clean not in valid_levels:
            result.add_issue(ValidationSeverity.WARNING,
                           f"Unknown detail level '{detail_level_clean}', using 'overview'. "
                           f"Valid levels: {', '.join(valid_levels)}")
            result.value = "overview"
        
        return result
    
    def validate_user_role(self, user_role: str) -> ValidationResult:
        """
        Validate user role parameter.
        
        Args:
            user_role: User role string
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(valid=True, value=user_role)
        
        if not user_role:
            result.value = "developer"
            result.add_issue(ValidationSeverity.INFO, "User role defaulted to 'developer'")
            return result
        
        if not isinstance(user_role, str):
            result.add_issue(ValidationSeverity.ERROR, "User role must be a string")
            return result
        
        user_role_clean = user_role.lower().strip()
        
        # Map to valid UserRole enum values
        valid_roles = {role.value for role in UserRole}
        
        if user_role_clean in valid_roles:
            result.value = user_role_clean
        else:
            result.add_issue(ValidationSeverity.WARNING,
                           f"Unknown user role '{user_role_clean}', using 'developer'. "
                           f"Valid roles: {', '.join(valid_roles)}")
            result.value = "developer"
        
        return result
    
    def validate_task_type(self, task_type: str) -> ValidationResult:
        """
        Validate task type parameter.
        
        Args:
            task_type: Task type string
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(valid=True, value=task_type)
        
        if not task_type:
            result.value = "exploration"
            result.add_issue(ValidationSeverity.INFO, "Task type defaulted to 'exploration'")
            return result
        
        if not isinstance(task_type, str):
            result.add_issue(ValidationSeverity.ERROR, "Task type must be a string")
            return result
        
        task_type_clean = task_type.lower().strip()
        
        # Map to valid TaskType enum values
        valid_types = {task.value for task in TaskType}
        
        if task_type_clean in valid_types:
            result.value = task_type_clean
        else:
            result.add_issue(ValidationSeverity.WARNING,
                           f"Unknown task type '{task_type_clean}', using 'exploration'. "
                           f"Valid types: {', '.join(valid_types)}")
            result.value = "exploration"
        
        return result
    
    def validate_search_queries(self, queries: Union[str, List[str]]) -> ValidationResult:
        """
        Validate search queries parameter.
        
        Args:
            queries: Search query or list of queries
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(valid=True)
        
        if not queries:
            result.add_issue(ValidationSeverity.ERROR, "Search queries cannot be empty")
            return result
        
        # Normalize to list
        if isinstance(queries, str):
            query_list = [queries]
        elif isinstance(queries, list):
            query_list = queries
        else:
            result.add_issue(ValidationSeverity.ERROR, "Search queries must be string or list of strings")
            return result
        
        validated_queries = []
        
        for i, query in enumerate(query_list):
            if not isinstance(query, str):
                result.add_issue(ValidationSeverity.WARNING, f"Query {i+1} is not a string, skipping")
                continue
            
            query_clean = query.strip()
            
            if not query_clean:
                result.add_issue(ValidationSeverity.WARNING, f"Query {i+1} is empty, skipping")
                continue
            
            if self._contains_dangerous_patterns(query_clean):
                result.add_issue(ValidationSeverity.WARNING, f"Query {i+1} contains potentially dangerous patterns, skipping")
                continue
            
            if len(query_clean) > 500:
                result.add_issue(ValidationSeverity.WARNING, f"Query {i+1} is very long, may not search effectively")
                query_clean = query_clean[:500]
            
            validated_queries.append(query_clean)
        
        if not validated_queries:
            result.add_issue(ValidationSeverity.ERROR, "No valid search queries found")
            return result
        
        result.value = validated_queries
        return result
    
    def validate_boolean_param(self, param: Any, param_name: str, default: bool = True) -> ValidationResult:
        """
        Validate boolean parameter.
        
        Args:
            param: Parameter value to validate
            param_name: Name of parameter for error messages
            default: Default value if invalid
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(valid=True)
        
        if param is None:
            result.value = default
            result.add_issue(ValidationSeverity.INFO, f"{param_name} defaulted to {default}")
            return result
        
        if isinstance(param, bool):
            result.value = param
            return result
        
        if isinstance(param, str):
            lower_param = param.lower().strip()
            if lower_param in ['true', 'yes', '1', 'on', 'enable', 'enabled']:
                result.value = True
                return result
            elif lower_param in ['false', 'no', '0', 'off', 'disable', 'disabled']:
                result.value = False
                return result
        
        result.add_issue(ValidationSeverity.WARNING, 
                        f"Invalid boolean value for {param_name}: '{param}', using default {default}")
        result.value = default
        return result
    
    def validate_integer_param(self, param: Any, param_name: str, min_val: int = 1, 
                             max_val: int = 100, default: int = 5) -> ValidationResult:
        """
        Validate integer parameter with range checking.
        
        Args:
            param: Parameter value to validate
            param_name: Name of parameter for error messages
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            default: Default value if invalid
            
        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(valid=True)
        
        if param is None:
            result.value = default
            result.add_issue(ValidationSeverity.INFO, f"{param_name} defaulted to {default}")
            return result
        
        try:
            int_val = int(param)
        except (ValueError, TypeError):
            result.add_issue(ValidationSeverity.WARNING,
                           f"Invalid integer value for {param_name}: '{param}', using default {default}")
            result.value = default
            return result
        
        if int_val < min_val:
            result.add_issue(ValidationSeverity.WARNING,
                           f"{param_name} value {int_val} below minimum {min_val}, using {min_val}")
            result.value = min_val
        elif int_val > max_val:
            result.add_issue(ValidationSeverity.WARNING,
                           f"{param_name} value {int_val} above maximum {max_val}, using {max_val}")
            result.value = max_val
        else:
            result.value = int_val
        
        return result
    
    def _contains_dangerous_patterns(self, text: str) -> bool:
        """Check if text contains potentially dangerous patterns."""
        text_lower = text.lower()
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, text_lower):
                self.logger.warning(f"Dangerous pattern detected: {pattern} in text: {text[:100]}")
                return True
        return False
    
    def validate_prompt_parameters(self, prompt_name: str, **kwargs) -> Dict[str, ValidationResult]:
        """
        Validate all parameters for a specific prompt.
        
        Args:
            prompt_name: Name of the prompt being validated
            **kwargs: Parameters to validate
            
        Returns:
            Dictionary mapping parameter names to validation results
        """
        results = {}
        
        self.logger.debug(f"Validating parameters for prompt: {prompt_name}")
        
        # Common parameter validations
        if 'directory' in kwargs:
            results['directory'] = self.validate_directory_path(kwargs['directory'])
        
        if 'component_name' in kwargs:
            results['component_name'] = self.validate_component_name(kwargs['component_name'])
        
        if 'component_type' in kwargs:
            results['component_type'] = self.validate_component_type(kwargs['component_type'])
        
        if 'detail_level' in kwargs:
            results['detail_level'] = self.validate_detail_level(kwargs['detail_level'])
        
        if 'user_role' in kwargs:
            results['user_role'] = self.validate_user_role(kwargs['user_role'])
        
        if 'task_type' in kwargs:
            results['task_type'] = self.validate_task_type(kwargs['task_type'])
        
        if 'previous_searches' in kwargs:
            results['previous_searches'] = self.validate_search_queries(kwargs['previous_searches'])
        
        # Boolean parameters
        for bool_param in ['include_dependencies', 'include_examples', 'include_usage_examples', 
                          'include_config', 'include_data_flow', 'learning_path', 'refine_strategy', 
                          'suggest_alternatives']:
            if bool_param in kwargs:
                results[bool_param] = self.validate_boolean_param(kwargs[bool_param], bool_param)
        
        # Integer parameters
        if 'context_chunks' in kwargs:
            results['context_chunks'] = self.validate_integer_param(
                kwargs['context_chunks'], 'context_chunks', min_val=0, max_val=10, default=1)
        
        # Log validation summary
        total_params = len(results)
        valid_params = sum(1 for r in results.values() if r.valid)
        issues_count = sum(len(r.errors) + len(r.warnings) for r in results.values())
        
        self.logger.info(f"Validation complete for {prompt_name}: "
                        f"{valid_params}/{total_params} valid parameters, "
                        f"{issues_count} issues found")
        
        return results
    
    def get_validation_summary(self, results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """
        Generate a summary of validation results.
        
        Args:
            results: Dictionary of validation results
            
        Returns:
            Summary dictionary with statistics and issues
        """
        summary = {
            'total_parameters': len(results),
            'valid_parameters': sum(1 for r in results.values() if r.valid),
            'parameters_with_issues': sum(1 for r in results.values() if r.has_issues()),
            'total_errors': sum(len(r.errors) for r in results.values()),
            'total_warnings': sum(len(r.warnings) for r in results.values()),
            'total_info': sum(len(r.info) for r in results.values()),
            'all_valid': all(r.valid for r in results.values()),
            'issues': []
        }
        
        # Collect all issues
        for param_name, result in results.items():
            for error in result.errors:
                summary['issues'].append({'param': param_name, 'severity': 'error', 'message': error})
            for warning in result.warnings:
                summary['issues'].append({'param': param_name, 'severity': 'warning', 'message': warning})
            for info in result.info:
                summary['issues'].append({'param': param_name, 'severity': 'info', 'message': info})
        
        return summary


# Global validator instance
prompt_validator = PromptValidator()