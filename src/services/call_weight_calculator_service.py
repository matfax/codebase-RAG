"""
Service for calculating function call weights and confidence scores.

This service implements configurable weight calculation algorithms for different
types of function calls, with support for frequency factors and confidence scoring.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

from src.models.function_call import CallType, FunctionCall

logger = logging.getLogger(__name__)


@dataclass
class WeightConfiguration:
    """
    Configuration for function call weight calculation.

    This class defines the base weights for different call types and
    configuration parameters for weight calculation algorithms.
    """

    # Base weights for different call types (as specified in PRD)
    base_weights: dict[CallType, float] = field(
        default_factory=lambda: {
            CallType.DIRECT: 1.0,  # Direct function calls: function_name()
            CallType.METHOD: 0.9,  # Method calls: object.method()
            CallType.ATTRIBUTE: 0.7,  # Attribute chain calls: obj.attr.method()
            CallType.SELF_METHOD: 0.95,  # Self method calls: self.method()
            CallType.ASYNC: 0.9,  # Async function calls: await function()
            CallType.ASYNC_METHOD: 0.85,  # Async method calls: await obj.method()
            CallType.ASYNCIO: 0.8,  # Asyncio library calls
            CallType.SUPER_METHOD: 0.8,  # Super method calls: super().method()
            CallType.CLASS_METHOD: 0.75,  # Class method calls: cls.method()
            CallType.DYNAMIC: 0.5,  # Dynamic calls: getattr(obj, 'method')()
            CallType.UNPACKING: 0.7,  # Calls with unpacking: func(*args, **kwargs)
            CallType.MODULE_FUNCTION: 0.85,  # Module function calls: module.function()
            CallType.SUBSCRIPT_METHOD: 0.6,  # Subscript method calls: obj[key].method()
        }
    )

    # Frequency scaling parameters
    frequency_scale_factor: float = 0.1  # How much frequency impacts weight
    max_frequency_multiplier: float = 2.0  # Maximum frequency multiplier
    min_frequency_multiplier: float = 0.5  # Minimum frequency multiplier

    # Context-based weight adjustments
    conditional_penalty: float = 0.9  # Weight multiplier for conditional calls
    nested_penalty: float = 0.95  # Weight multiplier for nested calls
    cross_module_bonus: float = 1.1  # Weight bonus for cross-module calls
    recursive_penalty: float = 0.8  # Weight penalty for recursive calls

    # Type hint and documentation bonuses
    type_hint_bonus: float = 1.05  # Weight bonus for calls with type hints
    docstring_bonus: float = 1.03  # Weight bonus for calls with documentation

    # Syntax error penalties
    syntax_error_penalty: float = 0.5  # Weight penalty for calls with syntax errors

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate base weights are non-negative
        for call_type, weight in self.base_weights.items():
            if weight < 0.0:
                raise ValueError(f"Base weight for {call_type} must be non-negative, got {weight}")

        # Validate frequency parameters
        if self.frequency_scale_factor < 0.0:
            raise ValueError(f"Frequency scale factor must be non-negative, got {self.frequency_scale_factor}")

        if self.max_frequency_multiplier < self.min_frequency_multiplier:
            raise ValueError(
                f"Max frequency multiplier ({self.max_frequency_multiplier}) must be >= "
                f"min frequency multiplier ({self.min_frequency_multiplier})"
            )

    def get_base_weight(self, call_type: CallType) -> float:
        """Get the base weight for a specific call type."""
        return self.base_weights.get(call_type, 0.5)  # Default weight for unknown types

    def update_base_weight(self, call_type: CallType, weight: float) -> None:
        """Update the base weight for a specific call type."""
        if weight < 0.0:
            raise ValueError(f"Weight must be non-negative, got {weight}")
        self.base_weights[call_type] = weight

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for serialization."""
        return {
            "base_weights": {call_type.value: weight for call_type, weight in self.base_weights.items()},
            "frequency_scale_factor": self.frequency_scale_factor,
            "max_frequency_multiplier": self.max_frequency_multiplier,
            "min_frequency_multiplier": self.min_frequency_multiplier,
            "conditional_penalty": self.conditional_penalty,
            "nested_penalty": self.nested_penalty,
            "cross_module_bonus": self.cross_module_bonus,
            "recursive_penalty": self.recursive_penalty,
            "type_hint_bonus": self.type_hint_bonus,
            "docstring_bonus": self.docstring_bonus,
            "syntax_error_penalty": self.syntax_error_penalty,
        }


class CallWeightCalculator:
    """
    Service for calculating weights and confidence scores for function calls.

    This service implements the core weight calculation algorithms with support for:
    - Configurable base weights per call type
    - Frequency-based weight adjustments
    - Context-based weight modifiers
    - Confidence scoring based on AST completeness
    """

    def __init__(self, config: WeightConfiguration | None = None):
        """
        Initialize the weight calculator with configuration.

        Args:
            config: Weight configuration. If None, uses default configuration.
        """
        self.config = config or WeightConfiguration()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def calculate_base_weight(self, call: FunctionCall) -> float:
        """
        Calculate the base weight for a function call based on its type.

        Args:
            call: The function call to calculate weight for

        Returns:
            Base weight value for the call type
        """
        base_weight = self.config.get_base_weight(call.call_type)

        self.logger.debug(f"Base weight for {call.call_type.value} call " f"'{call.call_expression}': {base_weight}")

        return base_weight

    def calculate_frequency_factor(self, frequency_in_file: int, total_calls_in_file: int) -> float:
        """
        Calculate frequency factor based on how often this call appears.

        This implements a logarithmic scaling to prevent extremely frequent calls
        from dominating the weight calculation.

        Args:
            frequency_in_file: Number of times this specific call appears
            total_calls_in_file: Total number of calls in the file

        Returns:
            Frequency factor (multiplier for base weight)
        """
        if frequency_in_file <= 0 or total_calls_in_file <= 0:
            return 1.0

        # Calculate relative frequency (0.0 to 1.0)
        relative_frequency = frequency_in_file / total_calls_in_file

        # Apply logarithmic scaling to prevent excessive dominance
        import math

        scaled_frequency = math.log(1 + relative_frequency * self.config.frequency_scale_factor * 10)

        # Convert to multiplier with bounds
        frequency_factor = 1.0 + scaled_frequency

        # Apply bounds
        frequency_factor = max(self.config.min_frequency_multiplier, frequency_factor)
        frequency_factor = min(self.config.max_frequency_multiplier, frequency_factor)

        self.logger.debug(
            f"Frequency factor: {frequency_in_file}/{total_calls_in_file} "
            f"-> relative={relative_frequency:.3f} -> factor={frequency_factor:.3f}"
        )

        return frequency_factor

    def calculate_context_modifiers(self, call: FunctionCall) -> float:
        """
        Calculate context-based weight modifiers.

        This applies various adjustments based on the call context:
        - Conditional calls (in if/try blocks) get reduced weight
        - Nested calls get slightly reduced weight
        - Cross-module calls get increased weight
        - Recursive calls get reduced weight
        - Calls with type hints/docs get increased weight
        - Calls with syntax errors get heavily reduced weight

        Args:
            call: The function call to analyze

        Returns:
            Context modifier (multiplier for base weight)
        """
        modifier = 1.0
        modifiers_applied = []

        # Apply conditional penalty
        if call.is_conditional:
            modifier *= self.config.conditional_penalty
            modifiers_applied.append(f"conditional({self.config.conditional_penalty})")

        # Apply nested call penalty
        if call.is_nested:
            modifier *= self.config.nested_penalty
            modifiers_applied.append(f"nested({self.config.nested_penalty})")

        # Apply cross-module bonus
        if call.is_cross_module_call():
            modifier *= self.config.cross_module_bonus
            modifiers_applied.append(f"cross_module({self.config.cross_module_bonus})")

        # Apply recursive penalty
        if call.is_recursive_call():
            modifier *= self.config.recursive_penalty
            modifiers_applied.append(f"recursive({self.config.recursive_penalty})")

        # Apply type hint bonus
        if call.has_type_hints:
            modifier *= self.config.type_hint_bonus
            modifiers_applied.append(f"type_hints({self.config.type_hint_bonus})")

        # Apply docstring bonus
        if call.has_docstring:
            modifier *= self.config.docstring_bonus
            modifiers_applied.append(f"docstring({self.config.docstring_bonus})")

        # Apply syntax error penalty
        if call.has_syntax_errors:
            modifier *= self.config.syntax_error_penalty
            modifiers_applied.append(f"syntax_error({self.config.syntax_error_penalty})")

        if modifiers_applied:
            self.logger.debug(f"Context modifiers for '{call.call_expression}': " f"{' * '.join(modifiers_applied)} = {modifier:.3f}")

        return modifier

    def calculate_weight(self, call: FunctionCall, frequency_context: dict[str, int] | None = None) -> float:
        """
        Calculate the final weight for a function call.

        This combines base weight, frequency factor, and context modifiers to
        produce the final weight value.

        Args:
            call: The function call to calculate weight for
            frequency_context: Optional context with total call counts for frequency calculation

        Returns:
            Final calculated weight
        """
        # Calculate base weight
        base_weight = self.calculate_base_weight(call)

        # Calculate frequency factor
        total_calls = frequency_context.get("total_calls", 1) if frequency_context else 1
        frequency_factor = self.calculate_frequency_factor(call.frequency_in_file, total_calls)

        # Calculate context modifiers
        context_modifier = self.calculate_context_modifiers(call)

        # Combine all factors
        final_weight = base_weight * frequency_factor * context_modifier

        self.logger.debug(
            f"Weight calculation for '{call.call_expression}': "
            f"base={base_weight:.3f} * freq={frequency_factor:.3f} * "
            f"context={context_modifier:.3f} = {final_weight:.3f}"
        )

        return final_weight

    def calculate_weights_for_calls(self, calls: list[FunctionCall]) -> list[FunctionCall]:
        """
        Calculate weights for a list of function calls.

        This method updates the weight and frequency_factor fields for each call
        in the list, taking into account the overall call distribution.

        Args:
            calls: List of function calls to calculate weights for

        Returns:
            List of function calls with updated weight information
        """
        if not calls:
            return calls

        # Build frequency context
        total_calls = len(calls)
        frequency_context = {"total_calls": total_calls}

        # Calculate frequency distribution
        call_frequency = {}
        for call in calls:
            key = f"{call.target_breadcrumb}@{call.line_number}"
            call_frequency[key] = call_frequency.get(key, 0) + 1

        # Update frequency counts and calculate weights
        updated_calls = []
        for call in calls:
            # Update frequency information
            key = f"{call.target_breadcrumb}@{call.line_number}"
            call.frequency_in_file = call_frequency[key]

            # Calculate weight
            weight = self.calculate_weight(call, frequency_context)
            call.weight = weight

            # Calculate and store frequency factor
            call.frequency_factor = self.calculate_frequency_factor(call.frequency_in_file, total_calls)

            updated_calls.append(call)

        self.logger.info(
            f"Calculated weights for {len(updated_calls)} function calls. "
            f"Weight range: {min(c.weight for c in updated_calls):.3f} - "
            f"{max(c.weight for c in updated_calls):.3f}"
        )

        return updated_calls

    def get_weight_statistics(self, calls: list[FunctionCall]) -> dict[str, any]:
        """
        Generate weight-related statistics for a list of calls.

        Args:
            calls: List of function calls to analyze

        Returns:
            Dictionary containing weight statistics
        """
        if not calls:
            return {"total_calls": 0}

        weights = [call.weight for call in calls]
        effective_weights = [call.effective_weight for call in calls]

        # Weight distribution by call type
        weight_by_type = {}
        for call in calls:
            call_type = call.call_type.value
            if call_type not in weight_by_type:
                weight_by_type[call_type] = []
            weight_by_type[call_type].append(call.weight)

        # Calculate averages by type
        avg_weight_by_type = {call_type: sum(weights) / len(weights) for call_type, weights in weight_by_type.items()}

        return {
            "total_calls": len(calls),
            "weight_stats": {
                "min": min(weights),
                "max": max(weights),
                "avg": sum(weights) / len(weights),
                "median": sorted(weights)[len(weights) // 2],
            },
            "effective_weight_stats": {
                "min": min(effective_weights),
                "max": max(effective_weights),
                "avg": sum(effective_weights) / len(effective_weights),
                "median": sorted(effective_weights)[len(effective_weights) // 2],
            },
            "weight_by_call_type": avg_weight_by_type,
            "frequency_distribution": {
                f"freq_{freq}": len([c for c in calls if c.frequency_in_file == freq])
                for freq in sorted({call.frequency_in_file for call in calls})
            },
        }

    def update_configuration(self, new_config: WeightConfiguration) -> None:
        """
        Update the weight calculation configuration.

        Args:
            new_config: New configuration to use
        """
        self.config = new_config
        self.logger.info("Updated weight calculation configuration")

    def get_configuration(self) -> WeightConfiguration:
        """
        Get the current weight calculation configuration.

        Returns:
            Current configuration
        """
        return self.config

    def validate_call_weights(self, calls: list[FunctionCall]) -> tuple[bool, list[str]]:
        """
        Validate that all calls have valid weight values.

        Args:
            calls: List of function calls to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        for i, call in enumerate(calls):
            # Check weight is non-negative
            if call.weight < 0.0:
                errors.append(f"Call {i}: Weight {call.weight} is negative")

            # Check frequency factor is valid
            if call.frequency_factor < 0.0:
                errors.append(f"Call {i}: Frequency factor {call.frequency_factor} is negative")

            # Check effective weight calculation
            expected_effective = call.weight * call.frequency_factor
            if abs(call.effective_weight - expected_effective) > 1e-6:
                errors.append(
                    f"Call {i}: Effective weight mismatch. " f"Expected {expected_effective:.6f}, got {call.effective_weight:.6f}"
                )

        return len(errors) == 0, errors


# Utility functions for weight calculation


def create_default_weight_config() -> WeightConfiguration:
    """
    Create a default weight configuration with PRD-specified values.

    Returns:
        Default weight configuration
    """
    return WeightConfiguration()


def create_conservative_weight_config() -> WeightConfiguration:
    """
    Create a conservative weight configuration with lower variance.

    Returns:
        Conservative weight configuration
    """
    config = WeightConfiguration()

    # Reduce weight differences between call types
    config.base_weights.update(
        {
            CallType.DIRECT: 1.0,
            CallType.METHOD: 0.95,
            CallType.ATTRIBUTE: 0.9,
            CallType.DYNAMIC: 0.8,
        }
    )

    # Reduce frequency impact
    config.frequency_scale_factor = 0.05
    config.max_frequency_multiplier = 1.5

    # Reduce context penalties
    config.conditional_penalty = 0.95
    config.nested_penalty = 0.98
    config.recursive_penalty = 0.9

    return config


def create_aggressive_weight_config() -> WeightConfiguration:
    """
    Create an aggressive weight configuration with higher variance.

    Returns:
        Aggressive weight configuration
    """
    config = WeightConfiguration()

    # Increase weight differences between call types
    config.base_weights.update(
        {
            CallType.DIRECT: 1.0,
            CallType.METHOD: 0.8,
            CallType.ATTRIBUTE: 0.5,
            CallType.DYNAMIC: 0.3,
        }
    )

    # Increase frequency impact
    config.frequency_scale_factor = 0.2
    config.max_frequency_multiplier = 3.0

    # Increase context penalties
    config.conditional_penalty = 0.8
    config.nested_penalty = 0.9
    config.recursive_penalty = 0.6

    return config
