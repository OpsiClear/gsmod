"""Operation specifications for pipeline configuration.

This module defines the OperationSpec dataclass that specifies parameter
ranges, defaults, and composition behavior for pipeline operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class OperationSpec:
    """Specification for a pipeline operation parameter.

    Attributes:
        name: Operation name (e.g., "brightness", "temperature")
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        default: Default value when not specified
        neutral: Value that causes no change (identity)
        composition: How multiple operations combine ("multiplicative" or "additive")
        description: Human-readable description
    """

    name: str
    min_value: float
    max_value: float
    default: float
    neutral: float
    composition: Literal["multiplicative", "additive"]
    description: str = ""

    def validate(self, value: float) -> float:
        """Validate and clamp value to allowed range.

        :param value: Value to validate
        :returns: Clamped value within [min_value, max_value]
        :raises ValueError: If value is not a number
        """
        if not isinstance(value, int | float):
            raise ValueError(f"{self.name}: expected number, got {type(value).__name__}")

        # Clamp to range
        return max(self.min_value, min(self.max_value, float(value)))

    def is_neutral(self, value: float, tolerance: float = 1e-6) -> bool:
        """Check if value is effectively neutral (no change).

        :param value: Value to check
        :param tolerance: Tolerance for floating point comparison
        :returns: True if value is within tolerance of neutral
        """
        return abs(value - self.neutral) < tolerance

    def combine(self, a: float, b: float) -> float:
        """Combine two values according to composition rule.

        :param a: First value
        :param b: Second value
        :returns: Combined value
        """
        if self.composition == "multiplicative":
            return a * b
        else:  # additive
            return a + b - self.neutral  # Subtract neutral to avoid double-counting

    def __repr__(self) -> str:
        return (
            f"OperationSpec({self.name}, "
            f"range=[{self.min_value}, {self.max_value}], "
            f"default={self.default}, neutral={self.neutral}, "
            f"{self.composition})"
        )
