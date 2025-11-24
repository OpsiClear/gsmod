"""
Parameter placeholders for parameterized pipeline templates.

This module provides the Param class for creating pipeline templates with
named parameters that can be substituted at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Param:
    """
    Parameter placeholder for pipeline templates.

    Use this to create parameterized pipelines that can be efficiently
    reused with different parameter values. Parameters are validated
    against their defined ranges and cached for performance.

    Attributes:
        name: Parameter identifier (e.g., "brightness", "contrast")
        default: Default value to use if not overridden
        range: Optional (min, max) tuple for validation

    Example:
        >>> from gsmod import Color, Param
        >>> template = Color.template(
        ...     brightness=Param("b", default=1.2, range=(0.5, 2.0)),
        ...     contrast=Param("c", default=1.1, range=(0.5, 2.0))
        ... )
        >>> result = template(data, params={"b": 1.5, "c": 1.2})
    """

    name: str
    default: float
    range: tuple[float, float] | None = None

    def __post_init__(self):
        """Validate parameter definition."""
        if self.range is not None:
            min_val, max_val = self.range
            if min_val >= max_val:
                raise ValueError(
                    f"Invalid range for {self.name}: min ({min_val}) must be < max ({max_val})"
                )
            if not (min_val <= self.default <= max_val):
                raise ValueError(
                    f"Default value {self.default} for {self.name} outside range {self.range}"
                )

    def validate(self, value: float) -> float:
        """
        Validate a parameter value against its range.

        :param value: Value to validate
        :return: Validated value as float
        :raises ValueError: If value is outside defined range

        Example:
            >>> param = Param("brightness", default=1.0, range=(0.5, 2.0))
            >>> param.validate(1.5)  # OK
            1.5
            >>> param.validate(3.0)  # Raises ValueError
        """
        if self.range is not None:
            min_val, max_val = self.range
            if not (min_val <= value <= max_val):
                raise ValueError(f"{self.name}={value} outside valid range {self.range}")
        return float(value)

    def __repr__(self) -> str:
        """String representation."""
        if self.range is not None:
            return f'Param(name="{self.name}", default={self.default}, range={self.range})'
        return f'Param(name="{self.name}", default={self.default})'
