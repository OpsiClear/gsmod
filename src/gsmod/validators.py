"""
Validation decorators for gsmod pipelines.

Provides reusable validation logic for parameter checking across all pipelines.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

# Python 3.12+ type alias for callables
type F = Callable[..., Any]


def validate_range(
    min_val: float,
    max_val: float,
    param_name: str = "value",
    param_index: int = 1,
) -> Callable[[F], F]:
    """
    Decorator for validating numeric parameter ranges.

    :param min_val: Minimum allowed value (inclusive)
    :param max_val: Maximum allowed value (inclusive)
    :param param_name: Name of parameter for error messages
    :param param_index: Position of parameter in function signature (default: 1 = first arg after self)
    :return: Decorated function with range validation

    Example:
        >>> @validate_range(0.0, 1.0, "opacity")
        ... def opacity(self, threshold: float) -> Self:
        ...     self._opacity = threshold
        ...     return self
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get value from args or kwargs
            if len(args) > param_index:
                value = args[param_index]
            elif param_name in kwargs:
                value = kwargs[param_name]
            else:
                # No value provided, let function handle it
                return func(*args, **kwargs)

            # Validate range
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{param_name} must be a number, got {type(value).__name__}. "
                    f"Provide a numeric value (int or float)."
                )

            if not min_val <= value <= max_val:
                # Provide helpful suggestions based on the parameter name and range
                suggestion = ""
                if param_name == "lut_size":
                    suggestion = " Larger LUTs increase memory usage with diminishing quality gains. Use 1024 (default) or 4096 for high precision."
                elif "opacity" in param_name:
                    suggestion = " Use 0.0 for fully transparent, 1.0 for fully opaque."
                elif (
                    "brightness" in param_name
                    or "contrast" in param_name
                    or "saturation" in param_name
                ):
                    suggestion = " Use 1.0 for no change, >1.0 to increase, <1.0 to decrease."
                elif "temperature" in param_name:
                    suggestion = " Use 0.0 for cool (blue), 0.5 for neutral, 1.0 for warm (orange)."

                raise ValueError(
                    f"{param_name}={value} is outside valid range [{min_val}, {max_val}].{suggestion}"
                )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def validate_positive(param_name: str = "value", param_index: int = 1) -> Callable[[F], F]:
    """
    Decorator for validating positive numeric parameters.

    :param param_name: Name of parameter for error messages
    :param param_index: Position of parameter in function signature
    :return: Decorated function with positive validation

    Example:
        >>> @validate_positive("scale")
        ... def scale(self, factor: float) -> Self:
        ...     self._scale = factor
        ...     return self
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get value from args or kwargs
            if len(args) > param_index:
                value = args[param_index]
            elif param_name in kwargs:
                value = kwargs[param_name]
            else:
                return func(*args, **kwargs)

            # Validate positive
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{param_name} must be a number, got {type(value).__name__}. "
                    f"Provide a numeric value (int or float)."
                )

            if value <= 0:
                suggestion = ""
                if "gamma" in param_name:
                    suggestion = " Gamma must be positive. Use 1.0 for linear, <1.0 to brighten, >1.0 to darken."
                elif "scale" in param_name:
                    suggestion = " Scale must be positive. Use 1.0 for no change, >1.0 to enlarge, <1.0 to shrink."

                raise ValueError(f"{param_name}={value} must be positive (> 0).{suggestion}")

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def validate_type(
    expected_type: type | tuple[type, ...],
    param_name: str = "value",
    param_index: int = 1,
) -> Callable[[F], F]:
    """
    Decorator for validating parameter types.

    :param expected_type: Expected type or tuple of types
    :param param_name: Name of parameter for error messages
    :param param_index: Position of parameter in function signature
    :return: Decorated function with type validation

    Example:
        >>> @validate_type(np.ndarray, "vector")
        ... def translate(self, vector: np.ndarray) -> Self:
        ...     self._translation = vector
        ...     return self
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get value from args or kwargs
            if len(args) > param_index:
                value = args[param_index]
            elif param_name in kwargs:
                value = kwargs[param_name]
            else:
                return func(*args, **kwargs)

            # Validate type
            if not isinstance(value, expected_type):
                if isinstance(expected_type, tuple):
                    type_names = ", ".join(t.__name__ for t in expected_type)
                    raise TypeError(
                        f"{param_name} must be one of ({type_names}), got {type(value).__name__}"
                    )
                else:
                    raise TypeError(
                        f"{param_name} must be {expected_type.__name__}, got {type(value).__name__}"
                    )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def validate_choices(
    valid_choices: set[str],
    param_name: str = "value",
    param_index: int = 1,
) -> Callable[[F], F]:
    """
    Decorator for validating parameter choices.

    :param valid_choices: Set of valid string choices
    :param param_name: Name of parameter for error messages
    :param param_index: Position of parameter in function signature
    :return: Decorated function with choice validation

    Example:
        >>> @validate_choices({"quaternion", "matrix", "euler"}, "format")
        ... def rotate(self, rotation, format: str = "quaternion") -> Self:
        ...     ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get value from args or kwargs
            if len(args) > param_index:
                value = args[param_index]
            elif param_name in kwargs:
                value = kwargs[param_name]
            else:
                return func(*args, **kwargs)

            # Validate choice
            if value not in valid_choices:
                choices_str = ", ".join(sorted(valid_choices))
                raise ValueError(
                    f'{param_name}="{value}" is not valid. Valid options are: {choices_str}'
                )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator
