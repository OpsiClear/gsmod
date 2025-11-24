"""Tests for validation decorators.

Tests cover:
- validate_range decorator
- validate_positive decorator
- validate_type decorator
- validate_choices decorator
"""

import numpy as np
import pytest

from gsmod.validators import (
    validate_range,
    validate_positive,
    validate_type,
    validate_choices,
)


class TestValidateRange:
    """Test validate_range decorator."""

    def test_valid_value(self):
        """Test decorator passes valid values."""
        @validate_range(0.0, 1.0, "opacity")
        def set_opacity(self, opacity: float) -> float:
            return opacity

        result = set_opacity(None, 0.5)
        assert result == 0.5

    def test_boundary_values(self):
        """Test decorator accepts boundary values."""
        @validate_range(0.0, 1.0, "value")
        def func(self, value: float) -> float:
            return value

        # Min boundary
        assert func(None, 0.0) == 0.0
        # Max boundary
        assert func(None, 1.0) == 1.0

    def test_below_range_raises(self):
        """Test decorator raises for values below range."""
        @validate_range(0.0, 1.0, "opacity")
        def func(self, opacity: float) -> float:
            return opacity

        with pytest.raises(ValueError, match="opacity=-0.1 is outside valid range"):
            func(None, -0.1)

    def test_above_range_raises(self):
        """Test decorator raises for values above range."""
        @validate_range(0.0, 1.0, "opacity")
        def func(self, opacity: float) -> float:
            return opacity

        with pytest.raises(ValueError, match="opacity=1.5 is outside valid range"):
            func(None, 1.5)

    def test_non_numeric_raises(self):
        """Test decorator raises for non-numeric values."""
        @validate_range(0.0, 1.0, "value")
        def func(self, value: float) -> float:
            return value

        with pytest.raises(TypeError, match="must be a number"):
            func(None, "not a number")

    def test_kwarg_validation(self):
        """Test decorator validates keyword arguments."""
        @validate_range(0.0, 1.0, "opacity")
        def func(self, opacity: float = 0.5) -> float:
            return opacity

        # Valid kwarg
        assert func(None, opacity=0.5) == 0.5

        # Invalid kwarg
        with pytest.raises(ValueError):
            func(None, opacity=2.0)

    def test_suggestion_for_opacity(self):
        """Test error includes suggestion for opacity parameters."""
        @validate_range(0.0, 1.0, "opacity")
        def func(self, opacity: float) -> float:
            return opacity

        with pytest.raises(ValueError, match="fully transparent"):
            func(None, -0.1)

    def test_suggestion_for_brightness(self):
        """Test error includes suggestion for brightness parameters."""
        @validate_range(0.1, 3.0, "brightness")
        def func(self, brightness: float) -> float:
            return brightness

        with pytest.raises(ValueError, match="no change"):
            func(None, 0.0)

    def test_suggestion_for_temperature(self):
        """Test error includes suggestion for temperature parameters."""
        @validate_range(0.0, 1.0, "temperature")
        def func(self, temperature: float) -> float:
            return temperature

        with pytest.raises(ValueError, match="neutral"):
            func(None, 1.5)

    def test_suggestion_for_lut_size(self):
        """Test error includes suggestion for lut_size parameters."""
        @validate_range(16, 4096, "lut_size")
        def func(self, lut_size: int) -> int:
            return lut_size

        with pytest.raises(ValueError, match="Larger LUTs"):
            func(None, 10)

    def test_no_value_provided(self):
        """Test decorator passes through when no value provided."""
        @validate_range(0.0, 1.0, "opacity")
        def func(self) -> str:
            return "no args"

        result = func(None)
        assert result == "no args"


class TestValidatePositive:
    """Test validate_positive decorator."""

    def test_valid_positive_value(self):
        """Test decorator passes positive values."""
        @validate_positive("gamma")
        def set_gamma(self, gamma: float) -> float:
            return gamma

        result = set_gamma(None, 2.2)
        assert result == 2.2

    def test_zero_raises(self):
        """Test decorator raises for zero values."""
        @validate_positive("scale")
        def func(self, scale: float) -> float:
            return scale

        with pytest.raises(ValueError, match="must be positive"):
            func(None, 0.0)

    def test_negative_raises(self):
        """Test decorator raises for negative values."""
        @validate_positive("gamma")
        def func(self, gamma: float) -> float:
            return gamma

        with pytest.raises(ValueError, match="must be positive"):
            func(None, -1.0)

    def test_non_numeric_raises(self):
        """Test decorator raises for non-numeric values."""
        @validate_positive("scale")
        def func(self, scale: float) -> float:
            return scale

        with pytest.raises(TypeError, match="must be a number"):
            func(None, "not a number")

    def test_suggestion_for_gamma(self):
        """Test error includes suggestion for gamma parameters."""
        @validate_positive("gamma")
        def func(self, gamma: float) -> float:
            return gamma

        with pytest.raises(ValueError, match="linear"):
            func(None, 0.0)

    def test_suggestion_for_scale(self):
        """Test error includes suggestion for scale parameters."""
        @validate_positive("scale")
        def func(self, scale: float) -> float:
            return scale

        with pytest.raises(ValueError, match="enlarge"):
            func(None, -1.0)

    def test_kwarg_validation(self):
        """Test decorator validates keyword arguments."""
        @validate_positive("scale")
        def func(self, scale: float = 1.0) -> float:
            return scale

        assert func(None, scale=2.0) == 2.0
        with pytest.raises(ValueError):
            func(None, scale=-1.0)


class TestValidateType:
    """Test validate_type decorator."""

    def test_valid_single_type(self):
        """Test decorator passes valid single type."""
        @validate_type(np.ndarray, "vector")
        def func(self, vector: np.ndarray) -> np.ndarray:
            return vector

        arr = np.array([1, 2, 3])
        result = func(None, arr)
        np.testing.assert_array_equal(result, arr)

    def test_valid_tuple_of_types(self):
        """Test decorator passes valid value from tuple of types."""
        @validate_type((int, float), "value")
        def func(self, value) -> float:
            return float(value)

        assert func(None, 5) == 5.0
        assert func(None, 3.14) == 3.14

    def test_invalid_single_type_raises(self):
        """Test decorator raises for invalid single type."""
        @validate_type(np.ndarray, "vector")
        def func(self, vector: np.ndarray) -> np.ndarray:
            return vector

        with pytest.raises(TypeError, match="must be ndarray"):
            func(None, [1, 2, 3])

    def test_invalid_tuple_types_raises(self):
        """Test decorator raises for invalid value from tuple types."""
        @validate_type((int, float), "value")
        def func(self, value) -> float:
            return float(value)

        with pytest.raises(TypeError, match="must be one of"):
            func(None, "not a number")

    def test_kwarg_validation(self):
        """Test decorator validates keyword arguments."""
        @validate_type(str, "name")
        def func(self, name: str = "default") -> str:
            return name

        assert func(None, name="test") == "test"
        with pytest.raises(TypeError):
            func(None, name=123)


class TestValidateChoices:
    """Test validate_choices decorator."""

    def test_valid_choice(self):
        """Test decorator passes valid choices."""
        @validate_choices({"quaternion", "matrix", "euler"}, "format")
        def func(self, format: str) -> str:
            return format

        assert func(None, "quaternion") == "quaternion"
        assert func(None, "matrix") == "matrix"
        assert func(None, "euler") == "euler"

    def test_invalid_choice_raises(self):
        """Test decorator raises for invalid choice."""
        @validate_choices({"a", "b", "c"}, "option")
        def func(self, option: str) -> str:
            return option

        with pytest.raises(ValueError, match='option="invalid" is not valid'):
            func(None, "invalid")

    def test_error_shows_valid_options(self):
        """Test error message shows all valid options."""
        @validate_choices({"red", "green", "blue"}, "color")
        def func(self, color: str) -> str:
            return color

        with pytest.raises(ValueError, match="blue") as exc:
            func(None, "yellow")
        assert "green" in str(exc.value)
        assert "red" in str(exc.value)

    def test_kwarg_validation(self):
        """Test decorator validates keyword arguments."""
        @validate_choices({"yes", "no"}, "confirm")
        def func(self, confirm: str = "no") -> str:
            return confirm

        assert func(None, confirm="yes") == "yes"
        with pytest.raises(ValueError):
            func(None, confirm="maybe")


class TestValidatorCombinations:
    """Test combinations of validators."""

    def test_multiple_decorators(self):
        """Test multiple decorators on same function."""
        @validate_range(0.0, 1.0, "opacity", param_index=1)
        @validate_positive("scale", param_index=2)
        def func(self, opacity: float, scale: float) -> tuple[float, float]:
            return opacity, scale

        result = func(None, 0.5, 2.0)
        assert result == (0.5, 2.0)

    def test_decorator_with_different_param_indices(self):
        """Test decorators with custom param indices."""
        @validate_range(0.0, 1.0, "alpha", param_index=2)
        def func(self, name: str, alpha: float) -> float:
            return alpha

        result = func(None, "test", 0.5)
        assert result == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
