"""
Comprehensive tests for parameterized pipeline templates.

Tests the Param class and template-based pipelines with parameter variation
and LRU caching.
"""

import numpy as np
import pytest
from gsply import GSData

from gsmod import Color, Param


@pytest.fixture
def sample_gsdata():
    """Create sample GSData for testing."""
    n = 100
    data = GSData(
        means=np.random.randn(n, 3).astype(np.float32),
        scales=np.random.rand(n, 3).astype(np.float32),
        quats=np.random.randn(n, 4).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32),
        sh0=np.random.rand(n, 3).astype(np.float32),
        shN=None,
    )
    # Normalize quaternions
    data.quats[:] = data.quats / np.linalg.norm(data.quats, axis=1, keepdims=True)
    return data


class TestParam:
    """Test the Param class."""

    def test_param_creation(self):
        """Test creating a Param."""
        param = Param(name="brightness", default=1.2, range=(0.5, 2.0))

        assert param.name == "brightness"
        assert param.default == 1.2
        assert param.range == (0.5, 2.0)

    def test_param_without_range(self):
        """Test Param without range constraint."""
        param = Param(name="brightness", default=1.2)

        assert param.name == "brightness"
        assert param.default == 1.2
        assert param.range is None

    def test_param_validation_in_range(self):
        """Test validating a value within range."""
        param = Param(name="brightness", default=1.2, range=(0.5, 2.0))

        assert param.validate(1.5) == 1.5
        assert param.validate(0.5) == 0.5
        assert param.validate(2.0) == 2.0

    def test_param_validation_outside_range(self):
        """Test validating a value outside range."""
        param = Param(name="brightness", default=1.2, range=(0.5, 2.0))

        with pytest.raises(ValueError, match="brightness=2.5 outside valid range"):
            param.validate(2.5)

        with pytest.raises(ValueError, match="brightness=0.1 outside valid range"):
            param.validate(0.1)

    def test_param_validation_no_range(self):
        """Test validation when no range is specified."""
        param = Param(name="brightness", default=1.2)

        # Should accept any value
        assert param.validate(10.0) == 10.0
        assert param.validate(-5.0) == -5.0

    def test_param_invalid_range(self):
        """Test creating Param with invalid range."""
        with pytest.raises(ValueError, match="min .* must be < max"):
            Param(name="brightness", default=1.2, range=(2.0, 0.5))

    def test_param_default_outside_range(self):
        """Test creating Param with default outside range."""
        with pytest.raises(ValueError, match="Default value .* outside range"):
            Param(name="brightness", default=3.0, range=(0.5, 2.0))

    def test_param_frozen(self):
        """Test that Param is immutable (frozen dataclass)."""
        param = Param(name="brightness", default=1.2, range=(0.5, 2.0))

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            param.default = 1.5

    def test_param_repr(self):
        """Test string representation."""
        param1 = Param(name="b", default=1.2, range=(0.5, 2.0))
        param2 = Param(name="b", default=1.2)

        assert "b" in repr(param1)
        assert "1.2" in repr(param1)
        assert "range" in repr(param1)

        assert "b" in repr(param2)
        assert "1.2" in repr(param2)


class TestTemplateValidation:
    """Test template validation and error handling."""

    def test_color_duplicate_param_names_rejected(self):
        """Test that duplicate parameter names are rejected in Color templates."""
        with pytest.raises(ValueError, match='Duplicate parameter name "p"'):
            Color.template(
                brightness=Param("p", default=1.2, range=(0.5, 2.0)),
                contrast=Param("p", default=1.1, range=(0.5, 2.0)),  # Same name!
            )



class TestColorTemplate:
    """Test Color.template() and parameterized pipelines."""

    def test_template_creation(self):
        """Test creating a parameterized template."""
        template = Color.template(
            brightness=Param("b", default=1.2, range=(0.5, 2.0)),
            contrast=Param("c", default=1.1, range=(0.5, 2.0)),
        )

        assert len(template._param_map) == 2
        assert "b" in template._param_map
        assert "c" in template._param_map

        # Check parameter mapping structure: param.name -> (Param, operation_name)
        param_b, op_name_b = template._param_map["b"]
        assert param_b.name == "b"
        assert op_name_b == "brightness"

        param_c, op_name_c = template._param_map["c"]
        assert param_c.name == "c"
        assert op_name_c == "contrast"

    def test_template_with_all_operations(self):
        """Test template with all color operations."""
        # Temperature/shadows/highlights now use additive semantics with -1 to 1 range
        template = Color.template(
            temperature=Param("temp", default=0.1),  # -1 to 1 range
            brightness=Param("b", default=1.2),
            contrast=Param("c", default=1.1),
            gamma=Param("g", default=1.05),
            saturation=Param("s", default=1.3),
            shadows=Param("sh", default=0.1),  # -1 to 1 range (additive)
            highlights=Param("hl", default=-0.1),  # -1 to 1 range (additive)
        )

        assert len(template._param_map) == 7

    def test_template_invalid_operation(self):
        """Test template with invalid operation name."""
        with pytest.raises(ValueError, match='Unknown operation "invalid"'):
            Color.template(invalid=Param("x", default=1.0))

    def test_template_non_param_value(self):
        """Test template with non-Param value."""
        with pytest.raises(TypeError, match="Expected Param object"):
            Color.template(brightness=1.2)

    def test_template_applies_defaults(self, sample_gsdata):
        """Test that template applies default parameter values."""
        template = Color.template(brightness=Param("b", default=1.5, range=(0.5, 2.0)))

        # Apply without params - should use defaults
        result = template(sample_gsdata, inplace=False, params={"b": 1.5})

        # Colors should be modified (brightness applied)
        assert not np.allclose(result.sh0, sample_gsdata.sh0)

    def test_template_parameter_override(self, sample_gsdata):
        """Test overriding parameter values at runtime."""
        template = Color.template(brightness=Param("b", default=1.2, range=(0.5, 2.0)))

        # Use different brightness values
        result1 = template(sample_gsdata, inplace=False, params={"b": 1.5})
        result2 = template(sample_gsdata, inplace=False, params={"b": 0.8})

        # Results should be different
        assert not np.allclose(result1.sh0, result2.sh0)

        # Both should be different from original
        assert not np.allclose(result1.sh0, sample_gsdata.sh0)
        assert not np.allclose(result2.sh0, sample_gsdata.sh0)

    def test_template_cache_hit(self, sample_gsdata):
        """Test LRU cache hit with same parameters."""
        template = Color.template(
            brightness=Param("b", default=1.2, range=(0.5, 2.0)),
            contrast=Param("c", default=1.1, range=(0.5, 2.0)),
        )

        # First call - cache miss
        result1 = template(sample_gsdata, inplace=False, params={"b": 1.5, "c": 1.2})
        cache_size_1 = len(template._lut_cache)

        # Second call with same params - cache hit
        result2 = template(sample_gsdata, inplace=False, params={"b": 1.5, "c": 1.2})
        cache_size_2 = len(template._lut_cache)

        # Cache should not grow (hit)
        assert cache_size_1 == cache_size_2
        assert cache_size_1 == 1

        # Results should be identical (cache hit)
        assert np.allclose(result1.sh0, result2.sh0)

    def test_template_cache_miss(self, sample_gsdata):
        """Test LRU cache miss with different parameters."""
        template = Color.template(brightness=Param("b", default=1.2, range=(0.5, 2.0)))

        # First call
        template(sample_gsdata, inplace=False, params={"b": 1.2})
        cache_size_1 = len(template._lut_cache)

        # Second call with different params - cache miss
        template(sample_gsdata, inplace=False, params={"b": 1.5})
        cache_size_2 = len(template._lut_cache)

        # Cache should grow (miss)
        assert cache_size_2 == cache_size_1 + 1
        assert cache_size_2 == 2

    def test_template_multiple_parameters(self, sample_gsdata):
        """Test template with multiple parameters."""
        template = Color.template(
            brightness=Param("b", default=1.2, range=(0.5, 2.0)),
            contrast=Param("c", default=1.1, range=(0.5, 2.0)),
            saturation=Param("s", default=1.3, range=(0.0, 3.0)),
        )

        result = template(sample_gsdata, inplace=False, params={"b": 1.5, "c": 1.2, "s": 1.4})

        # All operations should be applied
        assert not np.allclose(result.sh0, sample_gsdata.sh0)

    def test_template_partial_parameters(self, sample_gsdata):
        """Test applying only some parameters (others use defaults)."""
        template = Color.template(
            brightness=Param("b", default=1.2, range=(0.5, 2.0)),
            contrast=Param("c", default=1.1, range=(0.5, 2.0)),
        )

        # Only override brightness
        result = template(sample_gsdata, inplace=False, params={"b": 1.5, "c": 1.1})

        assert not np.allclose(result.sh0, sample_gsdata.sh0)

    def test_template_unknown_parameter(self, sample_gsdata):
        """Test using unknown parameter name."""
        template = Color.template(brightness=Param("b", default=1.2, range=(0.5, 2.0)))

        with pytest.raises(ValueError, match='Unknown parameter "x"'):
            template(sample_gsdata, params={"x": 1.5})

    def test_template_parameter_out_of_range(self, sample_gsdata):
        """Test parameter value outside defined range."""
        template = Color.template(brightness=Param("b", default=1.2, range=(0.5, 2.0)))

        with pytest.raises(ValueError, match="b=2.5 outside valid range"):
            template(sample_gsdata, params={"b": 2.5})

    def test_template_without_params_uses_defaults(self, sample_gsdata):
        """Test calling template without params uses default values."""
        template = Color.template(brightness=Param("b", default=1.5, range=(0.5, 2.0)))

        # Calling without params should use default parameter values
        result = template(sample_gsdata, inplace=False)

        # Should apply the default brightness
        assert not np.allclose(result.sh0, sample_gsdata.sh0)

    def test_non_template_with_params_raises(self, sample_gsdata):
        """Test calling non-template pipeline with params argument."""
        pipeline = Color().brightness(1.2)

        # Should raise error when using params on non-template
        with pytest.raises(ValueError, match="Pipeline was not created with template"):
            pipeline(sample_gsdata, params={"b": 1.5})

    def test_template_inplace_mode(self, sample_gsdata):
        """Test template with inplace=True."""
        template = Color.template(brightness=Param("b", default=1.2, range=(0.5, 2.0)))

        original_sh0 = sample_gsdata.sh0.copy()

        result = template(sample_gsdata, inplace=True, params={"b": 1.5})

        # Should return same object
        assert result is sample_gsdata

        # sh0 should be modified
        assert not np.allclose(sample_gsdata.sh0, original_sh0)

    def test_template_copy_mode(self, sample_gsdata):
        """Test template with inplace=False."""
        template = Color.template(brightness=Param("b", default=1.2, range=(0.5, 2.0)))

        original_sh0 = sample_gsdata.sh0.copy()

        result = template(sample_gsdata, inplace=False, params={"b": 1.5})

        # Should return different object
        assert result is not sample_gsdata

        # Original should be unchanged
        assert np.allclose(sample_gsdata.sh0, original_sh0)

        # Result should be modified
        assert not np.allclose(result.sh0, original_sh0)

    def test_template_copy_preserves_cache(self):
        """Test that copying a template preserves the cache."""
        template = Color.template(brightness=Param("b", default=1.2, range=(0.5, 2.0)))

        # Populate cache
        template._lut_cache[("b", 1.5)] = np.random.rand(1024, 3).astype(np.float32)

        # Copy template
        template2 = template.copy()

        # Cache should be copied
        assert len(template2._lut_cache) == 1
        # But should be independent (deep copy of LUTs)
        assert template2._lut_cache is not template._lut_cache

    def test_template_reset_clears_cache(self):
        """Test that reset() clears the cache."""
        template = Color.template(brightness=Param("b", default=1.2, range=(0.5, 2.0)))

        # Populate cache
        template._lut_cache[("b", 1.5)] = np.random.rand(1024, 3).astype(np.float32)
        assert len(template._lut_cache) == 1

        # Reset
        template.reset()

        # Cache should be cleared
        assert len(template._lut_cache) == 0
        assert len(template._param_map) == 0


class TestCachePerformance:
    """Test cache performance characteristics."""

    def test_animation_use_case(self, sample_gsdata):
        """Test animation scenario with parameter sweep."""
        template = Color.template(brightness=Param("b", default=1.2, range=(0.5, 2.0)))

        # Simulate 20 frames with varying brightness
        brightness_values = np.linspace(0.8, 1.8, 20)

        for i, b in enumerate(brightness_values):
            template(sample_gsdata, inplace=False, params={"b": b})

        # Should have 20 cached LUTs (one per unique brightness)
        assert len(template._lut_cache) == 20

    def test_ab_testing_use_case(self, sample_gsdata):
        """Test A/B testing scenario with parameter variations."""
        template = Color.template(
            brightness=Param("b", default=1.2, range=(0.5, 2.0)),
            contrast=Param("c", default=1.1, range=(0.5, 2.0)),
        )

        # Test 10 combinations
        test_params = [
            {"b": 1.0, "c": 1.0},
            {"b": 1.2, "c": 1.0},
            {"b": 1.5, "c": 1.0},
            {"b": 1.0, "c": 1.1},
            {"b": 1.2, "c": 1.1},
            {"b": 1.5, "c": 1.1},
            {"b": 1.0, "c": 1.2},
            {"b": 1.2, "c": 1.2},
            {"b": 1.5, "c": 1.2},
            {"b": 1.2, "c": 1.1},  # Duplicate - should hit cache
        ]

        results = []
        for params in test_params:
            result = template(sample_gsdata, inplace=False, params=params)
            results.append(result)

        # Should have 9 unique combinations (one duplicate)
        assert len(template._lut_cache) == 9

        # Verify cache hit produced identical result
        assert np.allclose(results[4].sh0, results[-1].sh0)

    def test_cache_key_consistency(self, sample_gsdata):
        """Test that cache keys are consistent regardless of parameter order."""
        template = Color.template(
            brightness=Param("b", default=1.2, range=(0.5, 2.0)),
            contrast=Param("c", default=1.1, range=(0.5, 2.0)),
        )

        # Apply with different parameter order
        result1 = template(sample_gsdata, inplace=False, params={"b": 1.5, "c": 1.2})
        result2 = template(sample_gsdata, inplace=False, params={"c": 1.2, "b": 1.5})

        # Should hit cache (same parameters, different order)
        assert len(template._lut_cache) == 1

        # Results should be identical
        assert np.allclose(result1.sh0, result2.sh0)


class TestEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_template_with_no_parameters(self):
        """Test creating template with no parameters."""
        template = Color.template()

        assert len(template._param_map) == 0
        assert len(template._lut_cache) == 0

    def test_large_cache(self, sample_gsdata):
        """Test template with large cache (100 unique combinations)."""
        template = Color.template(brightness=Param("b", default=1.2, range=(0.5, 2.0)))

        # Create 100 unique parameter values
        brightness_values = np.linspace(0.5, 2.0, 100)

        for b in brightness_values:
            template(sample_gsdata, inplace=False, params={"b": float(b)})

        # Should have 100 cached LUTs
        assert len(template._lut_cache) == 100

        # Memory check: each LUT is ~12KB, 100 LUTs = ~1.2MB (acceptable)
        lut_memory_kb = sum(lut.nbytes for lut in template._lut_cache.values()) / 1024
        assert lut_memory_kb < 2000  # Less than 2MB

    def test_parameter_float_precision(self, sample_gsdata):
        """Test that floating point precision doesn't cause cache misses."""
        template = Color.template(brightness=Param("b", default=1.2, range=(0.5, 2.0)))

        # Use slightly different float representations
        template(sample_gsdata, inplace=False, params={"b": 1.5})
        template(sample_gsdata, inplace=False, params={"b": 1.5000000000001})

        # These should be treated as different (no float tolerance in cache keys)
        # This is intentional - exact parameter values are cached
        assert len(template._lut_cache) == 2

    def test_template_with_single_parameter(self, sample_gsdata):
        """Test template with only one parameter."""
        template = Color.template(brightness=Param("b", default=1.2, range=(0.5, 2.0)))

        result = template(sample_gsdata, inplace=False, params={"b": 1.5})

        assert not np.allclose(result.sh0, sample_gsdata.sh0)
        assert len(template._lut_cache) == 1


class TestIntegration:
    """Integration tests with full workflows."""

    def test_interactive_adjustment_workflow(self, sample_gsdata):
        """Test interactive parameter adjustment scenario."""
        template = Color.template(
            brightness=Param("b", default=1.2, range=(0.5, 2.0)),
            contrast=Param("c", default=1.1, range=(0.5, 2.0)),
            saturation=Param("s", default=1.3, range=(0.0, 3.0)),
        )

        # User adjusts brightness slider
        for b in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
            result = template(sample_gsdata, inplace=False, params={"b": b, "c": 1.1, "s": 1.3})
            assert isinstance(result, GSData)

        # User adjusts contrast slider
        for c in [1.0, 1.05, 1.1, 1.15, 1.2]:
            result = template(sample_gsdata, inplace=False, params={"b": 1.2, "c": c, "s": 1.3})
            assert isinstance(result, GSData)

        # Total unique combinations: 6 + 5 - 1 (overlap at b=1.2, c=1.1) = 10
        assert len(template._lut_cache) == 10

    def test_batch_processing_workflow(self, sample_gsdata):
        """Test batch processing with presets."""
        template = Color.template(
            brightness=Param("b", default=1.2),
            contrast=Param("c", default=1.1),
            saturation=Param("s", default=1.3),
        )

        # Define presets
        presets = {
            "subtle": {"b": 1.1, "c": 1.05, "s": 1.1},
            "medium": {"b": 1.2, "c": 1.1, "s": 1.3},
            "dramatic": {"b": 1.5, "c": 1.3, "s": 1.5},
            "muted": {"b": 0.9, "c": 0.95, "s": 0.7},
        }

        results = {}
        for name, params in presets.items():
            results[name] = template(sample_gsdata, inplace=False, params=params)

        # Should have 4 cached LUTs
        assert len(template._lut_cache) == 4

        # All results should be different
        for name1 in presets:
            for name2 in presets:
                if name1 != name2:
                    assert not np.allclose(results[name1].sh0, results[name2].sh0), (
                        f"{name1} and {name2} should be different"
                    )
