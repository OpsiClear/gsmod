"""Tests for configuration value classes with merge support.

Tests the mathematically correct composition semantics:
- ColorValues: multiplicative/additive/wrapped composition
- FilterValues: stricter-wins (intersection) logic
- TransformValues: matrix composition (non-commutative)
"""

import numpy as np
import pytest

from gsmod.config.values import ColorValues, FilterValues, TransformValues


class TestColorValuesMerge:
    """Test ColorValues merge operations."""

    def test_multiplicative_fields(self):
        """Test multiplicative composition for brightness, contrast, gamma, saturation, vibrance."""
        v1 = ColorValues(brightness=1.5, contrast=2.0, gamma=0.8, saturation=1.2, vibrance=1.1)
        v2 = ColorValues(brightness=0.5, contrast=0.5, gamma=1.25, saturation=0.8, vibrance=0.9)
        result = v1 + v2

        assert abs(result.brightness - 0.75) < 1e-6  # 1.5 * 0.5
        assert abs(result.contrast - 1.0) < 1e-6  # 2.0 * 0.5
        assert abs(result.gamma - 1.0) < 1e-6  # 0.8 * 1.25
        assert abs(result.saturation - 0.96) < 1e-6  # 1.2 * 0.8
        assert abs(result.vibrance - 0.99) < 1e-6  # 1.1 * 0.9

    def test_additive_fields(self):
        """Test additive composition for temperature, shadows, highlights."""
        v1 = ColorValues(temperature=0.3, shadows=0.2, highlights=-0.1)
        v2 = ColorValues(temperature=-0.1, shadows=0.3, highlights=0.4)
        result = v1 + v2

        assert abs(result.temperature - 0.2) < 1e-6  # 0.3 + -0.1
        assert abs(result.shadows - 0.5) < 1e-6  # 0.2 + 0.3
        assert abs(result.highlights - 0.3) < 1e-6  # -0.1 + 0.4

    def test_hue_shift_wrapping(self):
        """Test hue_shift additive with wrap to [-180, 180]."""
        # Normal addition
        v1 = ColorValues(hue_shift=30)
        v2 = ColorValues(hue_shift=45)
        result = v1 + v2
        assert abs(result.hue_shift - 75) < 1e-6

        # Positive wrap
        v1 = ColorValues(hue_shift=170)
        v2 = ColorValues(hue_shift=30)
        result = v1 + v2
        assert abs(result.hue_shift - (-160)) < 1e-6  # 200 -> -160

        # Negative wrap
        v1 = ColorValues(hue_shift=-170)
        v2 = ColorValues(hue_shift=-30)
        result = v1 + v2
        assert abs(result.hue_shift - 160) < 1e-6  # -200 -> 160

        # Full circle
        v1 = ColorValues(hue_shift=180)
        v2 = ColorValues(hue_shift=180)
        result = v1 + v2
        assert abs(result.hue_shift - 0) < 1e-6  # 360 -> 0

    def test_neutral_values_identity(self):
        """Test that neutral values act as identity in merge."""
        original = ColorValues(brightness=1.5, temperature=0.3, hue_shift=45)
        neutral = ColorValues()

        result = original + neutral
        assert result.brightness == original.brightness
        assert result.temperature == original.temperature
        assert result.hue_shift == original.hue_shift

        result = neutral + original
        assert result.brightness == original.brightness
        assert result.temperature == original.temperature
        assert result.hue_shift == original.hue_shift

    def test_is_neutral(self):
        """Test is_neutral detection."""
        assert ColorValues().is_neutral()
        assert not ColorValues(brightness=1.1).is_neutral()
        assert not ColorValues(temperature=0.1).is_neutral()
        assert not ColorValues(hue_shift=1).is_neutral()

    def test_clamp(self):
        """Test clamping to valid ranges."""
        extreme = ColorValues(
            brightness=10.0,
            contrast=-1.0,
            gamma=0.01,
            saturation=100.0,
            vibrance=0.0,
            temperature=5.0,
            shadows=-10.0,
            highlights=2.0,
            hue_shift=500.0,
        )
        clamped = extreme.clamp()

        assert clamped.brightness == 5.0
        assert clamped.contrast == 0.0
        assert clamped.gamma == 0.1
        assert clamped.saturation == 5.0
        assert clamped.vibrance == 0.0
        assert clamped.temperature == 1.0
        assert clamped.shadows == -1.0
        assert clamped.highlights == 1.0
        assert clamped.hue_shift == 180.0

    def test_radd_sum_support(self):
        """Test __radd__ for sum() with initial value 0."""
        presets = [
            ColorValues(brightness=1.2),
            ColorValues(brightness=1.5),
            ColorValues(temperature=0.3),
        ]
        result = sum(presets)

        assert abs(result.brightness - 1.8) < 1e-6  # 1.2 * 1.5
        assert abs(result.temperature - 0.3) < 1e-6

    def test_chain_multiple_merges(self):
        """Test chaining multiple merges."""
        warm = ColorValues(temperature=0.3)
        bright = ColorValues(brightness=1.3)
        saturated = ColorValues(saturation=1.2)
        shifted = ColorValues(hue_shift=30)

        result = warm + bright + saturated + shifted

        assert result.temperature == 0.3
        assert result.brightness == 1.3
        assert result.saturation == 1.2
        assert result.hue_shift == 30

    def test_not_implemented_for_wrong_type(self):
        """Test that adding wrong types returns NotImplemented."""
        v = ColorValues()
        result = v.__add__("not a ColorValues")
        assert result is NotImplemented


class TestFilterValuesMerge:
    """Test FilterValues merge operations with stricter-wins logic."""

    def test_min_opacity_max_wins(self):
        """Test min_opacity uses max (higher threshold is stricter)."""
        f1 = FilterValues(min_opacity=0.3)
        f2 = FilterValues(min_opacity=0.5)
        result = f1 + f2
        assert result.min_opacity == 0.5

        result = f2 + f1
        assert result.min_opacity == 0.5

    def test_max_scale_min_wins(self):
        """Test max_scale uses min (lower cap is stricter)."""
        f1 = FilterValues(max_scale=5.0)
        f2 = FilterValues(max_scale=2.0)
        result = f1 + f2
        assert result.max_scale == 2.0

        result = f2 + f1
        assert result.max_scale == 2.0

    def test_sphere_radius_min_wins(self):
        """Test sphere_radius uses min (smaller region is stricter)."""
        f1 = FilterValues(sphere_radius=100.0)
        f2 = FilterValues(sphere_radius=50.0)
        result = f1 + f2
        assert result.sphere_radius == 50.0

    def test_combined_filters(self):
        """Test combining multiple filter criteria."""
        f1 = FilterValues(min_opacity=0.3, max_scale=10.0, sphere_radius=100.0)
        f2 = FilterValues(min_opacity=0.5, max_scale=5.0, sphere_radius=200.0)
        result = f1 + f2

        assert result.min_opacity == 0.5  # stricter
        assert result.max_scale == 5.0  # stricter
        assert result.sphere_radius == 100.0  # stricter

    def test_neutral_values_identity(self):
        """Test that neutral values preserve other filter."""
        strict = FilterValues(min_opacity=0.5, max_scale=2.0, sphere_radius=10.0)
        neutral = FilterValues()

        result = strict + neutral
        assert result.min_opacity == 0.5
        assert result.max_scale == 2.0
        assert result.sphere_radius == 10.0

    def test_is_neutral(self):
        """Test is_neutral detection."""
        assert FilterValues().is_neutral()
        assert not FilterValues(min_opacity=0.1).is_neutral()
        assert not FilterValues(max_scale=50.0).is_neutral()
        assert not FilterValues(sphere_radius=500.0).is_neutral()

    def test_clamp(self):
        """Test clamping to valid ranges."""
        extreme = FilterValues(
            min_opacity=2.0,
            max_scale=-10.0,
            sphere_radius=5000.0,
        )
        clamped = extreme.clamp()

        assert clamped.min_opacity == 1.0
        assert clamped.max_scale == 0.0
        assert clamped.sphere_radius == 5000.0  # sphere_radius is not upper-clamped

    def test_radd_sum_support(self):
        """Test __radd__ for sum() with initial value 0."""
        filters = [
            FilterValues(min_opacity=0.1),
            FilterValues(min_opacity=0.3),
            FilterValues(min_opacity=0.2),
        ]
        result = sum(filters)
        assert result.min_opacity == 0.3  # max of all

    def test_commutative(self):
        """Test that filter merge is commutative."""
        f1 = FilterValues(min_opacity=0.3, max_scale=5.0)
        f2 = FilterValues(min_opacity=0.5, max_scale=10.0)

        result1 = f1 + f2
        result2 = f2 + f1

        assert result1.min_opacity == result2.min_opacity
        assert result1.max_scale == result2.max_scale

    def test_not_implemented_for_wrong_type(self):
        """Test that adding wrong types returns NotImplemented."""
        f = FilterValues()
        result = f.__add__("not a FilterValues")
        assert result is NotImplemented


class TestTransformValuesMerge:
    """Test TransformValues merge operations with matrix composition."""

    def test_scale_composition(self):
        """Test scale composes multiplicatively."""
        t1 = TransformValues.from_scale(2.0)
        t2 = TransformValues.from_scale(3.0)
        result = t1 + t2

        assert abs(result.scale - 6.0) < 1e-5

    def test_translation_composition(self):
        """Test translation composes additively when no rotation."""
        t1 = TransformValues.from_translation(1, 0, 0)
        t2 = TransformValues.from_translation(0, 2, 0)
        result = t1 + t2

        assert abs(result.translation[0] - 1) < 1e-5
        assert abs(result.translation[1] - 2) < 1e-5
        assert abs(result.translation[2] - 0) < 1e-5

    def test_translate_then_scale(self):
        """Test non-commutative: translate then scale."""
        translate = TransformValues.from_translation(1, 0, 0)
        scale = TransformValues.from_scale(2.0)

        # translate + scale: point moves 1, then everything scales 2x
        result = translate + scale

        # Translation gets scaled by the following scale operation
        assert abs(result.translation[0] - 2.0) < 1e-5
        assert abs(result.scale - 2.0) < 1e-5

    def test_scale_then_translate(self):
        """Test non-commutative: scale then translate."""
        scale = TransformValues.from_scale(2.0)
        translate = TransformValues.from_translation(1, 0, 0)

        # scale + translate: point scales 2x, then moves 1
        result = scale + translate

        # Translation is applied after scaling, so translation is not scaled
        assert abs(result.translation[0] - 1.0) < 1e-5
        assert abs(result.scale - 2.0) < 1e-5

    def test_rotation_euler(self):
        """Test rotation from Euler angles."""
        rotate = TransformValues.from_rotation_euler(0, 0, 90)

        # Apply to a point
        M = rotate.to_matrix()
        point = np.array([1, 0, 0, 1], dtype=np.float32)
        rotated = M @ point

        # 90 degree Z rotation: (1,0,0) -> (0,1,0)
        assert abs(rotated[0] - 0) < 1e-5
        assert abs(rotated[1] - 1) < 1e-5
        assert abs(rotated[2] - 0) < 1e-5

    def test_rotation_axis_angle(self):
        """Test rotation from axis-angle."""
        rotate = TransformValues.from_rotation_axis_angle((0, 0, 1), 90)

        M = rotate.to_matrix()
        point = np.array([1, 0, 0, 1], dtype=np.float32)
        rotated = M @ point

        # 90 degree Z rotation: (1,0,0) -> (0,1,0)
        assert abs(rotated[0] - 0) < 1e-5
        assert abs(rotated[1] - 1) < 1e-5
        assert abs(rotated[2] - 0) < 1e-5

    def test_rotation_composition(self):
        """Test rotation composition is correct."""
        # Rotate 45 degrees then another 45 degrees = 90 degrees
        r1 = TransformValues.from_rotation_euler(0, 0, 45)
        r2 = TransformValues.from_rotation_euler(0, 0, 45)
        result = r1 + r2

        M = result.to_matrix()
        point = np.array([1, 0, 0, 1], dtype=np.float32)
        rotated = M @ point

        # Should be approximately 90 degree rotation
        assert abs(rotated[0] - 0) < 1e-4
        assert abs(rotated[1] - 1) < 1e-4

    def test_non_commutative(self):
        """Test that transform composition is non-commutative."""
        translate = TransformValues.from_translation(1, 0, 0)
        rotate = TransformValues.from_rotation_euler(0, 0, 90)

        # translate then rotate
        tr = translate + rotate

        # rotate then translate
        rt = rotate + translate

        # These should be different
        assert tr.translation != rt.translation or tr.rotation != rt.rotation

    def test_matrix_round_trip(self):
        """Test to_matrix and from_matrix preserve values."""
        original = TransformValues(
            scale=2.0,
            rotation=(0.7071067811865476, 0.0, 0.0, 0.7071067811865476),  # 90 deg Z
            translation=(1.0, 2.0, 3.0),
        )

        M = original.to_matrix()
        recovered = TransformValues.from_matrix(M)

        assert abs(recovered.scale - original.scale) < 1e-4
        assert abs(recovered.translation[0] - original.translation[0]) < 1e-4
        assert abs(recovered.translation[1] - original.translation[1]) < 1e-4
        assert abs(recovered.translation[2] - original.translation[2]) < 1e-4

    def test_is_neutral(self):
        """Test is_neutral detection."""
        assert TransformValues().is_neutral()
        assert not TransformValues.from_scale(2.0).is_neutral()
        assert not TransformValues.from_translation(1, 0, 0).is_neutral()
        assert not TransformValues.from_rotation_euler(0, 0, 45).is_neutral()

    def test_radd_sum_support(self):
        """Test __radd__ for sum() with initial value 0."""
        transforms = [
            TransformValues.from_translation(1, 0, 0),
            TransformValues.from_translation(0, 1, 0),
            TransformValues.from_translation(0, 0, 1),
        ]
        result = sum(transforms)

        assert abs(result.translation[0] - 1) < 1e-5
        assert abs(result.translation[1] - 1) < 1e-5
        assert abs(result.translation[2] - 1) < 1e-5

    def test_chain_multiple_transforms(self):
        """Test chaining multiple transforms."""
        scale = TransformValues.from_scale(2.0)
        translate = TransformValues.from_translation(1, 0, 0)
        rotate = TransformValues.from_rotation_euler(0, 0, 90)

        result = scale + translate + rotate

        # Apply to origin
        M = result.to_matrix()
        point = np.array([0, 0, 0, 1], dtype=np.float32)
        transformed = M @ point

        # Origin after scale+translate+rotate: scale doesn't affect origin,
        # translate moves to (1,0,0), rotate 90 deg Z makes it (0,1,0)
        assert abs(transformed[0] - 0) < 1e-4
        assert abs(transformed[1] - 1) < 1e-4

    def test_not_implemented_for_wrong_type(self):
        """Test that adding wrong types returns NotImplemented."""
        t = TransformValues()
        result = t.__add__("not a TransformValues")
        assert result is NotImplemented

    def test_zero_axis_fallback(self):
        """Test axis-angle with zero axis defaults to Z axis."""
        # Zero axis should default to Z
        rotate = TransformValues.from_rotation_axis_angle((0, 0, 0), 90)

        M = rotate.to_matrix()
        point = np.array([1, 0, 0, 1], dtype=np.float32)
        rotated = M @ point

        # Should rotate around Z axis
        assert abs(rotated[0] - 0) < 1e-5
        assert abs(rotated[1] - 1) < 1e-5


class TestCrossValueTypeErrors:
    """Test that mixing value types raises appropriate errors."""

    def test_color_filter_mix(self):
        """Test ColorValues + FilterValues returns NotImplemented."""
        c = ColorValues(brightness=1.2)
        f = FilterValues(min_opacity=0.3)

        result = c.__add__(f)
        assert result is NotImplemented

    def test_filter_transform_mix(self):
        """Test FilterValues + TransformValues returns NotImplemented."""
        f = FilterValues(min_opacity=0.3)
        t = TransformValues.from_scale(2.0)

        result = f.__add__(t)
        assert result is NotImplemented

    def test_transform_color_mix(self):
        """Test TransformValues + ColorValues returns NotImplemented."""
        t = TransformValues.from_scale(2.0)
        c = ColorValues(brightness=1.2)

        result = t.__add__(c)
        assert result is NotImplemented


class TestUnitAliases:
    """Test unit conversion factory methods and aliases."""

    def test_temperature_kelvin_neutral(self):
        """Test 6500K is neutral temperature."""
        cv = ColorValues.from_k(6500)
        assert abs(cv.temperature - 0.0) < 1e-6

    def test_temperature_kelvin_warm(self):
        """Test warm temperatures (< 6500K) are positive."""
        cv = ColorValues.from_k(2000)
        assert abs(cv.temperature - 1.0) < 1e-6

        cv = ColorValues.from_k(4250)  # midpoint
        assert abs(cv.temperature - 0.5) < 1e-6

    def test_temperature_kelvin_cool(self):
        """Test cool temperatures (> 6500K) are negative."""
        cv = ColorValues.from_k(11000)
        assert abs(cv.temperature - (-1.0)) < 1e-6

        cv = ColorValues.from_k(8750)  # midpoint
        assert abs(cv.temperature - (-0.5)) < 1e-6

    def test_temperature_kelvin_clamp(self):
        """Test extreme Kelvin values are clamped."""
        # Very cold
        cv = ColorValues.from_k(20000)
        assert cv.temperature == -1.0

        # Very warm
        cv = ColorValues.from_k(0)
        assert cv.temperature == 1.0

    def test_temperature_to_kelvin_round_trip(self):
        """Test temperature to Kelvin conversion round-trips."""
        for kelvin in [2000, 4000, 6500, 8000, 11000]:
            cv = ColorValues.from_k(kelvin)
            recovered = cv.to_k()
            assert abs(recovered - kelvin) < 1e-3

    def test_hue_shift_rad(self):
        """Test hue shift from radians."""
        import math

        # pi/2 radians = 90 degrees
        cv = ColorValues.from_rad(math.pi / 2)
        assert abs(cv.hue_shift - 90) < 1e-4

        # pi radians = 180 degrees (or -180)
        cv = ColorValues.from_rad(math.pi)
        assert abs(cv.hue_shift) == 180

        # -pi/2 radians = -90 degrees
        cv = ColorValues.from_rad(-math.pi / 2)
        assert abs(cv.hue_shift - (-90)) < 1e-4

    def test_hue_shift_rad_wrap(self):
        """Test hue shift from radians wraps correctly."""
        import math

        # 2*pi should wrap to 0
        cv = ColorValues.from_rad(2 * math.pi)
        assert abs(cv.hue_shift - 0) < 1e-4

        # 3*pi/2 should wrap to -90
        cv = ColorValues.from_rad(3 * math.pi / 2)
        assert abs(cv.hue_shift - (-90)) < 1e-4

    def test_hue_shift_to_rad_round_trip(self):
        """Test hue shift to radians conversion round-trips."""
        import math

        for degrees in [-180, -90, 0, 45, 90, 180]:
            cv = ColorValues(hue_shift=degrees)
            radians = cv.to_rad()
            expected = math.radians(degrees)
            assert abs(radians - expected) < 1e-6

    def test_rotation_euler_rad(self):
        """Test Euler rotation from radians."""
        import math

        # pi/2 radians = 90 degrees around Z
        rotate = TransformValues.from_euler_rad(0, 0, math.pi / 2)

        M = rotate.to_matrix()
        point = np.array([1, 0, 0, 1], dtype=np.float32)
        rotated = M @ point

        # Should rotate (1,0,0) to (0,1,0)
        assert abs(rotated[0] - 0) < 1e-5
        assert abs(rotated[1] - 1) < 1e-5

    def test_rotation_euler_rad_vs_degrees(self):
        """Test radian and degree versions produce same result."""
        import math

        angle_deg = 45
        angle_rad = math.radians(angle_deg)

        from_deg = TransformValues.from_rotation_euler(0, 0, angle_deg)
        from_rad = TransformValues.from_euler_rad(0, 0, angle_rad)

        # Quaternions should match
        for i in range(4):
            assert abs(from_deg.rotation[i] - from_rad.rotation[i]) < 1e-5

    def test_rotation_axis_angle_rad(self):
        """Test axis-angle rotation from radians."""
        import math

        # pi/2 radians around Z axis
        rotate = TransformValues.from_axis_angle_rad((0, 0, 1), math.pi / 2)

        M = rotate.to_matrix()
        point = np.array([1, 0, 0, 1], dtype=np.float32)
        rotated = M @ point

        # Should rotate (1,0,0) to (0,1,0)
        assert abs(rotated[0] - 0) < 1e-5
        assert abs(rotated[1] - 1) < 1e-5

    def test_rotation_axis_angle_rad_vs_degrees(self):
        """Test radian and degree versions produce same result."""
        import math

        angle_deg = 60
        angle_rad = math.radians(angle_deg)
        axis = (1, 1, 0)

        from_deg = TransformValues.from_rotation_axis_angle(axis, angle_deg)
        from_rad = TransformValues.from_axis_angle_rad(axis, angle_rad)

        # Quaternions should match
        for i in range(4):
            assert abs(from_deg.rotation[i] - from_rad.rotation[i]) < 1e-5


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_color_extreme_multiplication(self):
        """Test extreme multiplicative values."""
        v1 = ColorValues(brightness=5.0)
        v2 = ColorValues(brightness=5.0)
        result = v1 + v2

        # 25.0 is out of range but merge doesn't auto-clamp
        assert result.brightness == 25.0

        # Clamp brings it back
        clamped = result.clamp()
        assert clamped.brightness == 5.0

    def test_filter_all_zero_passthrough(self):
        """Test that permissive filters pass through."""
        # Use float('inf') for sphere_radius as it's the new neutral value
        permissive = FilterValues(min_opacity=0.0, max_scale=100.0, sphere_radius=float('inf'))
        assert permissive.is_neutral()

    def test_transform_identity_composition(self):
        """Test identity transform composition."""
        identity = TransformValues()
        some_transform = TransformValues.from_scale(2.0)

        result = identity + some_transform
        assert abs(result.scale - 2.0) < 1e-5

        result = some_transform + identity
        assert abs(result.scale - 2.0) < 1e-5

    def test_hue_boundary_values(self):
        """Test hue at exact boundary values."""
        v1 = ColorValues(hue_shift=180)
        v2 = ColorValues(hue_shift=0)
        result = v1 + v2
        # 180 and -180 are equivalent angles
        assert abs(result.hue_shift) == 180

        v1 = ColorValues(hue_shift=-180)
        v2 = ColorValues(hue_shift=0)
        result = v1 + v2
        assert abs(result.hue_shift) == 180

    def test_multiple_empty_sum(self):
        """Test summing empty list with each value type."""
        # Need to provide start value since sum([]) returns 0
        assert sum([ColorValues()], ColorValues()).is_neutral()
        assert sum([FilterValues()], FilterValues()).is_neutral()
        assert sum([TransformValues()], TransformValues()).is_neutral()
