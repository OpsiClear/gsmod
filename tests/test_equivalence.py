"""Test equivalence between CPU and GPU operations."""

import numpy as np
import pytest
import torch
from gsply import GSData
from gsply.gsdata import DataFormat

from gsmod import Color, Transform, GSDataPro, ColorValues, FilterValues, TransformValues
from gsmod.torch import ColorGPU, FilterGPU, GSTensorPro, PipelineGPU, TransformGPU
from gsmod.verification import FormatVerifier


def create_test_data(n_gaussians=1000, seed=42, format_rgb=True):
    """Create reproducible test data with proper format initialization.

    :param n_gaussians: Number of Gaussians to create
    :param seed: Random seed for reproducibility
    :param format_rgb: If True, create data in RGB format (default for color tests)
    """
    np.random.seed(seed)

    data = GSData(
        means=np.random.randn(n_gaussians, 3).astype(np.float32) * 2,
        scales=np.random.rand(n_gaussians, 3).astype(np.float32) * 0.5 + 0.1,  # Positive scales
        quats=np.random.randn(n_gaussians, 4).astype(np.float32),
        opacities=np.random.rand(n_gaussians).astype(np.float32),
        sh0=np.random.rand(n_gaussians, 3).astype(np.float32),  # Values in [0, 1]
        shN=None,  # SH0 only for simplicity
    )

    # Normalize quaternions
    norms = np.linalg.norm(data.quats, axis=1, keepdims=True)
    data.quats /= norms

    # Initialize format tracking
    # For color tests, use RGB format since sh0 values are in [0, 1] range
    # This ensures both CPU and GPU see the same input format
    if not hasattr(data, '_format'):
        data._format = {}

    if format_rgb:
        # sh0 values [0, 1] are already RGB format
        data._format['sh0'] = DataFormat.SH0_RGB
    else:
        # Treat as SH format (for testing conversion)
        data._format['sh0'] = DataFormat.SH0_SH

    return data


class TestColorEquivalence:
    """Test color operations produce equivalent results between CPU and GPU."""

    def test_brightness(self):
        """Test brightness adjustment equivalence."""
        data = create_test_data()
        factor = 1.2

        # CPU operation
        cpu_result = Color().brightness(factor)(data.copy(), inplace=True)

        # GPU operation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gstensor = GSTensorPro.from_gsdata(data.copy(), device=device)
        gpu_result = ColorGPU().brightness(factor)(gstensor, inplace=True)
        gpu_data = gpu_result.to_gsdata()

        # Verify equivalence with format check
        # Tolerances account for CPU LUT vs GPU direct calculation (~0.001)
        FormatVerifier.assert_equivalent(
            cpu_result,
            gpu_data,
            rtol=2e-3,
            atol=2e-3,
            check_all_fields=False,  # Only check sh0 for color tests
        )

    def test_contrast(self):
        """Test contrast adjustment equivalence."""
        data = create_test_data()
        factor = 1.1

        # CPU operation
        cpu_result = Color().contrast(factor)(data.copy(), inplace=True)

        # GPU operation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gstensor = GSTensorPro.from_gsdata(data.copy(), device=device)
        gpu_result = ColorGPU().contrast(factor)(gstensor, inplace=True)
        gpu_data = gpu_result.to_gsdata()

        # Verify equivalence - tolerance for CPU LUT vs GPU direct calculation (~0.001)
        FormatVerifier.assert_equivalent(
            cpu_result,
            gpu_data,
            rtol=2e-3,
            atol=2e-3,
            check_all_fields=False,
        )

    def test_saturation(self):
        """Test saturation adjustment equivalence."""
        data = create_test_data()
        factor = 1.3

        # CPU operation
        cpu_result = Color().saturation(factor)(data.copy(), inplace=True)

        # GPU operation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gstensor = GSTensorPro.from_gsdata(data.copy(), device=device)
        gpu_result = ColorGPU().saturation(factor)(gstensor, inplace=True)
        gpu_data = gpu_result.to_gsdata()

        # Verify equivalence - tolerance for CPU LUT vs GPU direct calculation (~0.001)
        FormatVerifier.assert_equivalent(
            cpu_result,
            gpu_data,
            rtol=2e-3,
            atol=2e-3,
            check_all_fields=False,
        )

    def test_combined_color_pipeline(self):
        """Test combined color pipeline equivalence."""
        data = create_test_data()

        # CPU pipeline
        cpu_pipeline = Color().brightness(1.1).contrast(1.05).saturation(1.2).gamma(0.95)
        cpu_result = cpu_pipeline(data.copy(), inplace=True)

        # GPU pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gstensor = GSTensorPro.from_gsdata(data.copy(), device=device)
        gpu_pipeline = ColorGPU().brightness(1.1).contrast(1.05).saturation(1.2).gamma(0.95)
        gpu_result = gpu_pipeline(gstensor, inplace=True)
        gpu_data = gpu_result.to_gsdata()

        # Verify equivalence - tolerance for CPU LUT vs GPU direct calculation (~0.001)
        FormatVerifier.assert_equivalent(
            cpu_result,
            gpu_data,
            rtol=2e-3,
            atol=2e-3,
            check_all_fields=False,
        )


class TestTransformEquivalence:
    """Test transform operations produce equivalent results."""

    def test_translation(self):
        """Test translation equivalence."""
        data = create_test_data()
        translation = [1.0, 2.0, 3.0]

        # CPU operation
        cpu_result = Transform().translate(translation)(data.copy(), inplace=True)

        # GPU operation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gstensor = GSTensorPro.from_gsdata(data.copy(), device=device)
        gpu_result = TransformGPU().translate(translation)(gstensor, inplace=True)
        gpu_data = gpu_result.to_gsdata()

        # Verify equivalence
        np.testing.assert_allclose(
            cpu_result.means,
            gpu_data.means,
            rtol=1e-6,
            atol=1e-7,
            err_msg="Translation results differ between CPU and GPU",
        )

    def test_uniform_scale(self):
        """Test uniform scale equivalence."""
        data = create_test_data()
        scale_factor = 2.0

        # CPU operation
        cpu_result = Transform().scale(scale_factor)(data.copy(), inplace=True)

        # GPU operation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gstensor = GSTensorPro.from_gsdata(data.copy(), device=device)
        gpu_result = TransformGPU().scale(scale_factor)(gstensor, inplace=True)
        gpu_data = gpu_result.to_gsdata()

        # Compare positions
        np.testing.assert_allclose(
            cpu_result.means,
            gpu_data.means,
            rtol=1e-6,
            atol=1e-7,
            err_msg="Scaled positions differ between CPU and GPU",
        )

        # Compare scales
        np.testing.assert_allclose(
            cpu_result.scales,
            gpu_data.scales,
            rtol=1e-6,
            atol=1e-7,
            err_msg="Scaled scales differ between CPU and GPU",
        )

    def test_rotation_axis_angle(self):
        """Test axis-angle rotation equivalence."""
        data = create_test_data()
        axis = [0.0, 1.0, 0.0]  # Y-axis rotation
        angle = np.pi / 4  # 45 degrees

        # CPU operation
        cpu_result = Transform().rotate_axis_angle(axis, angle)(data.copy(), inplace=True)

        # GPU operation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gstensor = GSTensorPro.from_gsdata(data.copy(), device=device)
        gpu_result = TransformGPU().rotate_axis_angle(axis, angle)(gstensor, inplace=True)
        gpu_data = gpu_result.to_gsdata()

        # Compare positions (rotated)
        np.testing.assert_allclose(
            cpu_result.means,
            gpu_data.means,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Rotated positions differ between CPU and GPU",
        )

        # Compare quaternions
        np.testing.assert_allclose(
            cpu_result.quats,
            gpu_data.quats,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Rotated quaternions differ between CPU and GPU",
        )

    def test_combined_transform_pipeline(self):
        """Test combined transform pipeline equivalence."""
        data = create_test_data()

        # CPU pipeline
        cpu_pipeline = (
            Transform().translate([1, 0, 0]).rotate_axis_angle([0, 1, 0], np.pi / 6).scale(1.5)
        )
        cpu_result = cpu_pipeline(data.copy(), inplace=True)

        # GPU pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gstensor = GSTensorPro.from_gsdata(data.copy(), device=device)
        gpu_pipeline = (
            TransformGPU().translate([1, 0, 0]).rotate_axis_angle([0, 1, 0], np.pi / 6).scale(1.5)
        )
        gpu_result = gpu_pipeline(gstensor, inplace=True)
        gpu_data = gpu_result.to_gsdata()

        # Compare results
        np.testing.assert_allclose(
            cpu_result.means,
            gpu_data.means,
            rtol=1e-4,
            atol=1e-5,
            err_msg="Combined transform pipeline positions differ",
        )

        np.testing.assert_allclose(
            cpu_result.quats,
            gpu_data.quats,
            rtol=1e-4,
            atol=1e-5,
            err_msg="Combined transform pipeline quaternions differ",
        )


class TestFilterEquivalence:
    """Test filter operations produce equivalent results."""

    def test_sphere_filter(self):
        """Test spherical filter equivalence."""
        data = create_test_data(n_gaussians=5000)
        radius = 6.0  # Absolute radius in world units (3 sigma to capture most points)

        # CPU operation
        cpu_data = GSDataPro.from_gsdata(data.copy())
        cpu_data.filter(FilterValues(sphere_radius=radius))

        # GPU operation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gstensor = GSTensorPro.from_gsdata(data.copy(), device=device)
        gpu_pipeline = FilterGPU().within_sphere(radius=radius)
        mask = gpu_pipeline.compute_mask(gstensor)
        gpu_result = gstensor[mask]
        gpu_data = gpu_result.to_gsdata()

        # Compare lengths first
        assert len(cpu_data) == len(gpu_data), (
            f"Different number of filtered points: CPU={len(cpu_data)}, GPU={len(gpu_data)}"
        )

        # Compare filtered positions
        np.testing.assert_allclose(
            cpu_data.means,
            gpu_data.means,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Filtered positions differ between CPU and GPU",
        )

    def test_opacity_filter(self):
        """Test opacity filter equivalence."""
        data = create_test_data(n_gaussians=5000)
        threshold = 0.5

        # CPU operation
        cpu_data = GSDataPro.from_gsdata(data.copy())
        cpu_data.filter(FilterValues(min_opacity=threshold))

        # GPU operation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gstensor = GSTensorPro.from_gsdata(data.copy(), device=device)
        gpu_pipeline = FilterGPU().min_opacity(threshold)
        mask = gpu_pipeline.compute_mask(gstensor)
        gpu_result = gstensor[mask]
        gpu_data = gpu_result.to_gsdata()

        # Compare lengths
        assert len(cpu_data) == len(gpu_data), (
            f"Different number of filtered points: CPU={len(cpu_data)}, GPU={len(gpu_data)}"
        )

        # Compare filtered opacities
        np.testing.assert_allclose(
            cpu_data.opacities,
            gpu_data.opacities,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Filtered opacities differ between CPU and GPU",
        )

    def test_combined_filter_pipeline(self):
        """Test combined filter pipeline equivalence."""
        data = create_test_data(n_gaussians=5000)

        # CPU pipeline - radius=6.0 captures most points (3 sigma)
        cpu_data = GSDataPro.from_gsdata(data.copy())
        cpu_data.filter(FilterValues(sphere_radius=6.0, min_opacity=0.3, max_scale=0.4))

        # GPU pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gstensor = GSTensorPro.from_gsdata(data.copy(), device=device)
        gpu_pipeline = FilterGPU().within_sphere(radius=6.0).min_opacity(0.3).max_scale(0.4)
        mask = gpu_pipeline.compute_mask(gstensor)
        gpu_result = gstensor[mask]
        gpu_data = gpu_result.to_gsdata()

        # Compare lengths
        assert len(cpu_data) == len(gpu_data), (
            f"Different number of filtered points: CPU={len(cpu_data)}, GPU={len(gpu_data)}"
        )

        # If we have any results, compare them
        if len(cpu_data) > 0:
            np.testing.assert_allclose(
                cpu_data.means,
                gpu_data.means,
                rtol=1e-5,
                atol=1e-6,
                err_msg="Combined filter results differ between CPU and GPU",
            )


class TestFullPipelineEquivalence:
    """Test full pipeline produces equivalent results."""

    def test_full_pipeline(self):
        """Test full pipeline with all operation types."""
        data = create_test_data(n_gaussians=5000)

        # CPU pipeline using GSDataPro (filter -> transform -> color order)
        # radius=6.0 captures most points (3 sigma for data with std=2)
        cpu_result = GSDataPro.from_gsdata(data.copy())
        cpu_result.filter(FilterValues(sphere_radius=6.0, min_opacity=0.2))
        cpu_result = (
            Transform()
            .translate([0.5, 0, 0])
            .rotate_axis_angle([0, 0, 1], np.pi / 8)
        )(cpu_result, inplace=True)
        cpu_result = Color().brightness(1.1).saturation(1.2)(cpu_result, inplace=True)

        # GPU pipeline
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gstensor = GSTensorPro.from_gsdata(data.copy(), device=device)
        gpu_pipeline = (
            PipelineGPU()
            .within_sphere(radius=6.0)
            .min_opacity(0.2)
            .translate([0.5, 0, 0])
            .rotate_axis_angle([0, 0, 1], np.pi / 8)
            .brightness(1.1)
            .saturation(1.2)
        )
        gpu_result = gpu_pipeline(gstensor, inplace=True)
        gpu_data = gpu_result.to_gsdata()

        # Compare lengths
        assert len(cpu_result) == len(gpu_data), (
            f"Different output sizes: CPU={len(cpu_result)}, GPU={len(gpu_data)}"
        )

        # Compare all fields with appropriate tolerances
        if len(cpu_result) > 0:
            # Verify format first
            FormatVerifier.assert_same_format(cpu_result, gpu_data)

            # Positions
            np.testing.assert_allclose(
                cpu_result.means,
                gpu_data.means,
                rtol=1e-4,
                atol=1e-5,
                err_msg="Full pipeline positions differ",
            )

            # Colors - tolerance for CPU LUT vs GPU direct calculation (~0.001)
            np.testing.assert_allclose(
                cpu_result.sh0,
                gpu_data.sh0,
                rtol=2e-3,
                atol=2e-3,
                err_msg="Full pipeline colors differ",
            )

            # Quaternions
            np.testing.assert_allclose(
                cpu_result.quats,
                gpu_data.quats,
                rtol=1e-4,
                atol=1e-5,
                err_msg="Full pipeline quaternions differ",
            )

            # Scales
            np.testing.assert_allclose(
                cpu_result.scales,
                gpu_data.scales,
                rtol=1e-5,
                atol=1e-6,
                err_msg="Full pipeline scales differ",
            )


class TestFormatTracking:
    """Test format is correctly tracked through pipelines."""

    def test_color_outputs_rgb(self):
        """Test that color pipeline outputs RGB format by default."""
        data = create_test_data()

        # CPU
        cpu_result = Color().brightness(1.2)(data.copy(), inplace=True)
        assert FormatVerifier.is_rgb(cpu_result), "CPU Color should output RGB format"

        # GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gstensor = GSTensorPro.from_gsdata(data.copy(), device=device)
        gpu_result = ColorGPU().brightness(1.2)(gstensor, inplace=True)
        gpu_data = gpu_result.to_gsdata()
        assert FormatVerifier.is_rgb(gpu_data), "GPU ColorGPU should output RGB format"

    def test_format_restore(self):
        """Test that restore_format=True restores original format."""
        # Create data in SH format so restoration can be tested
        data = create_test_data(format_rgb=False)

        # CPU with restore
        cpu_result = Color().brightness(1.2)(data.copy(), inplace=True, restore_format=True)
        assert FormatVerifier.is_sh(cpu_result), "CPU Color with restore_format should output SH"

        # GPU with restore
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gstensor = GSTensorPro.from_gsdata(data.copy(), device=device)
        gpu_result = ColorGPU().brightness(1.2)(gstensor, inplace=True, restore_format=True)
        gpu_data = gpu_result.to_gsdata()
        assert FormatVerifier.is_sh(gpu_data), "GPU ColorGPU with restore_format should output SH"


def run_equivalence_tests():
    """Run all equivalence tests and print results."""
    print("\n" + "=" * 60)
    print("CPU vs GPU EQUIVALENCE TESTS")
    print("=" * 60)

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Track results
    passed = []
    failed = []

    # Test classes
    test_classes = [
        TestColorEquivalence(),
        TestTransformEquivalence(),
        TestFilterEquivalence(),
        TestFullPipelineEquivalence(),
        TestFormatTracking(),
    ]

    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n[{class_name}]")

        # Get all test methods
        test_methods = [m for m in dir(test_class) if m.startswith("test_")]

        for method_name in test_methods:
            method = getattr(test_class, method_name)
            test_name = f"{class_name}.{method_name}"

            try:
                method()
                print(f"  [OK] {method_name}")
                passed.append(test_name)
            except AssertionError as e:
                print(f"  [FAIL] {method_name}: {str(e)[:100]}")
                failed.append((test_name, str(e)))
            except Exception as e:
                print(f"  [ERROR] {method_name}: {e}")
                failed.append((test_name, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed tests:")
        for test_name, error in failed:
            print(f"  - {test_name}")
            print(f"    {error[:200]}")

    # Return success status
    return len(failed) == 0


if __name__ == "__main__":
    success = run_equivalence_tests()
    exit(0 if success else 1)
