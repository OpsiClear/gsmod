"""Comprehensive validation of all CPU/GPU pathways and data transfer.

This test module provides thorough validation of:
1. Forward/backward data transfer (CPU -> GPU -> CPU roundtrip)
2. All color operations on both CPU and GPU with equivalence check
3. All transform operations on both CPU and GPU with equivalence check
4. All filter operations on both CPU and GPU with equivalence check
5. Format state preservation across transfers
"""

import numpy as np
import pytest
import torch
from gsply import GSData
from gsply.gsdata import DataFormat

from gsmod import Color, ColorValues, FilterValues, GSDataPro, Transform, TransformValues
from gsmod.torch import ColorGPU, FilterGPU, GSTensorPro, TransformGPU


def create_test_data(n_gaussians=1000, seed=42):
    """Create reproducible test data with proper format initialization."""
    np.random.seed(seed)

    data = GSData(
        means=np.random.randn(n_gaussians, 3).astype(np.float32) * 2,
        scales=np.random.rand(n_gaussians, 3).astype(np.float32) * 0.5 + 0.1,
        quats=np.random.randn(n_gaussians, 4).astype(np.float32),
        opacities=np.random.rand(n_gaussians).astype(np.float32),
        sh0=np.random.rand(n_gaussians, 3).astype(np.float32),
        shN=None,
    )

    # Normalize quaternions
    norms = np.linalg.norm(data.quats, axis=1, keepdims=True)
    data.quats /= norms

    return data


class TestDataTransferRoundtrip:
    """Test CPU -> GPU -> CPU data transfer preserves data integrity."""

    def test_gsdata_to_gstensor_to_gsdata(self):
        """Test GSData -> GSTensorPro -> GSData roundtrip."""
        original = create_test_data()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Forward: CPU -> GPU
        gstensor = GSTensorPro.from_gsdata(original.copy(), device=device)

        # Backward: GPU -> CPU
        result = gstensor.to_gsdata()

        # Verify all fields preserved
        np.testing.assert_allclose(
            original.means, result.means, rtol=1e-6, atol=1e-7, err_msg="means differ"
        )
        np.testing.assert_allclose(
            original.scales, result.scales, rtol=1e-6, atol=1e-7, err_msg="scales differ"
        )
        np.testing.assert_allclose(
            original.quats, result.quats, rtol=1e-6, atol=1e-7, err_msg="quats differ"
        )
        np.testing.assert_allclose(
            original.opacities,
            result.opacities.flatten(),
            rtol=1e-6,
            atol=1e-7,
            err_msg="opacities differ",
        )
        np.testing.assert_allclose(
            original.sh0, result.sh0, rtol=1e-6, atol=1e-7, err_msg="sh0 differ"
        )

    def test_gsdatapro_to_gstensorpro_to_gsdatapro(self):
        """Test GSDataPro -> GSTensorPro -> GSDataPro roundtrip."""
        original = GSDataPro.from_gsdata(create_test_data())
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Forward: CPU -> GPU
        gstensor = GSTensorPro.from_gsdata(original.copy(), device=device)

        # Backward: GPU -> CPU
        result = gstensor.to_cpu()

        # Verify all fields preserved
        np.testing.assert_allclose(
            original.means, result.means, rtol=1e-6, atol=1e-7, err_msg="means differ"
        )
        np.testing.assert_allclose(
            original.scales, result.scales, rtol=1e-6, atol=1e-7, err_msg="scales differ"
        )

    def test_format_preserved_through_transfer(self):
        """Test that format state is preserved through CPU<->GPU transfer."""
        data = create_test_data()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set known format state
        if not hasattr(data, "_format"):
            data._format = {}
        data._format["sh0"] = DataFormat.SH0_RGB

        # Transfer to GPU
        gstensor = GSTensorPro.from_gsdata(data, device=device)
        assert gstensor._format.get("sh0") == DataFormat.SH0_RGB

        # Transfer back to CPU
        result = gstensor.to_gsdata()
        assert result._format.get("sh0") == DataFormat.SH0_RGB

    def test_multiple_roundtrips(self):
        """Test multiple CPU<->GPU roundtrips maintain data integrity."""
        original = create_test_data()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        data = original.copy()
        for _ in range(3):
            # Forward
            gstensor = GSTensorPro.from_gsdata(data, device=device)
            # Backward
            data = gstensor.to_gsdata()

        # Verify still matches original
        np.testing.assert_allclose(
            original.means,
            data.means,
            rtol=1e-5,
            atol=1e-6,
            err_msg="means differ after roundtrips",
        )


class TestColorPathwayEquivalence:
    """Test all color operations produce equivalent results on CPU and GPU."""

    @pytest.fixture
    def test_data(self):
        return create_test_data()

    def test_brightness_pathway(self, test_data):
        """Test brightness adjustment pathway."""
        factor = 1.2
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # CPU pathway
        cpu_data = GSDataPro.from_gsdata(test_data.copy())
        cpu_data.color(ColorValues(brightness=factor))

        # GPU pathway
        gpu_data = GSTensorPro.from_gsdata(test_data.copy(), device=device)
        gpu_data.color(ColorValues(brightness=factor))
        gpu_result = gpu_data.to_gsdata()

        np.testing.assert_allclose(
            cpu_data.sh0, gpu_result.sh0, rtol=2e-3, atol=2e-3, err_msg="brightness differs"
        )

    def test_contrast_pathway(self, test_data):
        """Test contrast adjustment pathway."""
        factor = 1.1
        device = "cuda" if torch.cuda.is_available() else "cpu"

        cpu_data = GSDataPro.from_gsdata(test_data.copy())
        cpu_data.color(ColorValues(contrast=factor))

        gpu_data = GSTensorPro.from_gsdata(test_data.copy(), device=device)
        gpu_data.color(ColorValues(contrast=factor))
        gpu_result = gpu_data.to_gsdata()

        np.testing.assert_allclose(
            cpu_data.sh0, gpu_result.sh0, rtol=2e-3, atol=2e-3, err_msg="contrast differs"
        )

    def test_saturation_pathway(self, test_data):
        """Test saturation adjustment pathway."""
        factor = 1.3
        device = "cuda" if torch.cuda.is_available() else "cpu"

        cpu_data = GSDataPro.from_gsdata(test_data.copy())
        cpu_data.color(ColorValues(saturation=factor))

        gpu_data = GSTensorPro.from_gsdata(test_data.copy(), device=device)
        gpu_data.color(ColorValues(saturation=factor))
        gpu_result = gpu_data.to_gsdata()

        np.testing.assert_allclose(
            cpu_data.sh0, gpu_result.sh0, rtol=2e-3, atol=2e-3, err_msg="saturation differs"
        )

    def test_gamma_pathway(self, test_data):
        """Test gamma correction pathway."""
        gamma = 0.9
        device = "cuda" if torch.cuda.is_available() else "cpu"

        cpu_data = GSDataPro.from_gsdata(test_data.copy())
        cpu_data.color(ColorValues(gamma=gamma))

        gpu_data = GSTensorPro.from_gsdata(test_data.copy(), device=device)
        gpu_data.color(ColorValues(gamma=gamma))
        gpu_result = gpu_data.to_gsdata()

        np.testing.assert_allclose(
            cpu_data.sh0, gpu_result.sh0, rtol=2e-3, atol=2e-3, err_msg="gamma differs"
        )

    def test_temperature_pathway(self, test_data):
        """Test temperature adjustment pathway."""
        temp = 0.2
        device = "cuda" if torch.cuda.is_available() else "cpu"

        cpu_data = GSDataPro.from_gsdata(test_data.copy())
        cpu_data.color(ColorValues(temperature=temp))

        gpu_data = GSTensorPro.from_gsdata(test_data.copy(), device=device)
        gpu_data.color(ColorValues(temperature=temp))
        gpu_result = gpu_data.to_gsdata()

        np.testing.assert_allclose(
            cpu_data.sh0, gpu_result.sh0, rtol=2e-3, atol=2e-3, err_msg="temperature differs"
        )

    def test_combined_color_pathway(self, test_data):
        """Test combined color operations pathway."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        values = ColorValues(brightness=1.1, contrast=1.05, saturation=1.2, gamma=0.95)

        cpu_data = GSDataPro.from_gsdata(test_data.copy())
        cpu_data.color(values)

        gpu_data = GSTensorPro.from_gsdata(test_data.copy(), device=device)
        gpu_data.color(values)
        gpu_result = gpu_data.to_gsdata()

        np.testing.assert_allclose(
            cpu_data.sh0, gpu_result.sh0, rtol=2e-3, atol=2e-3, err_msg="combined color differs"
        )


class TestTransformPathwayEquivalence:
    """Test all transform operations produce equivalent results on CPU and GPU."""

    @pytest.fixture
    def test_data(self):
        return create_test_data()

    def test_translation_pathway(self, test_data):
        """Test translation pathway."""
        translation = (1.0, 2.0, 3.0)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        cpu_data = GSDataPro.from_gsdata(test_data.copy())
        cpu_data.transform(TransformValues.from_translation(*translation))

        gpu_data = GSTensorPro.from_gsdata(test_data.copy(), device=device)
        gpu_data.transform(TransformValues.from_translation(*translation))
        gpu_result = gpu_data.to_gsdata()

        np.testing.assert_allclose(
            cpu_data.means, gpu_result.means, rtol=1e-6, atol=1e-7, err_msg="translation differs"
        )

    def test_scale_pathway(self, test_data):
        """Test scale pathway."""
        scale = 2.0
        device = "cuda" if torch.cuda.is_available() else "cpu"

        cpu_data = GSDataPro.from_gsdata(test_data.copy())
        cpu_data.transform(TransformValues.from_scale(scale))

        gpu_data = GSTensorPro.from_gsdata(test_data.copy(), device=device)
        gpu_data.transform(TransformValues.from_scale(scale))
        gpu_result = gpu_data.to_gsdata()

        np.testing.assert_allclose(
            cpu_data.means, gpu_result.means, rtol=1e-6, atol=1e-7, err_msg="scale differs"
        )
        np.testing.assert_allclose(
            cpu_data.scales, gpu_result.scales, rtol=1e-6, atol=1e-7, err_msg="scales differ"
        )

    def test_rotation_euler_pathway(self, test_data):
        """Test Euler rotation pathway."""
        euler = (0.0, 45.0, 0.0)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        cpu_data = GSDataPro.from_gsdata(test_data.copy())
        cpu_data.transform(TransformValues.from_rotation_euler(*euler))

        gpu_data = GSTensorPro.from_gsdata(test_data.copy(), device=device)
        gpu_data.transform(TransformValues.from_rotation_euler(*euler))
        gpu_result = gpu_data.to_gsdata()

        np.testing.assert_allclose(
            cpu_data.means, gpu_result.means, rtol=1e-4, atol=1e-5, err_msg="rotation means differ"
        )
        np.testing.assert_allclose(
            cpu_data.quats, gpu_result.quats, rtol=1e-4, atol=1e-5, err_msg="rotation quats differ"
        )

    def test_combined_transform_pathway(self, test_data):
        """Test combined transform operations pathway."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        values = (
            TransformValues.from_translation(1.0, 0.0, 0.0)
            + TransformValues.from_rotation_euler(0.0, 30.0, 0.0)
            + TransformValues.from_scale(1.5)
        )

        cpu_data = GSDataPro.from_gsdata(test_data.copy())
        cpu_data.transform(values)

        gpu_data = GSTensorPro.from_gsdata(test_data.copy(), device=device)
        gpu_data.transform(values)
        gpu_result = gpu_data.to_gsdata()

        np.testing.assert_allclose(
            cpu_data.means,
            gpu_result.means,
            rtol=1e-4,
            atol=1e-5,
            err_msg="combined transform means differ",
        )
        np.testing.assert_allclose(
            cpu_data.quats,
            gpu_result.quats,
            rtol=1e-4,
            atol=1e-5,
            err_msg="combined transform quats differ",
        )


class TestFilterPathwayEquivalence:
    """Test all filter operations produce equivalent results on CPU and GPU."""

    @pytest.fixture
    def test_data(self):
        return create_test_data(n_gaussians=5000)

    def test_opacity_filter_pathway(self, test_data):
        """Test opacity filter pathway."""
        threshold = 0.5
        device = "cuda" if torch.cuda.is_available() else "cpu"

        cpu_data = GSDataPro.from_gsdata(test_data.copy())
        cpu_data.filter(FilterValues(min_opacity=threshold))

        gpu_data = GSTensorPro.from_gsdata(test_data.copy(), device=device)
        gpu_data.filter(FilterValues(min_opacity=threshold))
        gpu_result = gpu_data.to_gsdata()

        assert len(cpu_data) == len(gpu_result), "Filter counts differ"
        np.testing.assert_allclose(
            cpu_data.means, gpu_result.means, rtol=1e-6, atol=1e-7, err_msg="filtered means differ"
        )

    def test_scale_filter_pathway(self, test_data):
        """Test scale filter pathway."""
        threshold = 0.4
        device = "cuda" if torch.cuda.is_available() else "cpu"

        cpu_data = GSDataPro.from_gsdata(test_data.copy())
        cpu_data.filter(FilterValues(max_scale=threshold))

        gpu_data = GSTensorPro.from_gsdata(test_data.copy(), device=device)
        gpu_data.filter(FilterValues(max_scale=threshold))
        gpu_result = gpu_data.to_gsdata()

        assert len(cpu_data) == len(gpu_result), "Filter counts differ"

    def test_sphere_filter_pathway(self, test_data):
        """Test sphere filter pathway."""
        radius = 6.0
        device = "cuda" if torch.cuda.is_available() else "cpu"

        cpu_data = GSDataPro.from_gsdata(test_data.copy())
        cpu_data.filter(FilterValues(sphere_radius=radius))

        gpu_data = GSTensorPro.from_gsdata(test_data.copy(), device=device)
        gpu_data.filter(FilterValues(sphere_radius=radius))
        gpu_result = gpu_data.to_gsdata()

        assert len(cpu_data) == len(gpu_result), "Sphere filter counts differ"
        np.testing.assert_allclose(
            cpu_data.means, gpu_result.means, rtol=1e-6, atol=1e-7, err_msg="sphere filter differs"
        )

    def test_box_filter_pathway(self, test_data):
        """Test box filter pathway."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        values = FilterValues(box_min=(-2.0, -2.0, -2.0), box_max=(2.0, 2.0, 2.0))

        cpu_data = GSDataPro.from_gsdata(test_data.copy())
        cpu_data.filter(values)

        gpu_data = GSTensorPro.from_gsdata(test_data.copy(), device=device)
        gpu_data.filter(values)
        gpu_result = gpu_data.to_gsdata()

        assert len(cpu_data) == len(gpu_result), "Box filter counts differ"

    def test_rotated_box_filter_pathway(self, test_data):
        """Test rotated box filter pathway."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        values = FilterValues(
            box_min=(-2.0, -2.0, -2.0),
            box_max=(2.0, 2.0, 2.0),
            box_rot=(0.0, np.pi / 4, 0.0),
        )

        cpu_data = GSDataPro.from_gsdata(test_data.copy())
        cpu_data.filter(values)

        gpu_data = GSTensorPro.from_gsdata(test_data.copy(), device=device)
        gpu_data.filter(values)
        gpu_result = gpu_data.to_gsdata()

        assert len(cpu_data) == len(gpu_result), "Rotated box filter counts differ"

    def test_ellipsoid_filter_pathway(self, test_data):
        """Test ellipsoid filter pathway."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        values = FilterValues(
            ellipsoid_center=(0.0, 0.0, 0.0),
            ellipsoid_radii=(3.0, 2.0, 2.0),
            ellipsoid_rot=(0.0, 0.0, np.pi / 6),
        )

        cpu_data = GSDataPro.from_gsdata(test_data.copy())
        cpu_data.filter(values)

        gpu_data = GSTensorPro.from_gsdata(test_data.copy(), device=device)
        gpu_data.filter(values)
        gpu_result = gpu_data.to_gsdata()

        assert len(cpu_data) == len(gpu_result), "Ellipsoid filter counts differ"

    def test_frustum_filter_pathway(self, test_data):
        """Test frustum filter pathway."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        values = FilterValues(
            frustum_pos=(0.0, 0.0, 8.0),
            frustum_rot=None,
            frustum_fov=1.047,
            frustum_aspect=1.0,
            frustum_near=0.1,
            frustum_far=20.0,
        )

        cpu_data = GSDataPro.from_gsdata(test_data.copy())
        cpu_data.filter(values)

        gpu_data = GSTensorPro.from_gsdata(test_data.copy(), device=device)
        gpu_data.filter(values)
        gpu_result = gpu_data.to_gsdata()

        assert len(cpu_data) == len(gpu_result), "Frustum filter counts differ"

    def test_combined_filter_pathway(self, test_data):
        """Test combined filter pathway."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        values = FilterValues(min_opacity=0.3, max_scale=0.4, sphere_radius=6.0)

        cpu_data = GSDataPro.from_gsdata(test_data.copy())
        cpu_data.filter(values)

        gpu_data = GSTensorPro.from_gsdata(test_data.copy(), device=device)
        gpu_data.filter(values)
        gpu_result = gpu_data.to_gsdata()

        assert len(cpu_data) == len(gpu_result), "Combined filter counts differ"


class TestFullPipelinePathway:
    """Test full pipeline with all operation types."""

    def test_filter_transform_color_pipeline(self):
        """Test full pipeline: filter -> transform -> color."""
        data = create_test_data(n_gaussians=5000)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # CPU pipeline
        cpu_data = GSDataPro.from_gsdata(data.copy())
        cpu_data.filter(FilterValues(min_opacity=0.2, sphere_radius=6.0))
        cpu_data.transform(TransformValues.from_translation(0.5, 0.0, 0.0))
        cpu_data.color(ColorValues(brightness=1.1, saturation=1.2))

        # GPU pipeline
        gpu_data = GSTensorPro.from_gsdata(data.copy(), device=device)
        gpu_data.filter(FilterValues(min_opacity=0.2, sphere_radius=6.0))
        gpu_data.transform(TransformValues.from_translation(0.5, 0.0, 0.0))
        gpu_data.color(ColorValues(brightness=1.1, saturation=1.2))
        gpu_result = gpu_data.to_gsdata()

        assert len(cpu_data) == len(gpu_result), "Full pipeline counts differ"

        if len(cpu_data) > 0:
            np.testing.assert_allclose(
                cpu_data.means,
                gpu_result.means,
                rtol=1e-4,
                atol=1e-5,
                err_msg="Full pipeline means differ",
            )
            np.testing.assert_allclose(
                cpu_data.sh0,
                gpu_result.sh0,
                rtol=2e-3,
                atol=2e-3,
                err_msg="Full pipeline colors differ",
            )


class TestAdvancedPipelineAPI:
    """Test advanced pipeline APIs (Color, Transform, Filter classes)."""

    def test_color_pipeline_class(self):
        """Test Color pipeline class equivalence."""
        data = create_test_data()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # CPU Color class
        cpu_pipeline = Color().brightness(1.2).saturation(1.3).gamma(0.9)
        cpu_result = cpu_pipeline(data.copy(), inplace=True)

        # GPU ColorGPU class
        gpu_data = GSTensorPro.from_gsdata(data.copy(), device=device)
        gpu_pipeline = ColorGPU().brightness(1.2).saturation(1.3).gamma(0.9)
        gpu_result = gpu_pipeline(gpu_data, inplace=True).to_gsdata()

        np.testing.assert_allclose(
            cpu_result.sh0,
            gpu_result.sh0,
            rtol=2e-3,
            atol=2e-3,
            err_msg="Color pipeline class differs",
        )

    def test_transform_pipeline_class(self):
        """Test Transform pipeline class equivalence."""
        data = create_test_data()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # CPU Transform class
        cpu_pipeline = Transform().translate([1, 0, 0]).scale(2.0)
        cpu_result = cpu_pipeline(data.copy(), inplace=True)

        # GPU TransformGPU class
        gpu_data = GSTensorPro.from_gsdata(data.copy(), device=device)
        gpu_pipeline = TransformGPU().translate([1, 0, 0]).scale(2.0)
        gpu_result = gpu_pipeline(gpu_data, inplace=True).to_gsdata()

        np.testing.assert_allclose(
            cpu_result.means,
            gpu_result.means,
            rtol=1e-5,
            atol=1e-6,
            err_msg="Transform pipeline class means differ",
        )

    def test_filter_pipeline_class(self):
        """Test FilterGPU pipeline class."""
        data = create_test_data(n_gaussians=5000)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # CPU
        cpu_data = GSDataPro.from_gsdata(data.copy())
        cpu_data.filter(FilterValues(sphere_radius=6.0, min_opacity=0.3))

        # GPU FilterGPU class
        gpu_data = GSTensorPro.from_gsdata(data.copy(), device=device)
        gpu_pipeline = FilterGPU().within_sphere(radius=6.0).min_opacity(0.3)
        mask = gpu_pipeline.compute_mask(gpu_data)
        gpu_result = gpu_data[mask].to_gsdata()

        assert len(cpu_data) == len(gpu_result), "Filter pipeline class counts differ"


def run_validation():
    """Run all validation tests and print summary."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE PATHWAY VALIDATION")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    test_classes = [
        TestDataTransferRoundtrip(),
        TestColorPathwayEquivalence(),
        TestTransformPathwayEquivalence(),
        TestFilterPathwayEquivalence(),
        TestFullPipelinePathway(),
        TestAdvancedPipelineAPI(),
    ]

    passed = []
    failed = []

    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n[{class_name}]")

        test_methods = [m for m in dir(test_class) if m.startswith("test_")]

        for method_name in test_methods:
            method = getattr(test_class, method_name)
            test_name = f"{class_name}.{method_name}"

            try:
                # Handle fixtures by passing test_data directly
                if "test_data" in method_name or hasattr(method, "__code__"):
                    if "test_data" in method.__code__.co_varnames:
                        method(
                            create_test_data(
                                n_gaussians=5000 if "filter" in method_name.lower() else 1000
                            )
                        )
                    else:
                        method()
                else:
                    method()
                print(f"  [OK] {method_name}")
                passed.append(test_name)
            except Exception as e:
                print(f"  [FAIL] {method_name}: {str(e)[:80]}")
                failed.append((test_name, str(e)))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed tests:")
        for test_name, error in failed:
            print(f"  - {test_name}")
            print(f"    {error[:150]}")

    return len(failed) == 0


if __name__ == "__main__":
    success = run_validation()
    exit(0 if success else 1)
