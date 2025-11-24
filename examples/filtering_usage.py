"""
Example: Gaussian splat filtering usage.

Demonstrates how to use the gsmod filtering system for:
- Volume filtering (sphere and box)
- Opacity filtering
- Scale filtering
- Combined filtering

Uses the new absolute-value API with FilterValues and config objects.
"""

import logging

import numpy as np
from gsply import GSData

from gsmod import GSDataPro, FilterValues
from gsmod.filter import (
    SphereFilter,
    BoxFilter,
    calculate_recommended_max_scale,
    calculate_scene_bounds,
)
from gsmod.filter.api import apply_geometry_filter

# Configure logging to see filtering statistics
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def generate_sample_gsdata(n: int = 10000) -> GSData:
    """Generate sample GSData for demonstration."""
    rng = np.random.default_rng(42)

    means = rng.standard_normal((n, 3)).astype(np.float32) * 2.0
    quats = rng.standard_normal((n, 4)).astype(np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    scales = rng.random((n, 3)).astype(np.float32) * 2.0
    opacities = rng.random(n).astype(np.float32)
    sh0 = rng.random((n, 3)).astype(np.float32)

    # Add some outliers
    outlier_idx = rng.choice(n, size=int(n * 0.01), replace=False)
    scales[outlier_idx] = 10.0  # Large outlier scales

    return GSData(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        sh0=sh0,
        shN=None,
    )


def example_1_opacity_filtering():
    """Example 1: Simple opacity filtering with FilterValues."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Opacity Filtering with FilterValues")
    print("=" * 70)

    # Generate data
    data = GSDataPro.from_gsdata(generate_sample_gsdata())

    print(f"Original: {len(data)} Gaussians")

    # Remove Gaussians with opacity < 10%
    result = data.filter(FilterValues(min_opacity=0.1), inplace=False)

    print(f"After opacity filter (min 0.1): {len(result)} Gaussians")
    print(f"Removed: {len(data) - len(result)} Gaussians")
    print(f"Kept: {len(result) / len(data) * 100:.1f}%")


def example_2_scale_filtering():
    """Example 2: Scale filtering with auto-threshold."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Scale Filtering with Auto-Threshold")
    print("=" * 70)

    # Generate data
    data = GSDataPro.from_gsdata(generate_sample_gsdata())

    print(f"Original: {len(data)} Gaussians")

    # Calculate recommended threshold (99.5th percentile)
    recommended_threshold = calculate_recommended_max_scale(data.scales)
    print(f"Recommended max_scale threshold: {recommended_threshold:.4f}")

    # Apply scale filtering
    result = data.filter(FilterValues(max_scale=recommended_threshold), inplace=False)

    print(f"After scale filter: {len(result)} Gaussians")
    print(f"Removed: {len(data) - len(result)} outliers")


def example_3_sphere_filtering():
    """Example 3: Sphere volume filtering with absolute radius."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Sphere Volume Filtering (Absolute Radius)")
    print("=" * 70)

    # Generate data
    data = GSDataPro.from_gsdata(generate_sample_gsdata())

    print(f"Original: {len(data)} Gaussians")

    # Calculate scene bounds for reference
    bounds = calculate_scene_bounds(data.means)
    print(f"Scene bounds: min={bounds.min}, max={bounds.max}")
    print(f"Scene center: {bounds.center}")
    print(f"Scene max dimension: {bounds.max_size:.3f}")

    # Keep only Gaussians within radius 3.0 from center (absolute units)
    sphere_radius = 3.0
    result = data.filter(
        FilterValues(
            sphere_center=tuple(bounds.center),
            sphere_radius=sphere_radius,
        ),
        inplace=False
    )

    print(f"After sphere filter (radius={sphere_radius}): {len(result)} Gaussians")
    print(f"Kept: {len(result) / len(data) * 100:.1f}%")


def example_4_box_filtering():
    """Example 4: Box volume filtering with absolute bounds."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Box Volume Filtering (Absolute Bounds)")
    print("=" * 70)

    # Generate data
    data = GSDataPro.from_gsdata(generate_sample_gsdata())

    print(f"Original: {len(data)} Gaussians")

    # Calculate scene bounds for reference
    bounds = calculate_scene_bounds(data.means)

    # Keep only Gaussians within a box from -2 to +2 in all dimensions
    box_min = (-2.0, -2.0, -2.0)
    box_max = (2.0, 2.0, 2.0)
    result = data.filter(
        FilterValues(
            box_min=box_min,
            box_max=box_max,
        ),
        inplace=False
    )

    print(f"After box filter ({box_min} to {box_max}): {len(result)} Gaussians")
    print(f"Kept: {len(result) / len(data) * 100:.1f}%")


def example_5_combined_filtering():
    """Example 5: Combined filtering (volume + opacity + scale)."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Combined Filtering (Volume + Opacity + Scale)")
    print("=" * 70)

    # Generate data
    data = GSDataPro.from_gsdata(generate_sample_gsdata())

    print(f"Original: {len(data)} Gaussians")

    # Calculate scene bounds and recommended scale threshold
    bounds = calculate_scene_bounds(data.means)
    recommended_scale = calculate_recommended_max_scale(data.scales)

    # Combine all filters: sphere + opacity + scale
    result = data.filter(
        FilterValues(
            sphere_center=tuple(bounds.center),
            sphere_radius=4.0,  # Absolute radius in world units
            min_opacity=0.05,   # Remove < 5% opacity
            max_scale=recommended_scale,  # Remove outliers
        ),
        inplace=False
    )

    print(f"After combined filtering: {len(result)} Gaussians")
    print(f"Removed: {len(data) - len(result)} Gaussians")
    print(f"Kept: {len(result) / len(data) * 100:.1f}%")


def example_6_geometry_config():
    """Example 6: Using geometry config objects directly."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Geometry Config Objects (SphereFilter, BoxFilter)")
    print("=" * 70)

    # Generate data
    data = GSDataPro.from_gsdata(generate_sample_gsdata())
    positions = data.means

    print(f"Original: {len(positions)} Gaussians")

    # Use SphereFilter config
    sphere_config = SphereFilter(center=(0, 0, 0), radius=3.0)
    sphere_mask = apply_geometry_filter(positions, sphere_config)
    print(f"Sphere filter (radius=3.0): {sphere_mask.sum()} Gaussians")

    # Use BoxFilter config
    box_config = BoxFilter(center=(0, 0, 0), size=(4, 4, 4))
    box_mask = apply_geometry_filter(positions, box_config)
    print(f"Box filter (size=4x4x4): {box_mask.sum()} Gaussians")

    # Combine with quality filter
    from gsmod.filter import QualityFilter
    quality = QualityFilter(min_opacity=0.1, max_scale=3.0)
    combined_mask = apply_geometry_filter(positions, sphere_config, quality)
    print(f"Sphere + quality filter: {combined_mask.sum()} Gaussians")


def example_7_custom_percentiles():
    """Example 7: Custom percentile for scale threshold."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Custom Percentiles for Scale Threshold")
    print("=" * 70)

    # Generate data
    data = GSDataPro.from_gsdata(generate_sample_gsdata())
    scales = data.scales

    # Try different percentiles
    percentiles = [90.0, 95.0, 99.0, 99.5, 99.9]

    print("Scale thresholds at different percentiles:")
    for p in percentiles:
        threshold = calculate_recommended_max_scale(scales, percentile=p)
        print(f"  {p:5.1f}th percentile: {threshold:.4f}")

    # Use more aggressive filtering (95th percentile)
    aggressive_threshold = calculate_recommended_max_scale(scales, percentile=95.0)
    result = data.filter(
        FilterValues(max_scale=aggressive_threshold),
        inplace=False
    )

    print("\nWith 95th percentile threshold:")
    print(f"  Kept: {len(result)} / {len(data)} Gaussians ({len(result) / len(data) * 100:.1f}%)")


def example_8_method_chaining():
    """Example 8: Method chaining for multiple operations."""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: Method Chaining")
    print("=" * 70)

    # Generate data
    data = GSDataPro.from_gsdata(generate_sample_gsdata())

    print(f"Original: {len(data)} Gaussians")

    # Chain multiple filter operations
    result = (data
        .filter(FilterValues(min_opacity=0.1), inplace=False)
        .filter(FilterValues(max_scale=3.0), inplace=False)
        .filter(FilterValues(sphere_radius=4.0), inplace=False)
    )

    print(f"After chained filters: {len(result)} Gaussians")
    print(f"Kept: {len(result) / len(data) * 100:.1f}%")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("GSPRO FILTERING SYSTEM EXAMPLES")
    print("(Using Absolute Values API)")
    print("=" * 70)

    example_1_opacity_filtering()
    example_2_scale_filtering()
    example_3_sphere_filtering()
    example_4_box_filtering()
    example_5_combined_filtering()
    example_6_geometry_config()
    example_7_custom_percentiles()
    example_8_method_chaining()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
