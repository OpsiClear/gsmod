"""
Benchmark new color operations: vibrance, hue_shift, and opacity multiplication.

This script measures performance and identifies optimization opportunities.
"""

import logging
import time
from collections.abc import Callable

import numpy as np
from gsply import GSData

from gsmod import Color, multiply_opacity

logging.basicConfig(level=logging.WARNING)


def create_test_data(n_gaussians: int) -> GSData:
    """Create synthetic GSData for benchmarking."""
    rng = np.random.default_rng(42)
    quats = rng.random((n_gaussians, 4), dtype=np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

    data = GSData(
        means=rng.random((n_gaussians, 3), dtype=np.float32),
        scales=rng.random((n_gaussians, 3), dtype=np.float32) * 0.1,
        quats=quats,
        sh0=rng.random((n_gaussians, 3), dtype=np.float32),
        opacities=rng.random(n_gaussians, dtype=np.float32),
        shN=None,
    )
    return data


def benchmark_operation(
    operation: Callable[[GSData], GSData],
    data: GSData,
    n_iterations: int = 100,
    warmup: int = 5,
) -> dict:
    """
    Benchmark a single operation with warmup and multiple iterations.

    Returns:
        dict with timing statistics
    """
    # Warmup
    for _ in range(warmup):
        _ = operation(data.copy())

    # Benchmark
    times = []
    for _ in range(n_iterations):
        data_copy = data.copy()
        start = time.perf_counter()
        _ = operation(data_copy)
        end = time.perf_counter()
        times.append(end - start)

    times_ms = np.array(times) * 1000
    n_gaussians = len(data)

    return {
        "mean_ms": np.mean(times_ms),
        "std_ms": np.std(times_ms),
        "min_ms": np.min(times_ms),
        "max_ms": np.max(times_ms),
        "median_ms": np.median(times_ms),
        "throughput_M_per_sec": n_gaussians / (np.mean(times_ms) / 1000) / 1e6,
    }


def benchmark_inplace_operation(
    operation: Callable[[GSData], GSData],
    data: GSData,
    n_iterations: int = 100,
    warmup: int = 5,
) -> dict:
    """
    Benchmark inplace operation (reuses same data).

    Returns:
        dict with timing statistics
    """
    # Warmup
    test_data = data.copy()
    for _ in range(warmup):
        _ = operation(test_data)

    # Benchmark
    times = []
    test_data = data.copy()
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = operation(test_data)
        end = time.perf_counter()
        times.append(end - start)

    times_ms = np.array(times) * 1000
    n_gaussians = len(data)

    return {
        "mean_ms": np.mean(times_ms),
        "std_ms": np.std(times_ms),
        "min_ms": np.min(times_ms),
        "max_ms": np.max(times_ms),
        "median_ms": np.median(times_ms),
        "throughput_M_per_sec": n_gaussians / (np.mean(times_ms) / 1000) / 1e6,
    }


def print_results(name: str, results: dict, n_gaussians: int):
    """Print benchmark results in a readable format."""
    print(f"\n{name} ({n_gaussians:,} Gaussians):")
    print(
        f"  Mean:       {results['mean_ms']:.3f} ms  ({results['throughput_M_per_sec']:.1f}M/sec)"
    )
    print(f"  Std Dev:    {results['std_ms']:.3f} ms")
    print(f"  Min:        {results['min_ms']:.3f} ms")
    print(f"  Max:        {results['max_ms']:.3f} ms")
    print(f"  Median:     {results['median_ms']:.3f} ms")


def main():
    """Run comprehensive benchmarks."""
    print("=" * 80)
    print("BENCHMARK: New Color Operations (vibrance, hue_shift, opacity)")
    print("=" * 80)

    # Test sizes
    sizes = [10_000, 100_000, 1_000_000]

    for n_gaussians in sizes:
        print(f"\n{'=' * 80}")
        print(f"DATA SIZE: {n_gaussians:,} Gaussians")
        print(f"{'=' * 80}")

        data = create_test_data(n_gaussians)

        # ====================================================================
        # Baseline: No operations
        # ====================================================================
        baseline = benchmark_operation(lambda d: d, data, n_iterations=100)
        print_results("Baseline (copy only)", baseline, n_gaussians)

        # ====================================================================
        # Individual Operations
        # ====================================================================

        # Vibrance
        def vibrance_op(d):
            return Color().vibrance(1.3)(d, inplace=True)

        vibrance = benchmark_inplace_operation(vibrance_op, data, n_iterations=100)
        print_results("Vibrance (1.3)", vibrance, n_gaussians)

        # Hue Shift
        def hue_shift_op(d):
            return Color().hue_shift(30)(d, inplace=True)

        hue_shift = benchmark_inplace_operation(hue_shift_op, data, n_iterations=100)
        print_results("Hue Shift (30°)", hue_shift, n_gaussians)

        # Opacity multiplication
        def opacity_op(d):
            return multiply_opacity(d, 0.7, inplace=True)

        opacity = benchmark_inplace_operation(opacity_op, data, n_iterations=100)
        print_results("Opacity (0.7x)", opacity, n_gaussians)

        # Saturation (for comparison)
        def saturation_op(d):
            return Color().saturation(1.3)(d, inplace=True)

        saturation = benchmark_inplace_operation(saturation_op, data, n_iterations=100)
        print_results("Saturation (1.3) [baseline]", saturation, n_gaussians)

        # ====================================================================
        # Combined Operations
        # ====================================================================

        # Saturation + Vibrance
        def sat_vib_op(d):
            return Color().saturation(1.2).vibrance(1.3)(d, inplace=True)

        sat_vib = benchmark_inplace_operation(sat_vib_op, data, n_iterations=100)
        print_results("Saturation + Vibrance", sat_vib, n_gaussians)

        # Saturation + Hue Shift
        def sat_hue_op(d):
            return Color().saturation(1.2).hue_shift(15)(d, inplace=True)

        sat_hue = benchmark_inplace_operation(sat_hue_op, data, n_iterations=100)
        print_results("Saturation + Hue Shift", sat_hue, n_gaussians)

        # All Phase 2 operations
        def all_phase2_op(d):
            return (
                Color()
                .saturation(1.2)
                .vibrance(1.1)
                .hue_shift(10)
                .shadows(1.1)
                .highlights(0.9)(d, inplace=True)
            )

        all_phase2 = benchmark_inplace_operation(all_phase2_op, data, n_iterations=100)
        print_results("All Phase 2 (sat+vib+hue+shadows+highlights)", all_phase2, n_gaussians)

        # Full pipeline (Phase 1 + Phase 2)
        def full_pipeline_op(d):
            return (
                Color()
                .brightness(1.2)
                .contrast(1.1)
                .saturation(1.3)
                .vibrance(1.2)
                .hue_shift(15)(d, inplace=True)
            )

        full_pipeline = benchmark_inplace_operation(full_pipeline_op, data, n_iterations=100)
        print_results("Full Pipeline (Phase 1 + Phase 2)", full_pipeline, n_gaussians)

        # ====================================================================
        # Analysis
        # ====================================================================
        print(f"\n{'-' * 80}")
        print("ANALYSIS:")
        print(f"{'-' * 80}")

        # Overhead analysis
        vibrance_overhead = vibrance["mean_ms"] - saturation["mean_ms"]
        hue_overhead = hue_shift["mean_ms"] - saturation["mean_ms"]

        print(
            f"Vibrance overhead vs Saturation: {vibrance_overhead:+.3f} ms ({vibrance_overhead / saturation['mean_ms'] * 100:+.1f}%)"
        )
        print(
            f"Hue Shift overhead vs Saturation: {hue_overhead:+.3f} ms ({hue_overhead / saturation['mean_ms'] * 100:+.1f}%)"
        )

        # Combined operation efficiency
        expected_sat_vib = saturation["mean_ms"] + vibrance["mean_ms"] - baseline["mean_ms"]
        actual_sat_vib = sat_vib["mean_ms"]
        fusion_benefit = (expected_sat_vib - actual_sat_vib) / expected_sat_vib * 100

        print("\nFusion Efficiency (Saturation + Vibrance):")
        print(f"  Expected (separate):  {expected_sat_vib:.3f} ms")
        print(f"  Actual (fused):       {actual_sat_vib:.3f} ms")
        print(f"  Fusion benefit:       {fusion_benefit:.1f}% faster")

    print(f"\n{'=' * 80}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
