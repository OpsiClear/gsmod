"""
Benchmark contiguous vs non-contiguous array performance.

Tests the trade-off between:
- Conversion overhead: ~3.2ms for 100K Gaussians
- Performance benefit: 2-45x speedup per operation (depending on operation count)

Helps determine when make_contiguous() is beneficial.

Usage:
    python benchmark_contiguous.py [ply_file_path]
    
If no PLY file is provided, uses synthetic data.
"""

import sys
import time
from pathlib import Path

import numpy as np
import gsply
from gsply import GSData

from gsmod import Color, Filter, Transform

# Test configurations
NUM_ITERATIONS = 100
WARMUP_ITERATIONS = 20

print("=" * 80)
print("CONTIGUOUS vs NON-CONTIGUOUS ARRAY PERFORMANCE BENCHMARK")
print("=" * 80)
print(f"Testing with {NUM_ITERATIONS} iterations per test")
print(f"Warmup: {WARMUP_ITERATIONS} iterations")


def create_non_contiguous_data(n: int) -> GSData:
    """Create GSData with non-contiguous arrays (simulating plyread behavior)."""
    # Create arrays and then make them non-contiguous by column reordering
    # This simulates PLY file loading which can create strided views
    means = np.random.randn(n, 3).astype(np.float32)
    scales = np.random.rand(n, 3).astype(np.float32) * 0.01
    quats = np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32)
    colors = np.random.rand(n, 3).astype(np.float32)
    opacities = np.random.rand(n).astype(np.float32)  # 1D array for filter compatibility

    # Create non-contiguous arrays by column reordering (creates strided view)
    # This simulates PLY I/O behavior where arrays may not be C-contiguous
    non_contig_means = means[:, [0, 1, 2]]  # Column reordering creates non-contiguous
    non_contig_scales = scales[:, [2, 0, 1]]  # Reordered columns
    non_contig_quats = quats[:, [1, 2, 3, 0]]  # Reordered columns
    non_contig_colors = colors[:, [2, 0, 1]]  # Reordered columns

    data = GSData(
        means=non_contig_means,
        scales=non_contig_scales,
        quats=non_contig_quats,
        opacities=opacities,
        sh0=non_contig_colors,
        shN=None,  # No higher-order SH
    )

    return data


def create_contiguous_data(n: int) -> GSData:
    """Create GSData with contiguous arrays."""
    means = np.random.randn(n, 3).astype(np.float32)
    scales = np.random.rand(n, 3).astype(np.float32) * 0.01
    quats = np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32)
    colors = np.random.rand(n, 3).astype(np.float32)
    opacities = np.random.rand(n).astype(np.float32)  # 1D array for filter compatibility

    data = GSData(
        means=np.ascontiguousarray(means),
        scales=np.ascontiguousarray(scales),
        quats=np.ascontiguousarray(quats),
        opacities=np.ascontiguousarray(opacities),
        sh0=np.ascontiguousarray(colors),
        shN=None,  # No higher-order SH
    )

    # Ensure contiguous
    if hasattr(data, "make_contiguous"):
        data.make_contiguous(inplace=True)

    return data


def benchmark_conversion_overhead(data: GSData) -> float:
    """Benchmark the overhead of make_contiguous()."""
    if not hasattr(data, "make_contiguous"):
        return 0.0

    times = []
    for _ in range(NUM_ITERATIONS):
        test_data = data.copy()
        start = time.perf_counter()
        test_data.make_contiguous(inplace=True)
        times.append((time.perf_counter() - start) * 1000)

    return np.mean(times)


def benchmark_color_pipeline(data: GSData, contiguous: bool) -> dict:
    """Benchmark Color pipeline performance."""
    pipeline = (
        Color()
        .temperature(0.6)
        .brightness(1.2)
        .contrast(1.1)
        .saturation(1.3)
        .shadows(1.1)
        .highlights(0.9)
        .compile()
    )

    # Warmup
    warmup_data = data.copy()
    for _ in range(WARMUP_ITERATIONS):
        pipeline(warmup_data, inplace=True)

    # Benchmark
    times = []
    for _ in range(NUM_ITERATIONS):
        test_data = data.copy()
        start = time.perf_counter()
        pipeline(test_data, inplace=True)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def benchmark_transform_pipeline(data: GSData, contiguous: bool) -> dict:
    """Benchmark Transform pipeline performance."""
    pipeline = (
        Transform()
        .translate([1.0, 0.5, -0.5])
        .rotate_euler([0.1, 0.2, 0.3])
        .scale(1.1)
        .compile()
    )

    # Warmup
    warmup_data = data.copy()
    for _ in range(WARMUP_ITERATIONS):
        pipeline(warmup_data, inplace=True, make_contiguous=False)

    # Benchmark
    times = []
    for _ in range(NUM_ITERATIONS):
        test_data = data.copy()
        start = time.perf_counter()
        pipeline(test_data, inplace=True, make_contiguous=False)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def benchmark_filter_pipeline(data: GSData, contiguous: bool) -> dict:
    """Benchmark Filter pipeline performance."""
    pipeline = (
        Filter()
        .within_sphere(radius=0.8)
        .min_opacity(0.1)
        .max_scale(2.5)
        .compile()
    )

    # Warmup
    warmup_data = data.copy()
    for _ in range(WARMUP_ITERATIONS):
        pipeline(warmup_data, inplace=True)

    # Benchmark
    times = []
    for _ in range(NUM_ITERATIONS):
        test_data = data.copy()
        start = time.perf_counter()
        pipeline(test_data, inplace=True)
        times.append((time.perf_counter() - start) * 1000)

    return {
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
    }


# Determine data source
use_real_data = len(sys.argv) > 1 and Path(sys.argv[1]).exists()
if use_real_data:
    ply_path = sys.argv[1]
    print(f"\n{'=' * 80}")
    print(f"Loading real PLY data from: {ply_path}")
    print("=" * 80)
    
    # Load real PLY data (non-contiguous by default)
    print("\nLoading PLY file...")
    real_data = gsply.plyread(ply_path)
    n = len(real_data)
    print(f"Loaded {n:,} Gaussians")
    
    # Check contiguity status
    if hasattr(real_data, "is_contiguous"):
        is_contig = real_data.is_contiguous()
        print(f"  Contiguity status: {'contiguous' if is_contig else 'NON-contiguous'}")
    
    # Use real data as non-contiguous
    non_contig_data = real_data
    
    # Create contiguous version
    print("Creating contiguous version...")
    contig_data = real_data.copy()
    if hasattr(contig_data, "make_contiguous"):
        contig_data.make_contiguous(inplace=True)
        print(f"  Contiguous version created")
else:
    # Use synthetic data for different sizes
    N_GAUSSIANS = [10_000, 100_000, 1_000_000]
    print("\nUsing synthetic data (no PLY file provided)")
    print("Usage: python benchmark_contiguous.py <path_to_ply_file>")
    print()

# Run benchmarks
if use_real_data:
    # Single test with real data
    n = len(non_contig_data)
    print(f"\n{'=' * 80}")
    print(f"Testing with {n:,} Gaussians (real PLY data)")
    print("=" * 80)
    
    # Check contiguity
    if hasattr(non_contig_data, "is_contiguous"):
        non_contig_status = "contiguous" if non_contig_data.is_contiguous() else "NON-contiguous"
        contig_status = "contiguous" if contig_data.is_contiguous() else "NON-contiguous"
        print(f"\nData status:")
        print(f"  Original (PLY): {non_contig_status}")
        print(f"  Contiguous copy: {contig_status}")
    
    # Benchmark conversion overhead
    if hasattr(non_contig_data, "make_contiguous"):
        print("\n1. Conversion Overhead (make_contiguous):")
        conv_time = benchmark_conversion_overhead(non_contig_data)
        print(f"   Mean: {conv_time:.3f} ms")
        print(f"   Overhead: {conv_time:.3f} ms per conversion")
    
    # Benchmark Color pipeline
    print("\n2. Color Pipeline Performance:")
    color_non_contig = benchmark_color_pipeline(non_contig_data, contiguous=False)
    color_contig = benchmark_color_pipeline(contig_data, contiguous=True)
    
    print(f"   Non-contiguous: {color_non_contig['mean']:.3f} ± {color_non_contig['std']:.3f} ms")
    print(f"   Contiguous:     {color_contig['mean']:.3f} ± {color_contig['std']:.3f} ms")
    speedup_color = color_non_contig["mean"] / color_contig["mean"] if color_contig["mean"] > 0 else 1.0
    print(f"   Speedup: {speedup_color:.2f}x")
    if hasattr(non_contig_data, "make_contiguous"):
        if color_non_contig["mean"] > color_contig["mean"]:
            break_even = conv_time / (color_non_contig["mean"] - color_contig["mean"])
            if break_even > 0:
                print(f"   Break-even: {break_even:.1f} operations (conversion pays off)")
    
    # Benchmark Transform pipeline
    print("\n3. Transform Pipeline Performance:")
    transform_non_contig = benchmark_transform_pipeline(non_contig_data, contiguous=False)
    transform_contig = benchmark_transform_pipeline(contig_data, contiguous=True)
    
    print(f"   Non-contiguous: {transform_non_contig['mean']:.3f} ± {transform_non_contig['std']:.3f} ms")
    print(f"   Contiguous:     {transform_contig['mean']:.3f} ± {transform_contig['std']:.3f} ms")
    speedup_transform = (
        transform_non_contig["mean"] / transform_contig["mean"]
        if transform_contig["mean"] > 0
        else 1.0
    )
    print(f"   Speedup: {speedup_transform:.2f}x")
    if hasattr(non_contig_data, "make_contiguous"):
        if transform_non_contig["mean"] > transform_contig["mean"]:
            break_even = conv_time / (transform_non_contig["mean"] - transform_contig["mean"])
            if break_even > 0:
                print(f"   Break-even: {break_even:.1f} operations (conversion pays off)")
    
    # Benchmark Filter pipeline
    print("\n4. Filter Pipeline Performance:")
    filter_non_contig = benchmark_filter_pipeline(non_contig_data, contiguous=False)
    filter_contig = benchmark_filter_pipeline(contig_data, contiguous=True)
    
    print(f"   Non-contiguous: {filter_non_contig['mean']:.3f} ± {filter_non_contig['std']:.3f} ms")
    print(f"   Contiguous:     {filter_contig['mean']:.3f} ± {filter_contig['std']:.3f} ms")
    speedup_filter = filter_non_contig["mean"] / filter_contig["mean"] if filter_contig["mean"] > 0 else 1.0
    print(f"   Speedup: {speedup_filter:.2f}x")
    if hasattr(non_contig_data, "make_contiguous"):
        if filter_non_contig["mean"] > filter_contig["mean"]:
            break_even = conv_time / (filter_non_contig["mean"] - filter_contig["mean"])
            if break_even > 0:
                print(f"   Break-even: {break_even:.1f} operations (conversion pays off)")
    
    # Summary
    print("\n" + "-" * 80)
    print("SUMMARY:")
    if hasattr(non_contig_data, "make_contiguous"):
        print(f"  Conversion overhead: {conv_time:.3f} ms")
    print(f"  Color speedup: {speedup_color:.2f}x")
    print(f"  Transform speedup: {speedup_transform:.2f}x")
    print(f"  Filter speedup: {speedup_filter:.2f}x")
    print("\nRecommendation:")
    if hasattr(non_contig_data, "make_contiguous"):
        if conv_time < color_non_contig["mean"] * 0.1:  # Less than 10% of operation time
            print("  [OK] Conversion overhead is low - consider using make_contiguous()")
        else:
            print("  [SKIP] Conversion overhead is high - only use for 100+ operations")
    else:
        print("  make_contiguous() not available in this gsply version")

else:
    # Original synthetic data benchmarks
    N_GAUSSIANS = [10_000, 100_000, 1_000_000]
    for n in N_GAUSSIANS:
        print(f"\n{'=' * 80}")
        print(f"Testing with {n:,} Gaussians")
        print("=" * 80)

        # Create test data
        print("\nCreating test data...")
        non_contig_data = create_non_contiguous_data(n)
        contig_data = create_contiguous_data(n)

        # Check contiguity
        if hasattr(non_contig_data, "is_contiguous"):
            non_contig_status = "contiguous" if non_contig_data.is_contiguous() else "NON-contiguous"
            contig_status = "contiguous" if contig_data.is_contiguous() else "NON-contiguous"
            print(f"  Non-contiguous data: {non_contig_status}")
            print(f"  Contiguous data: {contig_status}")

        # Benchmark conversion overhead
        if hasattr(non_contig_data, "make_contiguous"):
            print("\n1. Conversion Overhead (make_contiguous):")
            conv_time = benchmark_conversion_overhead(non_contig_data)
            print(f"   Mean: {conv_time:.3f} ms")
            print(f"   Overhead: {conv_time:.3f} ms per conversion")

        # Benchmark Color pipeline
        print("\n2. Color Pipeline Performance:")
        color_non_contig = benchmark_color_pipeline(non_contig_data, contiguous=False)
        color_contig = benchmark_color_pipeline(contig_data, contiguous=True)

        print(f"   Non-contiguous: {color_non_contig['mean']:.3f} ± {color_non_contig['std']:.3f} ms")
        print(f"   Contiguous:     {color_contig['mean']:.3f} ± {color_contig['std']:.3f} ms")
        speedup = color_non_contig["mean"] / color_contig["mean"] if color_contig["mean"] > 0 else 1.0
        print(f"   Speedup: {speedup:.2f}x")
        if hasattr(non_contig_data, "make_contiguous"):
            if color_non_contig["mean"] > color_contig["mean"]:
                break_even = conv_time / (color_non_contig["mean"] - color_contig["mean"])
                if break_even > 0:
                    print(f"   Break-even: {break_even:.1f} operations (conversion pays off)")

        # Benchmark Transform pipeline
        print("\n3. Transform Pipeline Performance:")
        transform_non_contig = benchmark_transform_pipeline(non_contig_data, contiguous=False)
        transform_contig = benchmark_transform_pipeline(contig_data, contiguous=True)

        print(f"   Non-contiguous: {transform_non_contig['mean']:.3f} ± {transform_non_contig['std']:.3f} ms")
        print(f"   Contiguous:     {transform_contig['mean']:.3f} ± {transform_contig['std']:.3f} ms")
        speedup = (
            transform_non_contig["mean"] / transform_contig["mean"]
            if transform_contig["mean"] > 0
            else 1.0
        )
        print(f"   Speedup: {speedup:.2f}x")
        if hasattr(non_contig_data, "make_contiguous"):
            if transform_non_contig["mean"] > transform_contig["mean"]:
                break_even = conv_time / (transform_non_contig["mean"] - transform_contig["mean"])
                if break_even > 0:
                    print(f"   Break-even: {break_even:.1f} operations (conversion pays off)")

        # Benchmark Filter pipeline
        print("\n4. Filter Pipeline Performance:")
        filter_non_contig = benchmark_filter_pipeline(non_contig_data, contiguous=False)
        filter_contig = benchmark_filter_pipeline(contig_data, contiguous=True)

        print(f"   Non-contiguous: {filter_non_contig['mean']:.3f} ± {filter_non_contig['std']:.3f} ms")
        print(f"   Contiguous:     {filter_contig['mean']:.3f} ± {filter_contig['std']:.3f} ms")
        speedup = filter_non_contig["mean"] / filter_contig["mean"] if filter_contig["mean"] > 0 else 1.0
        print(f"   Speedup: {speedup:.2f}x")
        if hasattr(non_contig_data, "make_contiguous"):
            if filter_non_contig["mean"] > filter_contig["mean"]:
                break_even = conv_time / (filter_non_contig["mean"] - filter_contig["mean"])
                if break_even > 0:
                    print(f"   Break-even: {break_even:.1f} operations (conversion pays off)")

        # Summary
        print("\n" + "-" * 80)
        print("SUMMARY:")
        if hasattr(non_contig_data, "make_contiguous"):
            print(f"  Conversion overhead: {conv_time:.3f} ms")
        print(f"  Color speedup: {speedup:.2f}x")
        print(f"  Transform speedup: {speedup:.2f}x")
        print(f"  Filter speedup: {speedup:.2f}x")
        print("\nRecommendation:")
        if hasattr(non_contig_data, "make_contiguous"):
            if conv_time < color_non_contig["mean"] * 0.1:  # Less than 10% of operation time
                print("  [OK] Conversion overhead is low - consider using make_contiguous()")
            else:
                print("  [SKIP] Conversion overhead is high - only use for 100+ operations")
        else:
            print("  make_contiguous() not available in this gsply version")

print("\n" + "=" * 80)
print("Benchmark complete!")
print("=" * 80)

