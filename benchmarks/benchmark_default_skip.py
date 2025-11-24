"""
Benchmark default value skipping optimizations.

Tests performance when operations are at default values.
"""

import time

import numpy as np
from gsply import GSData

from gsmod import Color

# Create test data
N = 1_000_000
rng = np.random.default_rng(42)
quats = rng.random((N, 4), dtype=np.float32)
quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

data = GSData(
    means=rng.random((N, 3), dtype=np.float32),
    scales=rng.random((N, 3), dtype=np.float32) * 0.1,
    quats=quats,
    sh0=rng.random((N, 3), dtype=np.float32),
    opacities=rng.random(N, dtype=np.float32),
    shN=None,
)

print("=" * 80)
print("BENCHMARK: Default Value Skip Optimization")
print(f"Testing with {N:,} Gaussians")
print("=" * 80)


def benchmark(name, pipeline, n_iterations=200):
    """Benchmark a pipeline with robust statistics."""
    # Warmup - more iterations for JIT stability
    for _ in range(20):
        _ = pipeline(data.copy(), inplace=True)

    # Benchmark - more iterations and discard outliers
    times = []
    for _ in range(n_iterations):
        test_data = data.copy()
        start = time.perf_counter()
        _ = pipeline(test_data, inplace=True)
        times.append((time.perf_counter() - start) * 1000)

    # Remove outliers (top/bottom 10%)
    times_sorted = np.sort(times)
    trim_count = int(len(times_sorted) * 0.1)
    if trim_count > 0:
        times_trimmed = times_sorted[trim_count:-trim_count]
    else:
        times_trimmed = times_sorted

    # Use median for robustness against outliers
    median_ms = np.median(times_trimmed)
    mean_ms = np.mean(times_trimmed)
    std_ms = np.std(times_trimmed)
    throughput = N / (median_ms / 1000) / 1e6

    print(f"\n{name}:")
    print(f"  Median:     {median_ms:.3f} ms")
    print(f"  Mean:       {mean_ms:.3f} ms +/- {std_ms:.3f} ms")
    print(f"  Throughput: {throughput:.1f} M/sec")
    return median_ms


# Test 1: Only Phase 1 operations (LUT only)
print("\n" + "=" * 80)
print("TEST 1: Phase 1 Only (brightness, contrast)")
print("=" * 80)

pipeline_phase1 = Color().brightness(1.2).contrast(1.1).compile()
t1 = benchmark("Phase 1 only (all Phase 2 at defaults)", pipeline_phase1)

# Test 2: Phase 1 + single Phase 2 operation at default
print("\n" + "=" * 80)
print("TEST 2: Phase 1 + Phase 2 (with defaults)")
print("=" * 80)

pipeline_with_defaults = Color().brightness(1.2).contrast(1.1).saturation(1.0).compile()
t2 = benchmark("Phase 1 + saturation(1.0) [default]", pipeline_with_defaults)

# Test 3: Phase 1 + single Phase 2 operation NOT at default
print("\n" + "=" * 80)
print("TEST 3: Phase 1 + Phase 2 (active)")
print("=" * 80)

pipeline_active = Color().brightness(1.2).contrast(1.1).saturation(1.3).compile()
t3 = benchmark("Phase 1 + saturation(1.3) [active]", pipeline_active)

# Test 4: All Phase 2 at defaults
print("\n" + "=" * 80)
print("TEST 4: All Phase 2 Operations at Defaults")
print("=" * 80)

pipeline_all_defaults = (
    Color()
    .brightness(1.2)
    .contrast(1.1)
    .saturation(1.0)
    .vibrance(1.0)
    .hue_shift(0.0)
    .shadows(1.0)
    .highlights(1.0)
    .compile()
)
t4 = benchmark("All Phase 2 at defaults", pipeline_all_defaults)

# Test 5: Some Phase 2 operations active
print("\n" + "=" * 80)
print("TEST 5: Mixed Phase 2 Operations")
print("=" * 80)

pipeline_mixed = (
    Color()
    .brightness(1.2)
    .contrast(1.1)
    .saturation(1.3)
    .vibrance(1.0)
    .hue_shift(0.0)
    .shadows(1.0)
    .highlights(1.0)
    .compile()
)
t5 = benchmark("One Phase 2 active, others at defaults", pipeline_mixed)

# Test 6: All Phase 2 active
pipeline_all_active = (
    Color()
    .brightness(1.2)
    .contrast(1.1)
    .saturation(1.3)
    .vibrance(1.1)
    .hue_shift(15)
    .shadows(1.1)
    .highlights(0.9)
    .compile()
)
t6 = benchmark("All Phase 2 active", pipeline_all_active)

# Analysis
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print("\nDefault Skip Benefit:")
print(f"  Phase 1 only:                {t1:.3f} ms (baseline)")
print(f"  + saturation(1.0) [default]: {t2:.3f} ms ({((t2 - t1) / t1 * 100):+.1f}% vs baseline)")
print(f"  + saturation(1.3) [active]:  {t3:.3f} ms ({((t3 - t1) / t1 * 100):+.1f}% vs baseline)")

print("\nFull Phase 2 Comparison:")
print(f"  All defaults:        {t4:.3f} ms (should be ~same as Phase 1 only)")
print(f"  One active:          {t5:.3f} ms")
print(f"  All active:          {t6:.3f} ms")

print("\nOptimization Effectiveness:")
overhead_defaults = (t4 - t1) / t1 * 100
overhead_active = (t6 - t1) / t1 * 100
print(f"  Overhead with all defaults: {overhead_defaults:+.1f}%")
print(f"  Overhead with all active:   {overhead_active:+.1f}%")

# Success if overhead is <10% (allowing for measurement noise)
if abs(overhead_defaults) < 10:
    print("\n[OK] SUCCESS: Default skip optimization working! (<10% overhead)")
    print("     Both Phase 1 only and all-defaults use the same fast path kernel.")
else:
    print(f"\n[WARNING] Default skip has {overhead_defaults:.1f}% overhead (expected <10%)")
    print("     This may indicate JIT compilation or memory allocation overhead.")

print("\n" + "=" * 80)
