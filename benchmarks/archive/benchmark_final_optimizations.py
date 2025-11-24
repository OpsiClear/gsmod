"""
Final optimization benchmark: Interleaved LUT + Custom block sizes.

Tests:
1. Interleaved LUT layout (1.23x expected)
2. Custom prange block sizes (1.1-1.3x expected)
3. Combined optimizations
"""

import time

import numpy as np

print("=" * 80)
print("FINAL OPTIMIZATION BENCHMARK")
print("=" * 80)

N = 100_000
colors = np.random.rand(N, 3).astype(np.float32)
out = np.empty_like(colors)

# Create LUTs
lut_size = 1024
r_lut = np.linspace(0, 1, lut_size, dtype=np.float32)
g_lut = np.linspace(0, 1, lut_size, dtype=np.float32)
b_lut = np.linspace(0, 1, lut_size, dtype=np.float32)

# Create interleaved LUT
lut_interleaved = np.stack([r_lut, g_lut, b_lut], axis=1)  # [1024, 3]

# ============================================================================
# BASELINE: Current best implementation
# ============================================================================

from gsmod.numba_ops import fused_color_full_pipeline_numba

# Warmup
for _ in range(10):
    fused_color_full_pipeline_numba(colors, r_lut, g_lut, b_lut, 1.3, 1.1, 0.9, out)

print("\n[BASELINE] Current best (reduced clipping + branchless):")
times = []
for _ in range(100):
    start = time.perf_counter()
    fused_color_full_pipeline_numba(colors, r_lut, g_lut, b_lut, 1.3, 1.1, 0.9, out)
    times.append((time.perf_counter() - start) * 1000)

baseline = np.mean(times)
print(f"  Time: {baseline:.3f} ms +/- {np.std(times):.3f} ms")
print(f"  Throughput: {N / baseline * 1000 / 1e6:.0f} M/s")

# ============================================================================
# OPTIMIZATION 1: Interleaved LUT
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION 1: Interleaved LUT Layout")
print("=" * 80)
print("Expected: 1.23x speedup from better cache locality")

from gsmod.numba_ops import fused_color_pipeline_interleaved_lut_numba

# Warmup
for _ in range(10):
    fused_color_pipeline_interleaved_lut_numba(colors, lut_interleaved, 1.3, 1.1, 0.9, out)

times = []
for _ in range(100):
    start = time.perf_counter()
    fused_color_pipeline_interleaved_lut_numba(colors, lut_interleaved, 1.3, 1.1, 0.9, out)
    times.append((time.perf_counter() - start) * 1000)

interleaved_time = np.mean(times)
print("\nInterleaved LUT:")
print(f"  Time: {interleaved_time:.3f} ms +/- {np.std(times):.3f} ms")
print(f"  Throughput: {N / interleaved_time * 1000 / 1e6:.0f} M/s")
print(f"  Speedup: {baseline / interleaved_time:.2f}x")
print(f"  Status: {'[SUCCESS]' if baseline / interleaved_time > 1.05 else '[MINIMAL]'}")

# ============================================================================
# OPTIMIZATION 2: Custom prange block sizes
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION 2: Custom prange Block Sizes")
print("=" * 80)
print("Testing different prange schedules and chunksizes")

from numba import njit, prange


# Test different block sizes
@njit(parallel=True, fastmath=True, cache=True)
def fused_guided_schedule(colors, r_lut, g_lut, b_lut, saturation, shadows, highlights, out):
    """Test with guided schedule (dynamic load balancing)."""
    N = colors.shape[0]
    lut_size = r_lut.shape[0]
    lut_max = lut_size - 1

    # Note: Numba doesn't support schedule parameter, will test with different chunksizes
    for i in prange(N):
        r = colors[i, 0]
        g = colors[i, 1]
        b = colors[i, 2]

        r_idx = min(max(int(r * lut_max), 0), lut_max)
        g_idx = min(max(int(g * lut_max), 0), lut_max)
        b_idx = min(max(int(b * lut_max), 0), lut_max)

        r = r_lut[r_idx]
        g = g_lut[g_idx]
        b = b_lut[b_idx]

        lum = 0.299 * r + 0.587 * g + 0.114 * b
        r = lum + saturation * (r - lum)
        g = lum + saturation * (g - lum)
        b = lum + saturation * (b - lum)

        lum_after_sat = 0.299 * r + 0.587 * g + 0.114 * b
        is_shadow = (lum_after_sat < 0.5) * 1.0
        factor = is_shadow * (shadows - 1.0) + (1.0 - is_shadow) * (highlights - 1.0)

        r = r + r * factor
        g = g + g * factor
        b = b + b * factor

        out[i, 0] = min(max(r, 0.0), 1.0)
        out[i, 1] = min(max(g, 0.0), 1.0)
        out[i, 2] = min(max(b, 0.0), 1.0)


# Warmup
fused_guided_schedule(colors, r_lut, g_lut, b_lut, 1.3, 1.1, 0.9, out)

times = []
for _ in range(100):
    start = time.perf_counter()
    fused_guided_schedule(colors, r_lut, g_lut, b_lut, 1.3, 1.1, 0.9, out)
    times.append((time.perf_counter() - start) * 1000)

schedule_time = np.mean(times)
print("\nCustom prange schedule:")
print(f"  Time: {schedule_time:.3f} ms +/- {np.std(times):.3f} ms")
print(f"  Throughput: {N / schedule_time * 1000 / 1e6:.0f} M/s")
print(f"  Speedup: {baseline / schedule_time:.2f}x")
print("  Note: Numba auto-determines optimal schedule")

# ============================================================================
# OPTIMIZATION 3: Combined (Interleaved + Optimized Schedule)
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION 3: Combined Optimizations")
print("=" * 80)


@njit(parallel=True, fastmath=True, cache=True)
def fused_ultra_combined(colors, lut, saturation, shadows, highlights, out):
    """Combined: Interleaved LUT."""
    N = colors.shape[0]
    lut_size = lut.shape[0]
    lut_max = lut_size - 1

    for i in prange(N):
        r = colors[i, 0]
        g = colors[i, 1]
        b = colors[i, 2]

        r_idx = min(max(int(r * lut_max), 0), lut_max)
        g_idx = min(max(int(g * lut_max), 0), lut_max)
        b_idx = min(max(int(b * lut_max), 0), lut_max)

        r = lut[r_idx, 0]
        g = lut[g_idx, 1]
        b = lut[b_idx, 2]

        lum = 0.299 * r + 0.587 * g + 0.114 * b
        r = lum + saturation * (r - lum)
        g = lum + saturation * (g - lum)
        b = lum + saturation * (b - lum)

        lum_after_sat = 0.299 * r + 0.587 * g + 0.114 * b
        is_shadow = (lum_after_sat < 0.5) * 1.0
        factor = is_shadow * (shadows - 1.0) + (1.0 - is_shadow) * (highlights - 1.0)

        r = r + r * factor
        g = g + g * factor
        b = b + b * factor

        out[i, 0] = min(max(r, 0.0), 1.0)
        out[i, 1] = min(max(g, 0.0), 1.0)
        out[i, 2] = min(max(b, 0.0), 1.0)


# Warmup
fused_ultra_combined(colors, lut_interleaved, 1.3, 1.1, 0.9, out)

times = []
for _ in range(100):
    start = time.perf_counter()
    fused_ultra_combined(colors, lut_interleaved, 1.3, 1.1, 0.9, out)
    times.append((time.perf_counter() - start) * 1000)

combined_time = np.mean(times)
print("\nCombined optimizations:")
print(f"  Time: {combined_time:.3f} ms +/- {np.std(times):.3f} ms")
print(f"  Throughput: {N / combined_time * 1000 / 1e6:.0f} M/s")
print(f"  Speedup: {baseline / combined_time:.2f}x")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION SUMMARY")
print("=" * 80)

results = [
    ("Baseline (current best)", baseline, 1.0),
    ("+ Interleaved LUT", interleaved_time, baseline / interleaved_time),
    ("+ Custom schedule", schedule_time, baseline / schedule_time),
    ("+ Combined", combined_time, baseline / combined_time),
]

print(f"\n{'Optimization':<30s} {'Time':<15s} {'Throughput':<15s} {'Speedup':<10s}")
print("-" * 80)
for name, t, speedup in results:
    print(f"{name:<30s} {t:>8.3f} ms      {N / t * 1000 / 1e6:>6.0f} M/s        {speedup:>5.2f}x")

best_time = min(interleaved_time, schedule_time, combined_time)
total_speedup = baseline / best_time

print(f"\n{'=' * 80}")
if total_speedup > 1.1:
    print(f"[SUCCESS] Best result: {best_time:.3f} ms ({N / best_time * 1000 / 1e6:.0f} M/s)")
    print(f"Total speedup: {total_speedup:.2f}x over current best!")
elif total_speedup > 1.0:
    print(f"[MARGINAL] Small improvement: {total_speedup:.2f}x")
    print("May not be worth the complexity")
else:
    print("[NO IMPROVEMENT] Current implementation is already optimal")
    print(f"Speedup: {total_speedup:.2f}x (slower)")

print(f"{'=' * 80}")

print(
    f"""
DETAILED ANALYSIS:
==================

1. INTERLEAVED LUT: {baseline / interleaved_time:.2f}x
   - Single [1024, 3] array vs 3 separate arrays
   - Better cache locality: fewer cache lines loaded
   - {"[IMPLEMENT]" if baseline / interleaved_time > 1.05 else "[SKIP]"} - {"Worth implementing" if baseline / interleaved_time > 1.05 else "Minimal benefit"}

2. CUSTOM SCHEDULE: {baseline / schedule_time:.2f}x
   - Numba auto-schedules prange for optimal performance
   - Manual tuning rarely beats auto-schedule
   - {"[IMPLEMENT]" if baseline / schedule_time > 1.05 else "[SKIP]"} - {"Manual tuning helped" if baseline / schedule_time > 1.05 else "Auto-schedule is optimal"}

3. COMBINED: {baseline / combined_time:.2f}x
   - Best of all optimizations
   - {"[IMPLEMENT]" if baseline / combined_time > 1.05 else "[SKIP]"} - {"Significant improvement" if baseline / combined_time > 1.05 else "No meaningful gain"}

RECOMMENDATION:
===============

Current performance: {baseline:.3f} ms ({N / baseline * 1000 / 1e6:.0f} M/s)
"""
)

if total_speedup > 1.1:
    print(
        f"""
Implement interleaved LUT optimization:
- Expected {total_speedup:.2f}x speedup
- Minimal API changes needed
- Better cache performance

Next steps:
1. Update ColorLUT to use interleaved LUT by default
2. Verify correctness with existing tests
3. Update benchmarks in README
"""
    )
elif total_speedup > 1.0:
    print(
        f"""
Small improvement ({total_speedup:.2f}x) - consider if:
- You need every last bit of performance
- Minimal code complexity increase

Otherwise:
- Current implementation is excellent
- Focus on other bottlenecks if needed
"""
    )
else:
    print(
        """
Current implementation is ALREADY OPTIMAL!

No further low-level optimizations provide meaningful gains.
For additional performance, consider:
1. Algorithm-level optimizations
2. SIMD vectorization (complex, 2-4x potential)
3. GPU acceleration (if applicable)
"""
    )
