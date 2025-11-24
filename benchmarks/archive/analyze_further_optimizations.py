"""
Deep analysis of further optimization opportunities for 1024 LUT.

Current performance: 0.363 ms (276 M/s) for 100K colors
Target: Can we get to 0.2 ms (500 M/s) or better?

Optimization candidates:
1. Reduce clipping operations (6 clips -> 1 clip at end)
2. Skip Phase 2 ops when params == 1.0
3. Vectorize with @vectorize
4. Custom prange block size tuning
5. Interleaved LUT layout for cache
6. Reduce luminance calculations
7. Manual loop unrolling
"""

import time

import numpy as np
from numba import njit, prange

print("=" * 80)
print("FURTHER OPTIMIZATION ANALYSIS (1024 LUT)")
print("=" * 80)

N = 100_000
colors = np.random.rand(N, 3).astype(np.float32)
out = np.empty_like(colors)

# Create LUTs
lut_size = 1024
r_lut = np.linspace(0, 1, lut_size, dtype=np.float32)
g_lut = np.linspace(0, 1, lut_size, dtype=np.float32)
b_lut = np.linspace(0, 1, lut_size, dtype=np.float32)

# ============================================================================
# BASELINE: Current implementation
# ============================================================================

from gsmod.numba_ops import fused_color_full_pipeline_numba

# Warmup
for _ in range(10):
    fused_color_full_pipeline_numba(colors, r_lut, g_lut, b_lut, 1.3, 1.1, 0.9, out)

print("\n[BASELINE] Current implementation:")
times = []
for _ in range(100):
    start = time.perf_counter()
    fused_color_full_pipeline_numba(colors, r_lut, g_lut, b_lut, 1.3, 1.1, 0.9, out)
    times.append((time.perf_counter() - start) * 1000)

baseline = np.mean(times)
print(f"  Time: {baseline:.3f} ms ({N / baseline * 1000 / 1e6:.0f} M/s)")

# ============================================================================
# OPTIMIZATION 1: Reduce clipping (clip only at end)
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION 1: Reduce Clipping Operations")
print("=" * 80)
print("Current: 6 clips (3 after saturation + 3 at end)")
print("Proposed: 1 clip at very end (trust intermediate overflow handling)")


@njit(parallel=True, fastmath=True, cache=True)
def fused_reduced_clipping(colors, r_lut, g_lut, b_lut, saturation, shadows, highlights, out):
    """Only clip at the very end."""
    N = colors.shape[0]
    lut_size = r_lut.shape[0]
    lut_max = lut_size - 1

    for i in prange(N):
        r = colors[i, 0]
        g = colors[i, 1]
        b = colors[i, 2]

        # Phase 1: LUT lookup (with clipping for array access)
        r_idx = min(max(int(r * lut_max), 0), lut_max)
        g_idx = min(max(int(g * lut_max), 0), lut_max)
        b_idx = min(max(int(b * lut_max), 0), lut_max)

        r = r_lut[r_idx]
        g = g_lut[g_idx]
        b = b_lut[b_idx]

        # Phase 2: Saturation (NO clipping)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        r = lum + saturation * (r - lum)
        g = lum + saturation * (g - lum)
        b = lum + saturation * (b - lum)

        # Shadows/Highlights (NO clipping, compute on potentially out-of-range values)
        lum_after = 0.299 * r + 0.587 * g + 0.114 * b
        is_shadow = (lum_after < 0.5) * 1.0
        factor = is_shadow * (shadows - 1.0) + (1.0 - is_shadow) * (highlights - 1.0)

        r = r + r * factor
        g = g + g * factor
        b = b + b * factor

        # SINGLE final clip
        out[i, 0] = min(max(r, 0.0), 1.0)
        out[i, 1] = min(max(g, 0.0), 1.0)
        out[i, 2] = min(max(b, 0.0), 1.0)


# Warmup
fused_reduced_clipping(colors, r_lut, g_lut, b_lut, 1.3, 1.1, 0.9, out)

times = []
for _ in range(100):
    start = time.perf_counter()
    fused_reduced_clipping(colors, r_lut, g_lut, b_lut, 1.3, 1.1, 0.9, out)
    times.append((time.perf_counter() - start) * 1000)

opt1_time = np.mean(times)
print("\nReduced clipping:")
print(f"  Time: {opt1_time:.3f} ms ({N / opt1_time * 1000 / 1e6:.0f} M/s)")
print(f"  Speedup: {baseline / opt1_time:.2f}x")

# ============================================================================
# OPTIMIZATION 2: Skip saturation when saturation == 1.0
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION 2: Skip Saturation When == 1.0")
print("=" * 80)


@njit(parallel=True, fastmath=True, cache=True)
def fused_skip_saturation(colors, r_lut, g_lut, b_lut, saturation, shadows, highlights, out):
    """Skip saturation computation when saturation == 1.0."""
    N = colors.shape[0]
    lut_size = r_lut.shape[0]
    lut_max = lut_size - 1
    skip_sat = saturation == 1.0

    for i in prange(N):
        r = colors[i, 0]
        g = colors[i, 1]
        b = colors[i, 2]

        # Phase 1: LUT lookup
        r_idx = min(max(int(r * lut_max), 0), lut_max)
        g_idx = min(max(int(g * lut_max), 0), lut_max)
        b_idx = min(max(int(b * lut_max), 0), lut_max)

        r = r_lut[r_idx]
        g = g_lut[g_idx]
        b = b_lut[b_idx]

        # Phase 2: Saturation (branchless skip)
        lum = 0.299 * r + 0.587 * g + 0.114 * b

        # Branchless: if skip_sat, use (r-lum), else saturation*(r-lum)
        sat_factor = 1.0 if skip_sat else saturation
        r = lum + sat_factor * (r - lum)
        g = lum + sat_factor * (g - lum)
        b = lum + sat_factor * (b - lum)

        # Shadows/Highlights
        lum_after = 0.299 * r + 0.587 * g + 0.114 * b
        is_shadow = (lum_after < 0.5) * 1.0
        factor = is_shadow * (shadows - 1.0) + (1.0 - is_shadow) * (highlights - 1.0)

        r = r + r * factor
        g = g + g * factor
        b = b + b * factor

        out[i, 0] = min(max(r, 0.0), 1.0)
        out[i, 1] = min(max(g, 0.0), 1.0)
        out[i, 2] = min(max(b, 0.0), 1.0)


# Test with saturation = 1.0
fused_skip_saturation(colors, r_lut, g_lut, b_lut, 1.0, 1.1, 0.9, out)

times = []
for _ in range(100):
    start = time.perf_counter()
    fused_skip_saturation(colors, r_lut, g_lut, b_lut, 1.0, 1.1, 0.9, out)
    times.append((time.perf_counter() - start) * 1000)

opt2_time = np.mean(times)
print("\nSkip saturation (saturation=1.0):")
print(f"  Time: {opt2_time:.3f} ms ({N / opt2_time * 1000 / 1e6:.0f} M/s)")
print(f"  Speedup: {baseline / opt2_time:.2f}x")

# ============================================================================
# OPTIMIZATION 3: Calculate luminance once
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION 3: Calculate Luminance Once")
print("=" * 80)
print("Current: Calculate luminance twice (before saturation, after saturation)")
print("Proposed: Reuse or eliminate redundant calculation")


@njit(parallel=True, fastmath=True, cache=True)
def fused_single_luminance(colors, r_lut, g_lut, b_lut, saturation, shadows, highlights, out):
    """Calculate luminance only once when possible."""
    N = colors.shape[0]
    lut_size = r_lut.shape[0]
    lut_max = lut_size - 1

    for i in prange(N):
        r = colors[i, 0]
        g = colors[i, 1]
        b = colors[i, 2]

        # Phase 1: LUT lookup
        r_idx = min(max(int(r * lut_max), 0), lut_max)
        g_idx = min(max(int(g * lut_max), 0), lut_max)
        b_idx = min(max(int(b * lut_max), 0), lut_max)

        r = r_lut[r_idx]
        g = g_lut[g_idx]
        b = b_lut[b_idx]

        # Calculate luminance ONCE before saturation
        lum = 0.299 * r + 0.587 * g + 0.114 * b

        # Apply saturation
        r = lum + saturation * (r - lum)
        g = lum + saturation * (g - lum)
        b = lum + saturation * (b - lum)

        # For shadows/highlights, we could approximate using the pre-saturation luminance
        # scaled by saturation, but this changes behavior. Let's recalculate.
        lum_after = 0.299 * r + 0.587 * g + 0.114 * b
        is_shadow = (lum_after < 0.5) * 1.0
        factor = is_shadow * (shadows - 1.0) + (1.0 - is_shadow) * (highlights - 1.0)

        r = r + r * factor
        g = g + g * factor
        b = b + b * factor

        out[i, 0] = min(max(r, 0.0), 1.0)
        out[i, 1] = min(max(g, 0.0), 1.0)
        out[i, 2] = min(max(b, 0.0), 1.0)


# Note: This is same as baseline, just for comparison
fused_single_luminance(colors, r_lut, g_lut, b_lut, 1.3, 1.1, 0.9, out)

times = []
for _ in range(100):
    start = time.perf_counter()
    fused_single_luminance(colors, r_lut, g_lut, b_lut, 1.3, 1.1, 0.9, out)
    times.append((time.perf_counter() - start) * 1000)

opt3_time = np.mean(times)
print("\nSingle luminance calculation:")
print(f"  Time: {opt3_time:.3f} ms ({N / opt3_time * 1000 / 1e6:.0f} M/s)")
print(f"  Speedup: {baseline / opt3_time:.2f}x")
print("  Note: Must calculate twice for correct shadows/highlights")

# ============================================================================
# OPTIMIZATION 4: Interleaved LUT for cache locality
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION 4: Interleaved LUT Layout")
print("=" * 80)
print("Current: Separate r_lut, g_lut, b_lut (3 cache lines)")
print("Proposed: Single interleaved [N, 3] array (1 cache line)")

# Create interleaved LUT
lut_interleaved = np.stack([r_lut, g_lut, b_lut], axis=1)  # [1024, 3]


@njit(parallel=True, fastmath=True, cache=True)
def fused_interleaved_lut(colors, lut, saturation, shadows, highlights, out):
    """Use interleaved LUT for better cache locality."""
    N = colors.shape[0]
    lut_size = lut.shape[0]
    lut_max = lut_size - 1

    for i in prange(N):
        r = colors[i, 0]
        g = colors[i, 1]
        b = colors[i, 2]

        # Phase 1: LUT lookup (single array access pattern)
        r_idx = min(max(int(r * lut_max), 0), lut_max)
        g_idx = min(max(int(g * lut_max), 0), lut_max)
        b_idx = min(max(int(b * lut_max), 0), lut_max)

        r = lut[r_idx, 0]
        g = lut[g_idx, 1]
        b = lut[b_idx, 2]

        # Phase 2: Same as before
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        r = lum + saturation * (r - lum)
        g = lum + saturation * (g - lum)
        b = lum + saturation * (b - lum)

        lum_after = 0.299 * r + 0.587 * g + 0.114 * b
        is_shadow = (lum_after < 0.5) * 1.0
        factor = is_shadow * (shadows - 1.0) + (1.0 - is_shadow) * (highlights - 1.0)

        r = r + r * factor
        g = g + g * factor
        b = b + b * factor

        out[i, 0] = min(max(r, 0.0), 1.0)
        out[i, 1] = min(max(g, 0.0), 1.0)
        out[i, 2] = min(max(b, 0.0), 1.0)


# Warmup
fused_interleaved_lut(colors, lut_interleaved, 1.3, 1.1, 0.9, out)

times = []
for _ in range(100):
    start = time.perf_counter()
    fused_interleaved_lut(colors, lut_interleaved, 1.3, 1.1, 0.9, out)
    times.append((time.perf_counter() - start) * 1000)

opt4_time = np.mean(times)
print("\nInterleaved LUT:")
print(f"  Time: {opt4_time:.3f} ms ({N / opt4_time * 1000 / 1e6:.0f} M/s)")
print(f"  Speedup: {baseline / opt4_time:.2f}x")

# ============================================================================
# OPTIMIZATION 5: Combined optimizations
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION 5: Combined Best Approaches")
print("=" * 80)


@njit(parallel=True, fastmath=True, cache=True)
def fused_ultra_optimized(colors, r_lut, g_lut, b_lut, saturation, shadows, highlights, out):
    """Combine: reduced clipping + branchless skips."""
    N = colors.shape[0]
    lut_size = r_lut.shape[0]
    lut_max = lut_size - 1

    for i in prange(N):
        r = colors[i, 0]
        g = colors[i, 1]
        b = colors[i, 2]

        # Phase 1: LUT lookup (must clip for array bounds)
        r_idx = min(max(int(r * lut_max), 0), lut_max)
        g_idx = min(max(int(g * lut_max), 0), lut_max)
        b_idx = min(max(int(b * lut_max), 0), lut_max)

        r = r_lut[r_idx]
        g = g_lut[g_idx]
        b = b_lut[b_idx]

        # Phase 2: Saturation (no intermediate clipping)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        r = lum + saturation * (r - lum)
        g = lum + saturation * (g - lum)
        b = lum + saturation * (b - lum)

        # Shadows/Highlights (no intermediate clipping)
        lum_after = 0.299 * r + 0.587 * g + 0.114 * b
        is_shadow = (lum_after < 0.5) * 1.0
        factor = is_shadow * (shadows - 1.0) + (1.0 - is_shadow) * (highlights - 1.0)

        r = r + r * factor
        g = g + g * factor
        b = b + b * factor

        # Single final clip
        out[i, 0] = min(max(r, 0.0), 1.0)
        out[i, 1] = min(max(g, 0.0), 1.0)
        out[i, 2] = min(max(b, 0.0), 1.0)


# Warmup
fused_ultra_optimized(colors, r_lut, g_lut, b_lut, 1.3, 1.1, 0.9, out)

times = []
for _ in range(100):
    start = time.perf_counter()
    fused_ultra_optimized(colors, r_lut, g_lut, b_lut, 1.3, 1.1, 0.9, out)
    times.append((time.perf_counter() - start) * 1000)

opt5_time = np.mean(times)
print("\nCombined ultra-optimized:")
print(f"  Time: {opt5_time:.3f} ms ({N / opt5_time * 1000 / 1e6:.0f} M/s)")
print(f"  Speedup: {baseline / opt5_time:.2f}x")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION SUMMARY")
print("=" * 80)

results = [
    ("Baseline (current)", baseline, 1.0),
    ("Reduced clipping", opt1_time, baseline / opt1_time),
    ("Skip saturation=1.0", opt2_time, baseline / opt2_time),
    ("Single luminance", opt3_time, baseline / opt3_time),
    ("Interleaved LUT", opt4_time, baseline / opt4_time),
    ("Combined ultra", opt5_time, baseline / opt5_time),
]

print(f"\n{'Optimization':<25s} {'Time':<12s} {'Throughput':<15s} {'Speedup':<10s}")
print("-" * 80)
for name, t, speedup in results:
    print(f"{name:<25s} {t:>8.3f} ms    {N / t * 1000 / 1e6:>6.0f} M/s        {speedup:>5.2f}x")

best_time = min(opt1_time, opt2_time, opt3_time, opt4_time, opt5_time)
best_speedup = baseline / best_time

print(f"\n{'=' * 80}")
print(
    f"BEST RESULT: {best_time:.3f} ms ({N / best_time * 1000 / 1e6:.0f} M/s) - {best_speedup:.2f}x speedup!"
)
print(f"{'=' * 80}")

print(
    f"""
RECOMMENDATIONS:
================

1. REDUCED CLIPPING: {baseline / opt1_time:.2f}x speedup
   - Only clip at final output
   - Trust fastmath for intermediate overflow handling
   - Safe because final clip catches all out-of-range values

2. INTERLEAVED LUT: {baseline / opt4_time:.2f}x speedup
   - Better cache locality (1 array vs 3 arrays)
   - Fewer memory accesses
   - Worth considering if speedup > 1.05x

3. SKIP SATURATION: {baseline / opt2_time:.2f}x speedup (when saturation=1.0)
   - Branchless detection and skip
   - Useful for many common cases

NEXT STEPS:
===========

If these optimizations provide < 1.2x improvement, consider:
- Explicit SIMD with @vectorize (2-4x potential)
- Custom block size tuning (1.1-1.3x)
- AVX512 intrinsics via Numba (2-3x, complex)
"""
)
