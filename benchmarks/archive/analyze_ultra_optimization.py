"""
Ultra-deep analysis for further optimization opportunities.

Current performance: 0.086 ms for 100K colors (1,165 M/s)
Target: Can we reach 0.02-0.04 ms (2,500-5,000 M/s)?
"""

import time

import numpy as np
from gsmod.numba_ops import fused_color_full_pipeline_numba

print("=" * 80)
print("ULTRA-OPTIMIZATION ANALYSIS")
print("=" * 80)

N = 100_000
colors = np.random.rand(N, 3).astype(np.float32)
out = np.empty_like(colors)

# Create LUTs
lut_size = 1024
r_lut = np.linspace(0, 1, lut_size, dtype=np.float32)
g_lut = np.linspace(0, 1, lut_size, dtype=np.float32)
b_lut = np.linspace(0, 1, lut_size, dtype=np.float32)

# Warmup
for _ in range(10):
    fused_color_full_pipeline_numba(colors, r_lut, g_lut, b_lut, 1.3, 1.1, 0.9, out)

print("\n[BASELINE] Current fused kernel performance:")
times = []
for _ in range(100):
    start = time.perf_counter()
    fused_color_full_pipeline_numba(colors, r_lut, g_lut, b_lut, 1.3, 1.1, 0.9, out)
    times.append((time.perf_counter() - start) * 1000)

baseline = np.mean(times)
print(f"  Time: {baseline:.3f} ms ({N / baseline * 1000 / 1e6:.0f} M/s)")

# ============================================================================
# OPTIMIZATION 1: Branchless Phase 2 (eliminate branch misprediction)
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION 1: Branchless Phase 2")
print("=" * 80)

from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def fused_branchless(colors, r_lut, g_lut, b_lut, saturation, shadows, highlights, out):
    """Branchless version - no if statements."""
    N = colors.shape[0]
    lut_size = r_lut.shape[0]
    lut_max = lut_size - 1

    for i in prange(N):
        # Phase 1: LUT lookup
        r = colors[i, 0]
        g = colors[i, 1]
        b = colors[i, 2]

        r_idx = min(max(int(r * lut_max), 0), lut_max)
        g_idx = min(max(int(g * lut_max), 0), lut_max)
        b_idx = min(max(int(b * lut_max), 0), lut_max)

        r = r_lut[r_idx]
        g = g_lut[g_idx]
        b = b_lut[b_idx]

        # Phase 2: Saturation (branchless)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        r = lum + saturation * (r - lum)
        g = lum + saturation * (g - lum)
        b = lum + saturation * (b - lum)
        r = min(max(r, 0.0), 1.0)
        g = min(max(g, 0.0), 1.0)
        b = min(max(b, 0.0), 1.0)

        # Phase 2: Shadows/Highlights (branchless)
        lum_after = 0.299 * r + 0.587 * g + 0.114 * b
        # Branchless: convert boolean to float using multiplication
        is_shadow = (lum_after < 0.5) * 1.0  # True->1.0, False->0.0
        is_highlight = 1.0 - is_shadow

        factor = is_shadow * (shadows - 1.0) + is_highlight * (highlights - 1.0)
        r = r + r * factor
        g = g + g * factor
        b = b + b * factor

        out[i, 0] = min(max(r, 0.0), 1.0)
        out[i, 1] = min(max(g, 0.0), 1.0)
        out[i, 2] = min(max(b, 0.0), 1.0)


# Warmup
fused_branchless(colors, r_lut, g_lut, b_lut, 1.3, 1.1, 0.9, out)

times = []
for _ in range(100):
    start = time.perf_counter()
    fused_branchless(colors, r_lut, g_lut, b_lut, 1.3, 1.1, 0.9, out)
    times.append((time.perf_counter() - start) * 1000)

branchless_time = np.mean(times)
print("\nBranchless version:")
print(f"  Time: {branchless_time:.3f} ms ({N / branchless_time * 1000 / 1e6:.0f} M/s)")
print(f"  Speedup: {baseline / branchless_time:.2f}x")

# ============================================================================
# OPTIMIZATION 2: Smaller LUT with linear interpolation
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION 2: Smaller LUT + Linear Interpolation")
print("=" * 80)

# Create smaller LUTs (128 entries instead of 1024)
small_lut_size = 128
r_lut_small = np.linspace(0, 1, small_lut_size, dtype=np.float32)
g_lut_small = np.linspace(0, 1, small_lut_size, dtype=np.float32)
b_lut_small = np.linspace(0, 1, small_lut_size, dtype=np.float32)


@njit(parallel=True, fastmath=True, cache=True)
def fused_interpolated_lut(colors, r_lut, g_lut, b_lut, saturation, shadows, highlights, out):
    """Version with linear interpolation for LUT lookup."""
    N = colors.shape[0]
    lut_size = r_lut.shape[0]
    lut_max_f = float(lut_size - 1)

    for i in prange(N):
        # Phase 1: LUT lookup with LINEAR INTERPOLATION
        r_in = colors[i, 0]
        g_in = colors[i, 1]
        b_in = colors[i, 2]

        # Linear interpolation for R
        r_pos = r_in * lut_max_f
        r_idx = int(r_pos)
        r_idx = min(max(r_idx, 0), lut_size - 2)  # Ensure we can access idx+1
        r_frac = r_pos - r_idx
        r = r_lut[r_idx] * (1.0 - r_frac) + r_lut[r_idx + 1] * r_frac

        # Linear interpolation for G
        g_pos = g_in * lut_max_f
        g_idx = int(g_pos)
        g_idx = min(max(g_idx, 0), lut_size - 2)
        g_frac = g_pos - g_idx
        g = g_lut[g_idx] * (1.0 - g_frac) + g_lut[g_idx + 1] * g_frac

        # Linear interpolation for B
        b_pos = b_in * lut_max_f
        b_idx = int(b_pos)
        b_idx = min(max(b_idx, 0), lut_size - 2)
        b_frac = b_pos - b_idx
        b = b_lut[b_idx] * (1.0 - b_frac) + b_lut[b_idx + 1] * b_frac

        # Phase 2: Same as before
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        r = lum + saturation * (r - lum)
        g = lum + saturation * (g - lum)
        b = lum + saturation * (b - lum)
        r = min(max(r, 0.0), 1.0)
        g = min(max(g, 0.0), 1.0)
        b = min(max(b, 0.0), 1.0)

        lum_after = 0.299 * r + 0.587 * g + 0.114 * b
        is_shadow = (lum_after < 0.5) * 1.0
        factor = is_shadow * (shadows - 1.0) + (1.0 - is_shadow) * (highlights - 1.0)

        r = r + r * factor
        g = g + g * factor
        b = b + b * factor

        out[i, 0] = min(max(r, 0.0), 1.0)
        out[i, 1] = min(max(g, 0.0), 1.0)
        out[i, 2] = min(max(b, 0.0), 1.0)


# Test with both 1024 and 128 LUT sizes
for test_lut_size, test_luts in [
    (1024, (r_lut, g_lut, b_lut)),
    (128, (r_lut_small, g_lut_small, b_lut_small)),
    (64, (np.linspace(0, 1, 64, dtype=np.float32),) * 3),
]:
    # Warmup
    fused_interpolated_lut(colors, test_luts[0], test_luts[1], test_luts[2], 1.3, 1.1, 0.9, out)

    times = []
    for _ in range(100):
        start = time.perf_counter()
        fused_interpolated_lut(colors, test_luts[0], test_luts[1], test_luts[2], 1.3, 1.1, 0.9, out)
        times.append((time.perf_counter() - start) * 1000)

    interp_time = np.mean(times)
    lut_bytes = test_lut_size * 3 * 4  # 3 LUTs, float32
    print(f"\nLUT size {test_lut_size} ({lut_bytes / 1024:.2f} KB):")
    print(f"  Time: {interp_time:.3f} ms ({N / interp_time * 1000 / 1e6:.0f} M/s)")
    print(f"  Speedup: {baseline / interp_time:.2f}x vs baseline")

# ============================================================================
# OPTIMIZATION 3: Skip identity LUT (when params are defaults)
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION 3: Skip Identity LUT Optimization")
print("=" * 80)


@njit(parallel=True, fastmath=True, cache=True)
def fused_skip_identity(colors, saturation, shadows, highlights, out):
    """Skip LUT lookup entirely when all Phase 1 params are defaults."""
    N = colors.shape[0]

    for i in prange(N):
        # Phase 1: SKIPPED (identity LUT)
        r = colors[i, 0]
        g = colors[i, 1]
        b = colors[i, 2]

        # Phase 2: Same as before
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        r = lum + saturation * (r - lum)
        g = lum + saturation * (g - lum)
        b = lum + saturation * (b - lum)
        r = min(max(r, 0.0), 1.0)
        g = min(max(g, 0.0), 1.0)
        b = min(max(b, 0.0), 1.0)

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
fused_skip_identity(colors, 1.3, 1.1, 0.9, out)

times = []
for _ in range(100):
    start = time.perf_counter()
    fused_skip_identity(colors, 1.3, 1.1, 0.9, out)
    times.append((time.perf_counter() - start) * 1000)

skip_time = np.mean(times)
print("\nSkip identity LUT (Phase 2 only):")
print(f"  Time: {skip_time:.3f} ms ({N / skip_time * 1000 / 1e6:.0f} M/s)")
print(f"  Speedup: {baseline / skip_time:.2f}x vs baseline")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("OPTIMIZATION SUMMARY")
print("=" * 80)

print(f"""
Baseline (current):          {baseline:.3f} ms ({N / baseline * 1000 / 1e6:.0f} M/s)  1.00x

Branchless Phase 2:          {branchless_time:.3f} ms ({N / branchless_time * 1000 / 1e6:.0f} M/s)  {baseline / branchless_time:.2f}x
Skip identity LUT:           {skip_time:.3f} ms ({N / skip_time * 1000 / 1e6:.0f} M/s)  {baseline / skip_time:.2f}x

Best small LUT (64 entries): See results above

RECOMMENDATIONS:
================

1. BRANCHLESS PHASE 2: {"[YES]" if baseline / branchless_time > 1.05 else "[NO]"} ({baseline / branchless_time:.2f}x speedup)
   - Eliminates branch misprediction
   - Same correctness

2. SMALL LUT + INTERPOLATION: Check results above
   - 64-128 entries fits in L1 cache
   - Higher quality (smooth gradients)
   - Trade computation for memory bandwidth

3. SKIP IDENTITY LUT: {baseline / skip_time:.2f}x speedup
   - Detect when temp=0.5, bright=1.0, contrast=1.0, gamma=1.0
   - Skip LUT lookup entirely
   - Use separate fast path

NEXT LEVEL OPTIMIZATIONS:
=========================

4. EXPLICIT SIMD/VECTORIZATION (2-4x potential)
   - Process 4-8 pixels at once
   - Requires @vectorize or manual SIMD
   - High effort

5. MEMORY PREFETCHING (1.2-1.5x potential)
   - Hint CPU to prefetch next pixels
   - Advanced, low-level

6. CUSTOM BLOCK SIZE TUNING (1.1-1.3x potential)
   - Manually tune prange block size
   - Currently auto-determined

Total potential: 2-5x additional speedup possible!
""")
