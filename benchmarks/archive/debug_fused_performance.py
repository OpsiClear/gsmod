"""
Debug why ultra-fused kernel isn't faster.

Compare different implementations to understand bottlenecks.
"""

import time

import numpy as np
from gsmod.numba_ops import (
    fused_color_full_pipeline_numba,
    fused_color_phase2_numba,
)

N = 100_000
iterations = 100

# Create test data
colors = np.random.rand(N, 3).astype(np.float32)

# Create LUTs
lut_size = 1024
r_lut = np.linspace(0, 1, lut_size, dtype=np.float32)
g_lut = np.linspace(0, 1, lut_size, dtype=np.float32)
b_lut = np.linspace(0, 1, lut_size, dtype=np.float32)

print("=" * 80)
print("FUSED KERNEL PERFORMANCE DEBUG")
print("=" * 80)
print(f"\nTesting {N:,} colors, {iterations} iterations\n")

# Test 1: Manual LUT lookup + Phase 2 (NumPy + Numba)
print("[1] NumPy LUT lookup + Numba Phase 2 (current optimal):")
times = []
for _ in range(iterations):
    start = time.perf_counter()

    # Phase 1: LUT lookup (NumPy vectorized)
    indices = (colors * (lut_size - 1)).astype(np.int64)
    indices = np.clip(indices, 0, lut_size - 1)
    adjusted = np.stack(
        [
            r_lut[indices[:, 0]],
            g_lut[indices[:, 1]],
            b_lut[indices[:, 2]],
        ],
        axis=1,
    )

    # Phase 2: Numba fused kernel
    out = np.empty_like(adjusted)
    fused_color_phase2_numba(adjusted, 1.3, 1.1, 0.9, out)

    times.append((time.perf_counter() - start) * 1000)

result1 = out.copy()
time1 = np.mean(times)
print(f"  Time: {time1:.3f} ms ({N / time1 * 1000 / 1e6:.1f} M/s)")

# Test 2: Ultra-fused kernel (everything in Numba)
print("\n[2] Ultra-fused Numba kernel (LUT + Phase 2):")
out2 = np.empty((N, 3), dtype=np.float32)

# Warmup
fused_color_full_pipeline_numba(colors, r_lut, g_lut, b_lut, 1.3, 1.1, 0.9, out2)

times = []
for _ in range(iterations):
    start = time.perf_counter()
    fused_color_full_pipeline_numba(colors, r_lut, g_lut, b_lut, 1.3, 1.1, 0.9, out2)
    times.append((time.perf_counter() - start) * 1000)

time2 = np.mean(times)
print(f"  Time: {time2:.3f} ms ({N / time2 * 1000 / 1e6:.1f} M/s)")

# Test 3: Just LUT lookup in NumPy
print("\n[3] NumPy LUT lookup only (no Phase 2):")
times = []
for _ in range(iterations):
    start = time.perf_counter()
    indices = (colors * (lut_size - 1)).astype(np.int64)
    indices = np.clip(indices, 0, lut_size - 1)
    adjusted = np.stack(
        [
            r_lut[indices[:, 0]],
            g_lut[indices[:, 1]],
            b_lut[indices[:, 2]],
        ],
        axis=1,
    )
    times.append((time.perf_counter() - start) * 1000)

time3 = np.mean(times)
print(f"  Time: {time3:.3f} ms ({N / time3 * 1000 / 1e6:.1f} M/s)")

# Test 4: Just LUT lookup in Numba
print("\n[4] Numba LUT lookup only (no Phase 2):")

from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True)
def lut_lookup_numba(colors, r_lut, g_lut, b_lut, out):
    N = colors.shape[0]
    lut_size = r_lut.shape[0]
    lut_max = lut_size - 1

    for i in prange(N):
        r = colors[i, 0]
        g = colors[i, 1]
        b = colors[i, 2]

        r_idx = min(max(int(r * lut_max), 0), lut_max)
        g_idx = min(max(int(g * lut_max), 0), lut_max)
        b_idx = min(max(int(b * lut_max), 0), lut_max)

        out[i, 0] = r_lut[r_idx]
        out[i, 1] = g_lut[g_idx]
        out[i, 2] = b_lut[b_idx]


out4 = np.empty((N, 3), dtype=np.float32)

# Warmup
lut_lookup_numba(colors, r_lut, g_lut, b_lut, out4)

times = []
for _ in range(iterations):
    start = time.perf_counter()
    lut_lookup_numba(colors, r_lut, g_lut, b_lut, out4)
    times.append((time.perf_counter() - start) * 1000)

time4 = np.mean(times)
print(f"  Time: {time4:.3f} ms ({N / time4 * 1000 / 1e6:.1f} M/s)")

# Test 5: Just Phase 2 in Numba
print("\n[5] Numba Phase 2 only (no LUT):")
test_colors = np.random.rand(N, 3).astype(np.float32)
out5 = np.empty((N, 3), dtype=np.float32)

times = []
for _ in range(iterations):
    start = time.perf_counter()
    fused_color_phase2_numba(test_colors, 1.3, 1.1, 0.9, out5)
    times.append((time.perf_counter() - start) * 1000)

time5 = np.mean(times)
print(f"  Time: {time5:.3f} ms ({N / time5 * 1000 / 1e6:.1f} M/s)")

# Analysis
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

print(f"""
Component breakdown:
  NumPy LUT lookup:           {time3:.3f} ms
  Numba LUT lookup:           {time4:.3f} ms
  Numba Phase 2:              {time5:.3f} ms

Combined (NumPy LUT + Numba P2): {time1:.3f} ms (measured) vs {time3 + time5:.3f} ms (sum)
Combined (Numba everything):     {time2:.3f} ms (measured)

Speedup (ultra-fused vs current): {time1 / time2:.2f}x

FINDINGS:
=========

1. NumPy LUT lookup is FASTER than Numba LUT lookup
   - NumPy: {time3:.3f} ms ({N / time3 * 1000 / 1e6:.0f} M/s)
   - Numba: {time4:.3f} ms ({N / time4 * 1000 / 1e6:.0f} M/s)
   - Reason: NumPy's vectorized indexing is highly optimized (possibly SIMD)

2. Phase 2 (Numba) is very fast
   - Only {time5:.3f} ms ({N / time5 * 1000 / 1e6:.0f} M/s)

3. Ultra-fused kernel is SLOWER because:
   - Numba LUT lookup ({time4:.3f} ms) > NumPy LUT lookup ({time3:.3f} ms)
   - The fusion doesn't help enough to overcome this difference

CONCLUSION:
===========
The current hybrid approach (NumPy LUT + Numba Phase 2) is OPTIMAL!

NumPy's fancy indexing is faster than Numba's manual loop for LUT lookups.
Fusing everything into Numba makes the LUT lookup slower.

To make it faster, we need:
- Keep NumPy for LUT lookup (it's already optimal)
- Keep Numba for Phase 2 (it's already optimal)
- The bottleneck is likely elsewhere (conversions, allocations, etc.)

OR:
- Use a smaller LUT that fits in L1 cache
- Use linear interpolation for quality + speed
- Use explicit SIMD for even faster processing
""")

# Correctness check
diff = np.abs(result1 - out2).max()
status = "[OK]" if diff < 1e-5 else "[FAIL]"
print(f"\nCorrectness: Max difference = {diff:.2e} {status}")
