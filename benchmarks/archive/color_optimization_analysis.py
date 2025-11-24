"""
Analyze ColorLUT performance and identify optimization opportunities.
"""

import time

import numpy as np
import torch

from gsmod import ColorLUT

# Test with 100K colors
N = 100_000
colors = torch.rand(N, 3, device="cpu")

lut = ColorLUT(device="cpu", lut_size=1024)

# Benchmark individual components
print("=" * 80)
print("COLOR LUT PERFORMANCE ANALYSIS")
print("=" * 80)

# Test Phase 1 only (LUT operations)
print("\n[Phase 1: LUT Operations]")
times = []
for _ in range(100):
    start = time.perf_counter()
    result = lut.apply(
        colors,
        temperature=0.7,
        brightness=1.2,
        contrast=1.1,
        gamma=0.9,
        saturation=1.0,
        shadows=1.0,
        highlights=1.0,
    )  # No Phase 2
    times.append((time.perf_counter() - start) * 1000)

print(f"Time: {np.mean(times):.3f} ms ({N / np.mean(times) * 1000 / 1e6:.1f}M points/sec)")

# Test Phase 2 only (Sequential operations)
print("\n[Phase 2: Sequential Operations]")
times = []
for _ in range(100):
    # Pre-apply Phase 1 once
    phase1_result = lut.apply(
        colors,
        temperature=0.5,
        brightness=1.0,
        contrast=1.0,
        gamma=1.0,
        saturation=1.0,
        shadows=1.0,
        highlights=1.0,
    )

    start = time.perf_counter()
    result = lut.apply(
        phase1_result,
        temperature=0.5,
        brightness=1.0,
        contrast=1.0,
        gamma=1.0,
        saturation=1.3,
        shadows=1.1,
        highlights=0.9,
    )  # Only Phase 2 changes
    times.append((time.perf_counter() - start) * 1000)

print(f"Time: {np.mean(times):.3f} ms ({N / np.mean(times) * 1000 / 1e6:.1f}M points/sec)")

# Test combined (both phases)
print("\n[Combined: Phase 1 + Phase 2]")
times = []
for _ in range(100):
    start = time.perf_counter()
    result = lut.apply(
        colors,
        temperature=0.7,
        brightness=1.2,
        contrast=1.1,
        gamma=0.9,
        saturation=1.3,
        shadows=1.1,
        highlights=0.9,
    )
    times.append((time.perf_counter() - start) * 1000)

print(f"Time: {np.mean(times):.3f} ms ({N / np.mean(times) * 1000 / 1e6:.1f}M points/sec)")

# Analyze Phase 2 breakdown
print("\n" + "=" * 80)
print("PHASE 2 BREAKDOWN")
print("=" * 80)

# Test saturation only
print("\n[Saturation only]")
times = []
for _ in range(100):
    start = time.perf_counter()
    result = lut.apply(
        colors,
        saturation=1.3,
        temperature=0.5,
        brightness=1.0,
        contrast=1.0,
        gamma=1.0,
        shadows=1.0,
        highlights=1.0,
    )
    times.append((time.perf_counter() - start) * 1000)

print(f"Time: {np.mean(times):.3f} ms ({N / np.mean(times) * 1000 / 1e6:.1f}M points/sec)")

# Test shadows only
print("\n[Shadows only]")
times = []
for _ in range(100):
    start = time.perf_counter()
    result = lut.apply(
        colors,
        shadows=1.1,
        temperature=0.5,
        brightness=1.0,
        contrast=1.0,
        gamma=1.0,
        saturation=1.0,
        highlights=1.0,
    )
    times.append((time.perf_counter() - start) * 1000)

print(f"Time: {np.mean(times):.3f} ms ({N / np.mean(times) * 1000 / 1e6:.1f}M points/sec)")

# Test highlights only
print("\n[Highlights only]")
times = []
for _ in range(100):
    start = time.perf_counter()
    result = lut.apply(
        colors,
        highlights=0.9,
        temperature=0.5,
        brightness=1.0,
        contrast=1.0,
        gamma=1.0,
        saturation=1.0,
        shadows=1.0,
    )
    times.append((time.perf_counter() - start) * 1000)

print(f"Time: {np.mean(times):.3f} ms ({N / np.mean(times) * 1000 / 1e6:.1f}M points/sec)")

# Test saturation + shadows/highlights
print("\n[Saturation + Shadows + Highlights]")
times = []
for _ in range(100):
    start = time.perf_counter()
    result = lut.apply(
        colors,
        saturation=1.3,
        shadows=1.1,
        highlights=0.9,
        temperature=0.5,
        brightness=1.0,
        contrast=1.0,
        gamma=1.0,
    )
    times.append((time.perf_counter() - start) * 1000)

print(f"Time: {np.mean(times):.3f} ms ({N / np.mean(times) * 1000 / 1e6:.1f}M points/sec)")

print("\n" + "=" * 80)
print("OPTIMIZATION OPPORTUNITIES")
print("=" * 80)
print("""
1. Duplicate Luminance Calculation:
   - Saturation calculates luminance: L = 0.299*R + 0.587*G + 0.114*B
   - Shadows/Highlights recalculate the same luminance
   - Opportunity: Calculate once, reuse

2. Tensor Operations Overhead:
   - Multiple unsqueeze(), expand(), clamp() operations
   - Each creates temporary tensors
   - Opportunity: Fuse into single Numba kernel

3. No Parallel Processing:
   - PyTorch broadcasting is fast but not parallel on CPU
   - Opportunity: Use Numba prange for explicit parallelization

4. Potential Fused Kernel:
   ```python
   @njit(parallel=True)
   def fused_phase2_numba(colors, saturation, shadows, highlights, out):
       N = colors.shape[0]
       for i in prange(N):
           r, g, b = colors[i, 0], colors[i, 1], colors[i, 2]

           # Calculate luminance once
           lum = 0.299*r + 0.587*g + 0.114*b

           # Apply saturation
           if saturation != 1.0:
               r = lum + saturation * (r - lum)
               g = lum + saturation * (g - lum)
               b = lum + saturation * (b - lum)

           # Apply shadows/highlights
           if lum < 0.5:  # Shadow region
               if shadows != 1.0:
                   r *= shadows
                   g *= shadows
                   b *= shadows
           else:  # Highlight region
               if highlights != 1.0:
                   r *= highlights
                   g *= highlights
                   b *= highlights

           # Clamp and store
           out[i, 0] = min(max(r, 0.0), 1.0)
           out[i, 1] = min(max(g, 0.0), 1.0)
           out[i, 2] = min(max(b, 0.0), 1.0)
   ```

Expected speedup: 2-3x for Phase 2 operations
""")
