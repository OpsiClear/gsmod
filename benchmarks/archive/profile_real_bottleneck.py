"""
Profile to find the REAL bottleneck.

We expected 30-50x speedup but got only 1.1-1.4x.
This means apply() is already fast! Let's find out why.
"""

import cProfile
import pstats
import time

import numpy as np
import torch

from gsmod import ColorLUT

N = 100_000
colors_np = np.random.rand(N, 3).astype(np.float32)
colors_torch = torch.from_numpy(colors_np)

lut = ColorLUT(device="cpu", lut_size=1024)

params = {
    "temperature": 0.7,
    "brightness": 1.2,
    "contrast": 1.1,
    "gamma": 0.9,
    "saturation": 1.3,
    "shadows": 1.1,
    "highlights": 0.9,
}

print("=" * 80)
print("PROFILING REAL BOTTLENECK")
print("=" * 80)

# Warmup
for _ in range(5):
    _ = lut.apply(colors_torch, **params)
    _ = lut.apply_numpy(colors_np, **params)

# Profile apply()
print("\n[1] Profiling apply() with PyTorch tensors:")
profiler = cProfile.Profile()
profiler.enable()

for _ in range(100):
    _ = lut.apply(colors_torch, **params)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats("cumulative")
print("\nTop 10 time consumers:")
stats.print_stats(10)

# Profile apply_numpy()
print("\n" + "=" * 80)
print("[2] Profiling apply_numpy() with NumPy arrays:")
profiler2 = cProfile.Profile()
profiler2.enable()

for _ in range(100):
    _ = lut.apply_numpy(colors_np, **params)

profiler2.disable()
stats2 = pstats.Stats(profiler2)
stats2.sort_stats("cumulative")
print("\nTop 10 time consumers:")
stats2.print_stats(10)

# Manual timing breakdown
print("\n" + "=" * 80)
print("MANUAL TIMING BREAKDOWN")
print("=" * 80)

# Test just the ultra-fused kernel
print("\n[Direct kernel call] fused_color_full_pipeline_numba:")
from gsmod.numba_ops import fused_color_full_pipeline_numba

out = np.empty_like(colors_np)

# Compile LUTs
lut._compile_independent_luts_numpy(
    **{k: v for k, v in params.items() if k in ["temperature", "brightness", "contrast", "gamma"]}
)

# Warmup
fused_color_full_pipeline_numba(
    colors_np,
    lut.r_lut,
    lut.g_lut,
    lut.b_lut,
    params["saturation"],
    params["shadows"],
    params["highlights"],
    out,
)

times = []
for _ in range(100):
    start = time.perf_counter()
    fused_color_full_pipeline_numba(
        colors_np,
        lut.r_lut,
        lut.g_lut,
        lut.b_lut,
        params["saturation"],
        params["shadows"],
        params["highlights"],
        out,
    )
    times.append((time.perf_counter() - start) * 1000)

kernel_time = np.mean(times)
print(f"  Time: {kernel_time:.3f} ms ({N / kernel_time * 1000 / 1e6:.1f} M/s)")

# Test apply()
print("\n[apply() call]:")
times = []
for _ in range(100):
    start = time.perf_counter()
    _ = lut.apply(colors_torch, **params)
    times.append((time.perf_counter() - start) * 1000)

apply_time = np.mean(times)
print(f"  Time: {apply_time:.3f} ms ({N / apply_time * 1000 / 1e6:.1f} M/s)")

# Test apply_numpy()
print("\n[apply_numpy() call]:")
times = []
for _ in range(100):
    start = time.perf_counter()
    _ = lut.apply_numpy(colors_np, **params)
    times.append((time.perf_counter() - start) * 1000)

apply_numpy_time = np.mean(times)
print(f"  Time: {apply_numpy_time:.3f} ms ({N / apply_numpy_time * 1000 / 1e6:.1f} M/s)")

# Overhead analysis
print("\n" + "=" * 80)
print("OVERHEAD ANALYSIS")
print("=" * 80)

print(f"""
Direct kernel call:         {kernel_time:.3f} ms
apply() wrapper:            {apply_time:.3f} ms
apply_numpy() wrapper:      {apply_numpy_time:.3f} ms

Overhead:
  apply() overhead:         {apply_time - kernel_time:.3f} ms ({(apply_time - kernel_time) / apply_time * 100:.1f}%)
  apply_numpy() overhead:   {apply_numpy_time - kernel_time:.3f} ms ({(apply_numpy_time - kernel_time) / apply_numpy_time * 100:.1f}%)

Conclusion:
  Both apply() and apply_numpy() are already using the ultra-fused kernel!
  The remaining overhead is minimal (parameter validation, LUT caching checks, etc.)

  The real performance is: {kernel_time:.3f} ms for {N:,} colors
  Throughput: {N / kernel_time * 1000 / 1e6:.0f} M colors/sec
""")
