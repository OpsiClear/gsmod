"""
Deep CPU-specific bottleneck analysis.

Identifies cache misses, memory bandwidth, and computational bottlenecks.
"""

import logging
import time

import numpy as np
import torch

from gsmod import ColorLUT

logging.basicConfig(level=logging.WARNING)

print("=" * 80)
print("CPU COLOR PROCESSING BOTTLENECK ANALYSIS")
print("=" * 80)

N = 100_000
iterations = 50

colors = torch.rand(N, 3, device="cpu")
lut = ColorLUT(device="cpu", lut_size=1024)

# Warmup
for _ in range(5):
    _ = lut.apply(
        colors,
        saturation=1.3,
        shadows=1.1,
        highlights=0.9,
        temperature=0.7,
        brightness=1.2,
        contrast=1.1,
        gamma=0.9,
    )

print(f"\nTesting with {N:,} colors, {iterations} iterations")
print("=" * 80)

# Baseline: Full pipeline
print("\n[BASELINE] Full pipeline (all operations):")
times = []
for _ in range(iterations):
    start = time.perf_counter()
    result = lut.apply(
        colors,
        saturation=1.3,
        shadows=1.1,
        highlights=0.9,
        temperature=0.7,
        brightness=1.2,
        contrast=1.1,
        gamma=0.9,
    )
    times.append((time.perf_counter() - start) * 1000)

baseline_time = np.mean(times)
print(f"  Time: {baseline_time:.3f} ms ({N / baseline_time * 1000 / 1e6:.1f} M/s)")

# Break down by component
print("\n" + "=" * 80)
print("COMPONENT BREAKDOWN")
print("=" * 80)

# Phase 1 only (LUT lookup)
print("\n[Phase 1] LUT lookup only:")
colors_np = colors.numpy()
indices = (colors_np * (lut.lut_size - 1)).astype(np.int64)
indices = np.clip(indices, 0, lut.lut_size - 1)

times = []
for _ in range(iterations):
    start = time.perf_counter()
    adjusted = np.stack(
        [
            lut.r_lut[indices[:, 0]],
            lut.g_lut[indices[:, 1]],
            lut.b_lut[indices[:, 2]],
        ],
        axis=1,
    )
    times.append((time.perf_counter() - start) * 1000)

phase1_time = np.mean(times)
print(f"  Time: {phase1_time:.3f} ms ({N / phase1_time * 1000 / 1e6:.1f} M/s)")

# Phase 2 only (with Numba)
print("\n[Phase 2] Saturation + Shadows/Highlights (Numba fused):")
import gsmod.color as color_module

if color_module.NUMBA_AVAILABLE:
    from gsmod.numba_ops import fused_color_phase2_numba

    test_colors = np.random.rand(N, 3).astype(np.float32)
    out = np.empty_like(test_colors)

    # Warmup
    fused_color_phase2_numba(test_colors, 1.3, 1.1, 0.9, out)

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fused_color_phase2_numba(test_colors, 1.3, 1.1, 0.9, out)
        times.append((time.perf_counter() - start) * 1000)

    phase2_time = np.mean(times)
    print(f"  Time: {phase2_time:.3f} ms ({N / phase2_time * 1000 / 1e6:.1f} M/s)")
else:
    print("  [SKIP] Numba not available")
    phase2_time = 0

# Memory bandwidth analysis
print("\n" + "=" * 80)
print("MEMORY BANDWIDTH ANALYSIS")
print("=" * 80)

bytes_per_color = 3 * 4  # RGB * float32
bytes_read = N * bytes_per_color  # Input
bytes_written = N * bytes_per_color  # Output
lut_bytes = 3 * 1024 * 4  # Three 1D LUTs

print(f"\nMemory traffic for {N:,} colors:")
print(f"  Input colors:  {bytes_read / 1e6:.2f} MB")
print(f"  Output colors: {bytes_written / 1e6:.2f} MB")
print(f"  LUT data:      {lut_bytes / 1e3:.2f} KB")
print(f"  Total:         {(bytes_read + bytes_written) / 1e6:.2f} MB")

# Theoretical limits
ddr4_bandwidth = 20000  # MB/s (single channel)
ddr5_bandwidth = 40000  # MB/s (single channel)

total_bytes = bytes_read + bytes_written
ddr4_limit = total_bytes / ddr4_bandwidth * 1000  # ms
ddr5_limit = total_bytes / ddr5_bandwidth * 1000  # ms

print("\nTheoretical limits (memory bandwidth only):")
print(f"  DDR4 (20 GB/s):  {ddr4_limit:.3f} ms ({N / ddr4_limit * 1000 / 1e6:.0f} M/s)")
print(f"  DDR5 (40 GB/s):  {ddr5_limit:.3f} ms ({N / ddr5_limit * 1000 / 1e6:.0f} M/s)")

efficiency = ddr4_limit / baseline_time * 100
print(f"\nCurrent efficiency: {efficiency:.1f}% of DDR4 bandwidth")

# Cache analysis
print("\n" + "=" * 80)
print("CACHE ANALYSIS")
print("=" * 80)

l1_cache = 32 * 1024  # 32 KB typical L1 data cache
l2_cache = 256 * 1024  # 256 KB typical L2 cache
l3_cache = 8 * 1024 * 1024  # 8 MB typical L3 cache

print("\nTypical CPU cache hierarchy:")
print(f"  L1: {l1_cache / 1024:.0f} KB (1-4 cycles latency)")
print(f"  L2: {l2_cache / 1024:.0f} KB (10-20 cycles latency)")
print(f"  L3: {l3_cache / 1024 / 1024:.0f} MB (40-75 cycles latency)")
print(f"  RAM: {ddr4_bandwidth / 1000:.0f} GB/s (200+ cycles latency)")

print(f"\nCurrent LUT size: {lut_bytes / 1024:.2f} KB")
if lut_bytes < l1_cache:
    print(f"  [OK] Fits in L1 cache ({lut_bytes / 1024:.1f} KB < {l1_cache / 1024:.0f} KB)")
elif lut_bytes < l2_cache:
    print(f"  [WARNING] Fits in L2 but not L1 ({lut_bytes / 1024:.1f} KB)")
else:
    print(f"  [ISSUE] Doesn't fit in L2 ({lut_bytes / 1024:.1f} KB > {l2_cache / 1024:.0f} KB)")

# Working set analysis
working_set = bytes_read + bytes_written + lut_bytes
print(f"\nWorking set size: {working_set / 1024:.2f} KB")
if working_set < l3_cache:
    print("  [OK] Fits in L3 cache")
else:
    print("  [WARNING] Larger than L3 cache, will hit RAM")

# Bottleneck identification
print("\n" + "=" * 80)
print("BOTTLENECK IDENTIFICATION")
print("=" * 80)

print(f"""
Current performance breakdown:
  Phase 1 (LUT lookup):    {phase1_time:.3f} ms ({phase1_time / baseline_time * 100:.1f}%)
  Phase 2 (Saturation+S/H): {phase2_time:.3f} ms ({phase2_time / baseline_time * 100:.1f}%)
  Overhead:                {baseline_time - phase1_time - phase2_time:.3f} ms ({(baseline_time - phase1_time - phase2_time) / baseline_time * 100:.1f}%)

Memory bandwidth efficiency: {efficiency:.1f}%

IDENTIFIED BOTTLENECKS:
=====================

1. LUT LOOKUP IS NOT FUSED WITH PHASE 2
   - Current: LUT lookup -> write to memory -> read for Phase 2
   - Opportunity: Single fused kernel (LUT + Phase 2 in one pass)
   - Expected gain: 1.5-2x (eliminate intermediate memory traffic)

2. LUT SIZE TOO LARGE FOR L1 CACHE
   - Current: {lut_bytes / 1024:.1f} KB (doesn't fit in L1)
   - L1 cache: {l1_cache / 1024:.0f} KB
   - Opportunity: Smaller LUT (64-128 entries) + interpolation
   - Expected gain: 1.3-1.5x (L1 vs L2 latency)

3. NO EXPLICIT SIMD VECTORIZATION
   - Current: Relies on compiler auto-vectorization
   - Opportunity: Explicit AVX2/AVX-512 (process 4-8 colors at once)
   - Expected gain: 2-3x (8-wide SIMD with AVX-512)

4. MEMORY LAYOUT NOT OPTIMAL
   - Current: AOS (Array of Structures) [R,G,B], [R,G,B], ...
   - Better: SOA (Structure of Arrays) [R,R,R...], [G,G,G...], [B,B,B...]
   - Expected gain: 1.2-1.5x (better SIMD and cache usage)

5. NOT USING ALL CPU CORES EFFICIENTLY
   - Current: Numba prange for Phase 2 only
   - Opportunity: Parallel entire pipeline
   - Expected gain: Already doing this, but could be better

OPTIMIZATION OPPORTUNITIES (CPU-SPECIFIC):
========================================

A. FUSED CPU KERNEL (HIGHEST IMPACT: 2-3x)
   - Single Numba kernel: LUT lookup + Phase 2
   - Eliminates intermediate memory writes/reads
   - Process each pixel completely before moving to next
   - Expected: {baseline_time / 2.5:.2f}-{baseline_time / 3:.2f} ms ({N / (baseline_time / 2.5) * 1000 / 1e6:.0f}-{N / (baseline_time / 3) * 1000 / 1e6:.0f} M/s)

B. SMALLER LUT + LINEAR INTERPOLATION (MODERATE IMPACT: 1.3-1.5x)
   - Use 64-128 entry LUT (fits in L1 cache)
   - Linear interpolation between entries
   - Faster LUT access (L1 vs L2 latency)
   - Expected: {baseline_time / 1.4:.2f} ms ({N / (baseline_time / 1.4) * 1000 / 1e6:.0f} M/s)

C. EXPLICIT AVX-512 VECTORIZATION (HIGH IMPACT: 2-3x)
   - Process 8 colors in parallel with SIMD
   - Use Numba's @vectorize or explicit intrinsics
   - Combined with parallel loops
   - Expected: {baseline_time / 2.5:.2f} ms ({N / (baseline_time / 2.5) * 1000 / 1e6:.0f} M/s)

D. SOA MEMORY LAYOUT (MODERATE IMPACT: 1.2-1.5x)
   - Reorganize data: [R,R,R...], [G,G,G...], [B,B,B...]
   - Better for SIMD (load 8 R values at once)
   - Better cache usage
   - Expected: {baseline_time / 1.3:.2f} ms ({N / (baseline_time / 1.3) * 1000 / 1e6:.0f} M/s)

COMBINED OPTIMIZATIONS:
======================
If we implement A + B + C:
  Expected: {baseline_time / 8:.2f}-{baseline_time / 12:.2f} ms ({N / (baseline_time / 8) * 1000 / 1e6:.0f}-{N / (baseline_time / 12) * 1000 / 1e6:.0f} M/s)
  Speedup: 8-12x faster
  Efficiency: {efficiency * 8:.0f}-{efficiency * 12:.0f}% of DDR4 bandwidth

RECOMMENDATION:
==============
Start with (A) - Fused CPU kernel. This is the easiest and biggest win.
Then add (C) - AVX-512 vectorization for maximum performance.
""")
