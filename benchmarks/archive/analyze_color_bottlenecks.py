"""
Ultra-deep analysis of ColorLUT bottlenecks.

Identifies exact slowdowns and potential optimizations.
"""

import logging
import time

import numpy as np
import torch

from gsmod import ColorLUT

logging.basicConfig(level=logging.WARNING)

print("=" * 80)
print("COLOR PROCESSING BOTTLENECK ANALYSIS")
print("=" * 80)

# Test different scenarios
N = 100_000
iterations = 50

scenarios = [
    ("GPU tensors -> GPU ColorLUT", "cuda", "cuda"),
    ("CPU tensors -> CPU ColorLUT", "cpu", "cpu"),
    ("GPU tensors -> CPU ColorLUT (WORST CASE)", "cuda", "cpu"),
]

print(f"\nTesting with {N:,} colors, {iterations} iterations\n")

for scenario_name, data_device, lut_device in scenarios:
    print(f"\n[{scenario_name}]")
    print("-" * 80)

    if not torch.cuda.is_available() and "cuda" in (data_device, lut_device):
        print("  SKIPPED (CUDA not available)")
        continue

    # Create data
    colors = torch.rand(N, 3, device=data_device)
    lut = ColorLUT(device=lut_device, lut_size=1024)

    # Warmup
    for _ in range(5):
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

    # Benchmark total time
    if data_device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
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
    if data_device == "cuda" or lut_device == "cuda":
        torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) / iterations * 1000

    print(f"  Total time:      {total_time:.3f} ms")
    print(f"  Throughput:      {N / total_time * 1000 / 1e6:.1f} M colors/sec")
    print(f"  Result device:   {result.device}")
    print(f"  Input device:    {colors.device}")

# Now let's break down the GPU path into components
print("\n" + "=" * 80)
print("GPU PATH BREAKDOWN (if available)")
print("=" * 80)

if torch.cuda.is_available():
    colors = torch.rand(N, 3, device="cuda")
    lut = ColorLUT(device="cuda", lut_size=1024)

    # Pre-compile LUTs
    lut.apply(colors, temperature=0.7, brightness=1.2, contrast=1.1, gamma=0.9)

    # Test Phase 1 only
    print("\n[Phase 1: LUT lookup only]")
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        # Manually call Phase 1
        indices = (colors * (lut.lut_size - 1)).long().clamp(0, lut.lut_size - 1)
        adjusted = torch.stack(
            [
                lut.r_lut[indices[:, 0]],
                lut.g_lut[indices[:, 1]],
                lut.b_lut[indices[:, 2]],
            ],
            dim=1,
        )
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    phase1_time = np.mean(times)
    print(f"  Time: {phase1_time:.3f} ms ({N / phase1_time * 1000 / 1e6:.1f} M/s)")

    # Test Phase 2 only (standard PyTorch path)
    print("\n[Phase 2: Saturation + Shadows/Highlights (PyTorch)]")
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        test_colors = torch.rand(N, 3, device="cuda")
        start = time.perf_counter()

        # Saturation
        luminance = (
            0.299 * test_colors[:, 0] + 0.587 * test_colors[:, 1] + 0.114 * test_colors[:, 2]
        )
        luminance = luminance.unsqueeze(1).expand_as(test_colors)
        test_colors = torch.lerp(luminance, test_colors, 1.3).clamp(0, 1)

        # Shadows/Highlights
        luminance = (
            0.299 * test_colors[:, 0] + 0.587 * test_colors[:, 1] + 0.114 * test_colors[:, 2]
        ).unsqueeze(1)
        shadow_mask = (luminance < 0.5).float()
        highlight_mask = (luminance >= 0.5).float()
        shadow_adj = test_colors * shadow_mask * (1.1 - 1.0)
        highlight_adj = test_colors * highlight_mask * (0.9 - 1.0)
        test_colors = (test_colors + shadow_adj + highlight_adj).clamp(0, 1)

        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    phase2_time = np.mean(times)
    print(f"  Time: {phase2_time:.3f} ms ({N / phase2_time * 1000 / 1e6:.1f} M/s)")

    print("\n[Combined estimate]")
    print(f"  Phase 1 + Phase 2: {phase1_time + phase2_time:.3f} ms")
    print(f"  Throughput: {N / (phase1_time + phase2_time) * 1000 / 1e6:.1f} M/s")

# Memory bandwidth analysis
print("\n" + "=" * 80)
print("MEMORY BANDWIDTH ANALYSIS")
print("=" * 80)

bytes_read = N * 3 * 4  # Input colors (float32)
bytes_written = N * 3 * 4  # Output colors (float32)
lut_bytes = 3 * 1024 * 4  # Three 1D LUTs (negligible)
total_bytes = bytes_read + bytes_written

print(f"\nFor {N:,} colors:")
print(f"  Data read:    {bytes_read / 1e6:.2f} MB")
print(f"  Data written: {bytes_written / 1e6:.2f} MB")
print(f"  LUT data:     {lut_bytes / 1e3:.2f} KB (cached)")
print(f"  Total:        {total_bytes / 1e6:.2f} MB")

# Theoretical limits
cpu_bandwidth = 20000  # MB/s (typical DDR4)
gpu_bandwidth = 400000  # MB/s (typical GPU like RTX 3080)

cpu_limit = total_bytes / cpu_bandwidth * 1000  # ms
gpu_limit = total_bytes / gpu_bandwidth * 1000  # ms

print("\nTheoretical limits (memory bandwidth only):")
print(f"  CPU (DDR4 ~20 GB/s):   {cpu_limit:.3f} ms ({N / cpu_limit * 1000 / 1e6:.0f} M/s)")
print(f"  GPU (GDDR6 ~400 GB/s): {gpu_limit:.3f} ms ({N / gpu_limit * 1000 / 1e6:.0f} M/s)")

print("\n" + "=" * 80)
print("OPTIMIZATION OPPORTUNITIES")
print("=" * 80)

print("""
IDENTIFIED BOTTLENECKS:
======================

1. GPU PATH IS NOT OPTIMIZED
   - Phase 2 operations use standard PyTorch (no custom kernels)
   - Multiple temporary allocations (luminance, masks)
   - No fusion of operations
   - Kernel launch overhead for each operation

2. CPU-GPU TRANSFER PENALTY
   - If user passes GPU tensors to CPU ColorLUT, implicit transfer occurs
   - No warning or automatic device handling

3. NO FUSED LUT+PHASE2 KERNEL
   - Currently: LUT lookup -> write to memory -> read for Phase 2
   - Could be: Single fused kernel (LUT + Phase 2 in one pass)
   - Eliminates intermediate memory traffic

RADICAL OPTIMIZATIONS:
=====================

A. FUSED CUDA/TRITON KERNEL (HIGHEST IMPACT)
   - Single kernel: LUT lookup + saturation + shadows/highlights
   - Expected: 5-10x faster than current GPU path
   - Effort: Medium (can use Triton for easier development)

B. GPU-NATIVE PHASE 2 KERNEL
   - Custom CUDA kernel for Phase 2 only
   - Expected: 2-3x faster than PyTorch path
   - Effort: Low (similar to CPU Numba kernel)

C. TEXTURE MEMORY FOR LUT
   - Use CUDA texture memory for LUT lookups (hardware interpolation)
   - Expected: 1.5-2x faster LUT lookup
   - Effort: Medium

D. FLOAT16 PRECISION MODE
   - Option for fp16 processing (2x memory bandwidth)
   - Expected: 1.5-2x faster (memory bound)
   - Effort: Low

E. BATCHED PROCESSING WITH STREAMS
   - Process large batches in chunks with CUDA streams
   - Expected: Better GPU utilization for very large batches
   - Effort: Medium

RECOMMENDATION:
==============
Start with (A) - Fused CUDA/Triton kernel. This is the biggest win.
""")
