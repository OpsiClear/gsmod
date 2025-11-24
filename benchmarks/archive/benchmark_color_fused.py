"""
Benchmark fused color Phase 2 kernel vs standard path.
"""

import time

import numpy as np
import torch

from gsmod import ColorLUT

print("=" * 80)
print("COLOR FUSED KERNEL PERFORMANCE BENCHMARK")
print("=" * 80)

N = 100_000
colors = torch.rand(N, 3, device="cpu")

lut = ColorLUT(device="cpu", lut_size=1024)

# Disable Numba for baseline
import gsmod.color as color_module

original_numba_available = color_module.NUMBA_AVAILABLE

print(f"\n[Benchmarking {N:,} colors, 100 iterations]")

# Test 1: Standard path (Numba disabled)
print("\n[1] Standard Path (PyTorch operations):")
color_module.NUMBA_AVAILABLE = False

times = []
for _ in range(100):
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

standard_time = np.mean(times)
standard_std = np.std(times)
print(f"  Time: {standard_time:.3f} ms +/- {standard_std:.3f} ms")
print(f"  Throughput: {N / standard_time * 1000 / 1e6:.1f}M colors/sec")

# Test 2: Fused kernel (Numba enabled)
print("\n[2] Fused Kernel (Numba parallel loop):")
color_module.NUMBA_AVAILABLE = True

times = []
for _ in range(100):
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

fused_time = np.mean(times)
fused_std = np.std(times)
print(f"  Time: {fused_time:.3f} ms +/- {fused_std:.3f} ms")
print(f"  Throughput: {N / fused_time * 1000 / 1e6:.1f}M colors/sec")

# Overall improvement
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"\nStandard path:  {standard_time:.3f} ms ({N / standard_time * 1000 / 1e6:.1f}M/s)")
print(f"Fused kernel:   {fused_time:.3f} ms ({N / fused_time * 1000 / 1e6:.1f}M/s)")
print(f"\nSpeedup: {standard_time / fused_time:.2f}x faster")
print(f"Time saved: {standard_time - fused_time:.3f} ms per {N:,} colors")
print(f"For 1M colors: {(standard_time - fused_time) * 10:.1f} ms saved")

# Test different batch sizes
print("\n" + "=" * 80)
print("BATCH SIZE SCALING")
print("=" * 80)

batch_sizes = [1_000, 10_000, 100_000, 1_000_000]

for batch_n in batch_sizes:
    test_colors = torch.rand(batch_n, 3, device="cpu")

    # Standard
    color_module.NUMBA_AVAILABLE = False
    times = []
    for _ in range(20):
        start = time.perf_counter()
        lut.apply(
            test_colors,
            saturation=1.3,
            shadows=1.1,
            highlights=0.9,
            temperature=0.7,
            brightness=1.2,
            contrast=1.1,
            gamma=0.9,
        )
        times.append((time.perf_counter() - start) * 1000)
    std_time = np.mean(times)

    # Fused
    color_module.NUMBA_AVAILABLE = True
    times = []
    for _ in range(20):
        start = time.perf_counter()
        lut.apply(
            test_colors,
            saturation=1.3,
            shadows=1.1,
            highlights=0.9,
            temperature=0.7,
            brightness=1.2,
            contrast=1.1,
            gamma=0.9,
        )
        times.append((time.perf_counter() - start) * 1000)
    fused_time = np.mean(times)

    speedup = std_time / fused_time
    print(f"\nN={batch_n:>9,}:")
    print(f"  Standard: {std_time:>6.2f} ms ({batch_n / std_time * 1000 / 1e6:>5.1f}M/s)")
    print(f"  Fused:    {fused_time:>6.2f} ms ({batch_n / fused_time * 1000 / 1e6:>5.1f}M/s)")
    print(f"  Speedup:  {speedup:.2f}x")

# Restore original state
color_module.NUMBA_AVAILABLE = original_numba_available

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
The fused Numba kernel provides significant speedup by:
1. Calculating luminance once (not twice)
2. Eliminating temporary tensor allocations
3. Explicit parallelization with prange
4. Better memory locality (process each pixel completely)

This optimization is automatic - no code changes required!
Just install Numba and use ColorLUT on CPU.
""")
