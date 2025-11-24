"""
Benchmark color processing performance (pure NumPy/Numba, no PyTorch).
"""

import time

import numpy as np

from gsmod import Color

N = 100_000
NUM_ITERATIONS = 100

print("=" * 80)
print("COLOR PROCESSING BENCHMARK (NumPy/Numba - Raw Arrays)")
print(f"Testing with {N:,} colors, {NUM_ITERATIONS} iterations")
print("=" * 80)

# Setup
colors_np = np.random.rand(N, 3).astype(np.float32)

# Create Color pipeline with all adjustments
pipeline = (
    Color()
    .temperature(0.6)
    .brightness(1.2)
    .contrast(1.1)
    .saturation(1.3)
    .shadows(1.1)
    .highlights(0.9)
    .compile()
)  # Pre-compile for maximum performance

# Warmup
print("\nWarming up...")
colors_warmup = colors_np.copy()
for _ in range(20):
    pipeline._apply_to_colors(colors_warmup, inplace=True)

# Benchmark inplace (zero-copy)
print(f"\nBenchmarking _apply_to_colors(inplace=True) - {NUM_ITERATIONS} iterations...")
times_inplace = []
for _ in range(NUM_ITERATIONS):
    colors_test = colors_np.copy()
    start = time.perf_counter()
    pipeline._apply_to_colors(colors_test, inplace=True)
    times_inplace.append((time.perf_counter() - start) * 1000)

mean_time_inplace = np.mean(times_inplace)
std_time_inplace = np.std(times_inplace)

print("\nResults (inplace=True - zero-copy):")
print(f"  Time:       {mean_time_inplace:.3f} ms +/- {std_time_inplace:.3f} ms")
print(f"  Throughput: {N / mean_time_inplace * 1000 / 1e6:.0f} M colors/sec")

# Benchmark with copy
print(f"\nBenchmarking _apply_to_colors(inplace=False) - {NUM_ITERATIONS} iterations...")
times_copy = []
for _ in range(NUM_ITERATIONS):
    start = time.perf_counter()
    result = pipeline._apply_to_colors(colors_np, inplace=False)
    times_copy.append((time.perf_counter() - start) * 1000)

mean_time_copy = np.mean(times_copy)
std_time_copy = np.std(times_copy)

print("\nResults (inplace=False - with copy):")
print(f"  Time:       {mean_time_copy:.3f} ms +/- {std_time_copy:.3f} ms")
print(f"  Throughput: {N / mean_time_copy * 1000 / 1e6:.0f} M colors/sec")

# Speedup comparison
print("\n" + "=" * 80)
print("SPEEDUP COMPARISON")
print("=" * 80)
print(f"inplace=True vs inplace=False: {mean_time_copy / mean_time_inplace:.2f}x faster")

# Test different batch sizes
print("\n" + "=" * 80)
print("BATCH SIZE SCALING")
print("=" * 80)

batch_sizes = [1_000, 10_000, 100_000, 1_000_000]

for N_test in batch_sizes:
    colors_test = np.random.rand(N_test, 3).astype(np.float32)

    # Warmup
    colors_warmup = colors_test.copy()
    for _ in range(5):
        pipeline._apply_to_colors(colors_warmup, inplace=True)

    times_test = []
    for _ in range(20):
        colors_iter = colors_test.copy()
        start = time.perf_counter()
        pipeline._apply_to_colors(colors_iter, inplace=True)
        times_test.append((time.perf_counter() - start) * 1000)

    test_time = np.mean(times_test)
    throughput = N_test / test_time * 1000 / 1e6

    print(f"N={N_test:>9,}: {test_time:>7.3f} ms ({throughput:>5.0f} M colors/s)")

# Test LUT compilation overhead
print("\n" + "=" * 80)
print("LUT COMPILATION OVERHEAD")
print("=" * 80)

# Test compilation time
start = time.perf_counter()
test_pipeline = (
    Color()
    .temperature(0.6)
    .brightness(1.2)
    .contrast(1.1)
    .gamma(1.05)
    .saturation(1.3)
    .shadows(1.1)
    .highlights(0.9)
    .compile()
)
compile_time = (time.perf_counter() - start) * 1000

print(f"LUT compilation time: {compile_time:.3f} ms")

# Test recompilation detection
test_pipeline.brightness(1.3)  # Change parameter
start = time.perf_counter()
test_pipeline.compile()
recompile_time = (time.perf_counter() - start) * 1000

print(f"LUT recompilation time: {recompile_time:.3f} ms")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Best throughput: {N / mean_time_inplace * 1000 / 1e6:.0f}M colors/sec")
print(f"Best latency:    {mean_time_inplace:.3f} ms for {N:,} colors")
print("=" * 80)
