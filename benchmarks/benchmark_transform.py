"""
Benchmark 3D Gaussian transform performance (CPU with fused Numba kernel).
"""

import time

import numpy as np

from gsmod import Transform

N = 1_000_000
NUM_ITERATIONS = 100

print("=" * 80)
print("3D GAUSSIAN TRANSFORM BENCHMARK")
print(f"Testing with {N:,} Gaussians, {NUM_ITERATIONS} iterations")
print("=" * 80)

# Setup
means_np = np.random.randn(N, 3).astype(np.float32)
quats_np = np.random.randn(N, 4).astype(np.float32)
quats_np = quats_np / np.linalg.norm(quats_np, axis=1, keepdims=True)
scales_np = np.random.rand(N, 3).astype(np.float32)

translation_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)
rotation_np = np.array([0.9239, 0.0, 0.0, 0.3827], dtype=np.float32)
scale_factor = 2.0

# Create Transform pipeline with all transformations
pipeline = (
    Transform().scale(scale_factor).rotate_quat(rotation_np).translate(translation_np).compile()
)  # Pre-compile for maximum performance

# Warmup
print("\nWarming up...")
for _ in range(20):
    means_test = means_np.copy()
    quats_test = quats_np.copy()
    scales_test = scales_np.copy()
    pipeline.apply(means_test, quats_test, scales_test, inplace=True)

# Benchmark
print(f"Benchmarking {NUM_ITERATIONS} iterations...")
times = []
for _ in range(NUM_ITERATIONS):
    means_test = means_np.copy()
    quats_test = quats_np.copy()
    scales_test = scales_np.copy()

    start = time.perf_counter()
    pipeline.apply(means_test, quats_test, scales_test, inplace=True)
    times.append((time.perf_counter() - start) * 1000)

mean_time = np.mean(times)
std_time = np.std(times)

print("\nResults (1M Gaussians):")
print(f"  Time:       {mean_time:.3f} ms +/- {std_time:.3f} ms")
print(f"  Throughput: {N / mean_time * 1000 / 1e6:.1f}M Gaussians/sec")

# Test different batch sizes
print("\n" + "=" * 80)
print("BATCH SIZE SCALING")
print("=" * 80)

batch_sizes = [10_000, 100_000, 500_000, 1_000_000, 2_000_000]

for N_test in batch_sizes:
    means_test = np.random.randn(N_test, 3).astype(np.float32)
    quats_test = np.random.randn(N_test, 4).astype(np.float32)
    quats_test = quats_test / np.linalg.norm(quats_test, axis=1, keepdims=True)
    scales_test = np.random.rand(N_test, 3).astype(np.float32)

    # Warmup
    for _ in range(5):
        means_warmup = means_test.copy()
        quats_warmup = quats_test.copy()
        scales_warmup = scales_test.copy()
        pipeline.apply(means_warmup, quats_warmup, scales_warmup, inplace=True)

    times_test = []
    for _ in range(20):
        means_iter = means_test.copy()
        quats_iter = quats_test.copy()
        scales_iter = scales_test.copy()

        start = time.perf_counter()
        pipeline.apply(means_iter, quats_iter, scales_iter, inplace=True)
        times_test.append((time.perf_counter() - start) * 1000)

    test_time = np.mean(times_test)
    throughput = N_test / test_time * 1000 / 1e6

    print(f"N={N_test:>9,}: {test_time:>6.2f} ms ({throughput:>6.1f}M G/s)")

# Test matrix compilation overhead
print("\n" + "=" * 80)
print("MATRIX COMPILATION OVERHEAD")
print("=" * 80)

# Test compilation time
start = time.perf_counter()
test_pipeline = Transform().scale(2.0).rotate_quat(rotation_np).translate(translation_np).compile()
compile_time = (time.perf_counter() - start) * 1000

print(f"Matrix compilation time: {compile_time:.3f} ms")

# Test recompilation detection
test_pipeline.scale(3.0)  # Change parameter
start = time.perf_counter()
test_pipeline.compile()
recompile_time = (time.perf_counter() - start) * 1000

print(f"Matrix recompilation time: {recompile_time:.3f} ms")

# Real-world use case
print("\n" + "=" * 80)
print("REAL-WORLD USE CASES")
print("=" * 80)

print("\nAnimation processing (1M Gaussians):")
print(f"  1 frame:    {mean_time:.1f} ms")
print(f"  100 frames: {mean_time * 100:.0f} ms ({mean_time * 100 / 1000:.2f}s)")
print(f"  1000 frames: {mean_time * 1000 / 1000:.1f}s")

print("\nReal-time rendering:")
print(f"  Max FPS:    {1000 / mean_time:.0f} FPS")
print(f"  Frame time: {mean_time:.2f} ms (target: 16.67ms for 60 FPS)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Best throughput: {N / mean_time * 1000 / 1e6:.1f}M Gaussians/sec")
print(f"Best latency:    {mean_time:.3f} ms for {N:,} Gaussians")
print("=" * 80)
