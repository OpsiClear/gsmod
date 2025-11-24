"""
Benchmark ColorLUT performance (CPU).
"""

import time

import numpy as np
import torch

from gsmod import ColorLUT


def benchmark_function(func, warmup: int = 10, iterations: int = 100):
    """Benchmark a function and return average time in milliseconds."""
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)

    return np.mean(times), np.std(times)


print("=" * 80)
print("COLOR LUT BENCHMARK (CPU)")
print("=" * 80)

batch_sizes = [1_000, 10_000, 100_000, 1_000_000]

for batch_size in batch_sizes:
    print(f"\nBatch Size: {batch_size:,} points")

    # Generate test data
    colors = torch.rand(batch_size, 3, device="cpu")

    # Create ColorLUT
    lut = ColorLUT(device="cpu", lut_size=1024)

    def test_apply():
        return lut.apply(
            colors,
            temperature=0.7,
            brightness=1.2,
            contrast=1.1,
            gamma=0.9,
            saturation=1.3,
            shadows=1.1,
            highlights=0.9,
        )

    mean_time, std_time = benchmark_function(test_apply)
    throughput = (batch_size / mean_time) * 1000

    print(f"  Time:       {mean_time:8.3f} +/- {std_time:6.3f} ms")
    print(f"  Throughput: {throughput:12,.0f} points/sec")

# Individual operations
print("\n" + "=" * 80)
print("INDIVIDUAL OPERATIONS (100K points)")
print("=" * 80)

batch_size = 100_000
colors = torch.rand(batch_size, 3, device="cpu")
lut = ColorLUT(device="cpu")

operations = {
    "Temperature only": {"temperature": 0.7},
    "Brightness only": {"brightness": 1.5},
    "Contrast only": {"contrast": 1.5},
    "Gamma only": {"gamma": 0.8},
    "Saturation only": {"saturation": 1.5},
    "Shadows only": {"shadows": 1.3},
    "Highlights only": {"highlights": 0.7},
    "All operations": {
        "temperature": 0.7,
        "brightness": 1.2,
        "contrast": 1.1,
        "gamma": 0.9,
        "saturation": 1.3,
        "shadows": 1.1,
        "highlights": 0.9,
    },
}

for op_name, params in operations.items():

    def test_op():
        return lut.apply(colors, **params)

    mean_time, std_time = benchmark_function(test_op)
    throughput = (batch_size / mean_time) * 1000

    print(f"\n{op_name}:")
    print(f"  Time:       {mean_time:8.3f} +/- {std_time:6.3f} ms")
    print(f"  Throughput: {throughput:12,.0f} points/sec")

# LUT resolution
print("\n" + "=" * 80)
print("LUT RESOLUTION (100K points)")
print("=" * 80)

lut_sizes = [256, 512, 1024, 2048, 4096]
colors = torch.rand(100_000, 3, device="cpu")

for lut_size in lut_sizes:
    lut = ColorLUT(device="cpu", lut_size=lut_size)

    def test_lut():
        return lut.apply(colors, brightness=1.2, contrast=1.1)

    mean_time, std_time = benchmark_function(test_lut)
    throughput = (100_000 / mean_time) * 1000
    memory_kb = (lut_size * 3 * 4) / 1024

    print(f"\nLUT {lut_size}:")
    print(f"  Time:       {mean_time:8.3f} +/- {std_time:6.3f} ms")
    print(f"  Throughput: {throughput:12,.0f} points/sec")
    print(f"  Memory:     {memory_kb:.1f} KB")

print("\n" + "=" * 80)
