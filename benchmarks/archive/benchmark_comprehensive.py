"""
Comprehensive benchmark of all ColorLUT APIs and operations.

Tests:
1. All three APIs: apply(), apply_numpy(), apply_numpy_inplace()
2. Different batch sizes: 1K, 10K, 100K, 1M colors
3. LUT compilation overhead
4. Parameter variation overhead
5. Memory allocation overhead
"""

import time

import numpy as np

from gsmod import ColorLUT

print("=" * 80)
print("COMPREHENSIVE COLOR PROCESSING BENCHMARK")
print("=" * 80)
print()


def benchmark_api(api_name, func, colors, warmup=10, iterations=100):
    """Benchmark a single API function."""
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        times.append((time.perf_counter() - start) * 1000)

    return {
        "name": api_name,
        "mean": np.mean(times),
        "std": np.std(times),
        "min": np.min(times),
        "max": np.max(times),
        "throughput": len(colors) / np.mean(times) * 1000 / 1e6,  # M colors/sec
    }


# ============================================================================
# BENCHMARK 1: API Comparison (100K colors)
# ============================================================================

print("[BENCHMARK 1] API Comparison (100K colors)")
print("-" * 80)

N = 100_000
colors = np.random.rand(N, 3).astype(np.float32)
out = np.empty_like(colors)

lut = ColorLUT(device="cpu")

# Pre-compile LUTs to exclude compilation overhead
lut.apply_numpy(colors[:10], temperature=0.5, brightness=1.2, saturation=1.3)

results = []

# Test apply()
result = benchmark_api(
    "apply()",
    lambda: lut.apply(colors, temperature=0.5, brightness=1.2, saturation=1.3),
    colors,
)
results.append(result)

# Test apply_numpy()
result = benchmark_api(
    "apply_numpy()",
    lambda: lut.apply_numpy(colors, temperature=0.5, brightness=1.2, saturation=1.3),
    colors,
)
results.append(result)

# Test apply_numpy_inplace()
result = benchmark_api(
    "apply_numpy_inplace()",
    lambda: lut.apply_numpy_inplace(colors, out, temperature=0.5, brightness=1.2, saturation=1.3),
    colors,
)
results.append(result)

print(f"\n{'API Method':<25s} {'Time (ms)':<15s} {'Throughput':<15s} {'vs apply()':<10s}")
print("-" * 80)
baseline_time = results[0]["mean"]
for r in results:
    speedup = baseline_time / r["mean"]
    print(
        f"{r['name']:<25s} {r['mean']:>8.3f} +/- {r['std']:>5.3f}   "
        f"{r['throughput']:>6.0f} M/s        {speedup:>5.2f}x"
    )

# ============================================================================
# BENCHMARK 2: Batch Size Scaling
# ============================================================================

print("\n" + "=" * 80)
print("[BENCHMARK 2] Batch Size Scaling (apply_numpy_inplace)")
print("-" * 80)

batch_sizes = [1_000, 10_000, 100_000, 1_000_000]
scaling_results = []

for batch_size in batch_sizes:
    colors = np.random.rand(batch_size, 3).astype(np.float32)
    out = np.empty_like(colors)

    lut = ColorLUT(device="cpu")
    # Pre-compile
    lut.apply_numpy_inplace(colors[:10], out[:10], temperature=0.5, brightness=1.2, saturation=1.3)

    result = benchmark_api(
        f"{batch_size:,} colors",
        lambda: lut.apply_numpy_inplace(
            colors, out, temperature=0.5, brightness=1.2, saturation=1.3
        ),
        colors,
        warmup=5,
        iterations=50 if batch_size <= 100_000 else 20,
    )
    scaling_results.append(result)

print(f"\n{'Batch Size':<20s} {'Time (ms)':<15s} {'Throughput':<15s} {'Time/Color':<15s}")
print("-" * 80)
for r in scaling_results:
    time_per_color = r["mean"] / int(r["name"].split()[0].replace(",", "")) * 1e6
    print(
        f"{r['name']:<20s} {r['mean']:>8.3f} +/- {r['std']:>5.3f}   "
        f"{r['throughput']:>6.0f} M/s        {time_per_color:>8.3f} ns"
    )

# ============================================================================
# BENCHMARK 3: LUT Compilation Overhead
# ============================================================================

print("\n" + "=" * 80)
print("[BENCHMARK 3] LUT Compilation Overhead")
print("-" * 80)

N = 100_000
colors = np.random.rand(N, 3).astype(np.float32)
out = np.empty_like(colors)

# Test with fresh LUT (includes compilation)
lut_fresh = ColorLUT(device="cpu")
times_with_compile = []
for _ in range(20):
    lut_fresh = ColorLUT(device="cpu")  # Fresh LUT each time
    start = time.perf_counter()
    lut_fresh.apply_numpy_inplace(colors, out, temperature=0.5, brightness=1.2, saturation=1.3)
    times_with_compile.append((time.perf_counter() - start) * 1000)

# Test with cached LUT (no compilation)
lut_cached = ColorLUT(device="cpu")
lut_cached.apply_numpy_inplace(
    colors[:10], out[:10], temperature=0.5, brightness=1.2, saturation=1.3
)
times_cached = []
for _ in range(100):
    start = time.perf_counter()
    lut_cached.apply_numpy_inplace(colors, out, temperature=0.5, brightness=1.2, saturation=1.3)
    times_cached.append((time.perf_counter() - start) * 1000)

time_with_compile = np.mean(times_with_compile)
time_cached = np.mean(times_cached)
compile_overhead = time_with_compile - time_cached

print(
    f"\nWith LUT compilation:  {time_with_compile:>8.3f} ms +/- {np.std(times_with_compile):>5.3f} ms"
)
print(f"Cached LUT (no compile): {time_cached:>8.3f} ms +/- {np.std(times_cached):>5.3f} ms")
print(
    f"Compilation overhead:    {compile_overhead:>8.3f} ms ({compile_overhead / time_with_compile * 100:.1f}%)"
)

# ============================================================================
# BENCHMARK 4: Parameter Variation Impact
# ============================================================================

print("\n" + "=" * 80)
print("[BENCHMARK 4] Parameter Variation Impact")
print("-" * 80)

N = 100_000
colors = np.random.rand(N, 3).astype(np.float32)
out = np.empty_like(colors)

param_configs = [
    (
        "Identity (all defaults)",
        {
            "temperature": 0.5,
            "brightness": 1.0,
            "contrast": 1.0,
            "gamma": 1.0,
            "saturation": 1.0,
            "shadows": 1.0,
            "highlights": 1.0,
        },
    ),
    (
        "Temperature only",
        {
            "temperature": 0.8,
            "brightness": 1.0,
            "contrast": 1.0,
            "gamma": 1.0,
            "saturation": 1.0,
            "shadows": 1.0,
            "highlights": 1.0,
        },
    ),
    (
        "Phase 1 only",
        {
            "temperature": 0.7,
            "brightness": 1.2,
            "contrast": 1.1,
            "gamma": 0.9,
            "saturation": 1.0,
            "shadows": 1.0,
            "highlights": 1.0,
        },
    ),
    (
        "Phase 2 only",
        {
            "temperature": 0.5,
            "brightness": 1.0,
            "contrast": 1.0,
            "gamma": 1.0,
            "saturation": 1.3,
            "shadows": 1.1,
            "highlights": 0.9,
        },
    ),
    (
        "All parameters",
        {
            "temperature": 0.7,
            "brightness": 1.2,
            "contrast": 1.1,
            "gamma": 0.9,
            "saturation": 1.3,
            "shadows": 1.1,
            "highlights": 0.9,
        },
    ),
]

param_results = []
for name, params in param_configs:
    lut = ColorLUT(device="cpu")
    # Pre-compile for this config
    lut.apply_numpy_inplace(colors[:10], out[:10], **params)

    result = benchmark_api(
        name,
        lambda p=params: lut.apply_numpy_inplace(colors, out, **p),
        colors,
        warmup=10,
        iterations=100,
    )
    param_results.append(result)

print(f"\n{'Configuration':<25s} {'Time (ms)':<15s} {'Throughput':<15s}")
print("-" * 80)
for r in param_results:
    print(f"{r['name']:<25s} {r['mean']:>8.3f} +/- {r['std']:>5.3f}   {r['throughput']:>6.0f} M/s")

# ============================================================================
# BENCHMARK 5: Memory Allocation Overhead
# ============================================================================

print("\n" + "=" * 80)
print("[BENCHMARK 5] Memory Allocation Overhead")
print("-" * 80)

N = 100_000
colors = np.random.rand(N, 3).astype(np.float32)

lut = ColorLUT(device="cpu")
# Pre-compile
lut.apply_numpy(colors[:10], temperature=0.5, brightness=1.2)

# Measure pure computation (pre-allocated buffer)
out = np.empty_like(colors)
times_inplace = []
for _ in range(100):
    start = time.perf_counter()
    lut.apply_numpy_inplace(colors, out, temperature=0.5, brightness=1.2)
    times_inplace.append((time.perf_counter() - start) * 1000)

# Measure with allocation
times_with_alloc = []
for _ in range(100):
    start = time.perf_counter()
    result = lut.apply_numpy(colors, temperature=0.5, brightness=1.2)
    times_with_alloc.append((time.perf_counter() - start) * 1000)

time_inplace = np.mean(times_inplace)
time_with_alloc = np.mean(times_with_alloc)
alloc_overhead = time_with_alloc - time_inplace

print(f"\nPre-allocated buffer:  {time_inplace:>8.3f} ms +/- {np.std(times_inplace):>5.3f} ms")
print(f"With allocation:       {time_with_alloc:>8.3f} ms +/- {np.std(times_with_alloc):>5.3f} ms")
print(
    f"Allocation overhead:   {alloc_overhead:>8.3f} ms ({alloc_overhead / time_with_alloc * 100:.1f}%)"
)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)

best_throughput = max(r["throughput"] for r in results)
best_time = min(r["mean"] for r in results)

print("\nBest Performance:")
print("  Method: apply_numpy_inplace()")
print(f"  Time: {best_time:.3f} ms (100K colors)")
print(f"  Throughput: {best_throughput:.0f} M colors/sec")
print(f"  Latency: {best_time / N * 1e6:.3f} ns/color")
print()
print("Scaling:")
print(f"  1K colors:   {scaling_results[0]['mean']:.4f} ms")
print(f"  10K colors:  {scaling_results[1]['mean']:.4f} ms")
print(f"  100K colors: {scaling_results[2]['mean']:.3f} ms")
print(f"  1M colors:   {scaling_results[3]['mean']:.2f} ms")
print()
print("Overhead Analysis:")
print(
    f"  LUT compilation: {compile_overhead:.3f} ms ({compile_overhead / time_with_compile * 100:.1f}%)"
)
print(
    f"  Memory allocation: {alloc_overhead:.3f} ms ({alloc_overhead / time_with_alloc * 100:.1f}%)"
)
print()
print("Recommendations:")
print("  - Use apply_numpy_inplace() for maximum performance")
print("  - Reuse LUTs when processing multiple batches with same parameters")
print("  - Pre-allocate output buffers to eliminate allocation overhead")
print("=" * 80)
