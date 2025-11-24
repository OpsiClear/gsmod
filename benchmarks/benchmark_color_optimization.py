"""Benchmark color processing performance with optimization logic.

Tests early returns for neutral values and skip conditions across:
- CPU Color pipeline
- GPU GSTensorPro
- LearnableColor (differentiable)
"""

import time

import numpy as np
import torch
from gsply import GSData

from gsmod import ColorValues, GSDataPro
from gsmod.torch import GSTensorPro
from gsmod.torch.learn import ColorGradingConfig, LearnableColor


def create_test_data(n: int) -> GSData:
    """Create synthetic GSData for benchmarking."""
    rng = np.random.default_rng(42)
    means = rng.random((n, 3), dtype=np.float32)
    scales = rng.random((n, 3), dtype=np.float32) * 0.1
    quats = rng.random((n, 4), dtype=np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = rng.random(n, dtype=np.float32)
    sh0 = rng.random((n, 3), dtype=np.float32)

    return GSData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=None,
    )


def benchmark_cpu_color(n_gaussians: int, n_iterations: int = 100):
    """Benchmark CPU Color pipeline."""
    print(f"\n{'='*60}")
    print(f"CPU Color Pipeline Benchmark ({n_gaussians:,} Gaussians)")
    print(f"{'='*60}")

    gsdata = create_test_data(n_gaussians)
    data = GSDataPro.from_gsdata(gsdata)

    # Test 1: All neutral values (should skip all operations)
    neutral_values = ColorValues()

    # Warmup
    for _ in range(5):
        data.color(neutral_values, inplace=True)

    start = time.perf_counter()
    for _ in range(n_iterations):
        data.color(neutral_values, inplace=True)
    elapsed = time.perf_counter() - start

    neutral_time = elapsed / n_iterations * 1000
    neutral_throughput = n_gaussians / (elapsed / n_iterations) / 1e6
    print("\nNeutral values (all skipped):")
    print(f"  Time: {neutral_time:.3f} ms")
    print(f"  Throughput: {neutral_throughput:.1f} M/s")

    # Test 2: Single operation (brightness only)
    single_values = ColorValues(brightness=1.2)

    start = time.perf_counter()
    for _ in range(n_iterations):
        data.color(single_values, inplace=True)
    elapsed = time.perf_counter() - start

    single_time = elapsed / n_iterations * 1000
    single_throughput = n_gaussians / (elapsed / n_iterations) / 1e6
    print("\nSingle operation (brightness=1.2):")
    print(f"  Time: {single_time:.3f} ms")
    print(f"  Throughput: {single_throughput:.1f} M/s")

    # Test 3: Full pipeline (all operations active)
    full_values = ColorValues(
        brightness=1.2,
        contrast=1.1,
        gamma=1.05,
        saturation=1.3,
        vibrance=1.1,
        temperature=0.2,
        tint=0.1,
        shadows=0.1,
        highlights=-0.05,
        fade=0.05,
        hue_shift=15.0,
        shadow_tint_hue=-140,
        shadow_tint_sat=0.2,
        highlight_tint_hue=40,
        highlight_tint_sat=0.15,
    )

    start = time.perf_counter()
    for _ in range(n_iterations):
        data.color(full_values, inplace=True)
    elapsed = time.perf_counter() - start

    full_time = elapsed / n_iterations * 1000
    full_throughput = n_gaussians / (elapsed / n_iterations) / 1e6
    print("\nFull pipeline (15 operations):")
    print(f"  Time: {full_time:.3f} ms")
    print(f"  Throughput: {full_throughput:.1f} M/s")

    # Summary
    print(f"\nSpeedup from skipping neutral ops: {full_time/neutral_time:.1f}x")

    return {
        "neutral": neutral_time,
        "single": single_time,
        "full": full_time,
    }


def benchmark_gpu_tensor(n_gaussians: int, n_iterations: int = 100):
    """Benchmark GPU GSTensorPro color operations."""
    if not torch.cuda.is_available():
        print("\n[SKIP] GPU benchmarks - CUDA not available")
        return None

    print(f"\n{'='*60}")
    print(f"GPU GSTensorPro Benchmark ({n_gaussians:,} Gaussians)")
    print(f"{'='*60}")

    gsdata = create_test_data(n_gaussians)
    data = GSTensorPro.from_gsdata(gsdata, device="cuda")

    # Warmup
    torch.cuda.synchronize()
    for _ in range(10):
        data.adjust_brightness(1.0, inplace=True)
    torch.cuda.synchronize()

    # Test 1: Neutral brightness (should early return)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iterations):
        data.adjust_brightness(1.0, inplace=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    neutral_time = elapsed / n_iterations * 1000
    print("\nNeutral brightness (early return):")
    print(f"  Time: {neutral_time:.4f} ms")

    # Test 2: Active brightness
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iterations):
        data.adjust_brightness(1.2, inplace=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    active_time = elapsed / n_iterations * 1000
    throughput = n_gaussians / (elapsed / n_iterations) / 1e6
    print("\nActive brightness (factor=1.2):")
    print(f"  Time: {active_time:.4f} ms")
    print(f"  Throughput: {throughput:.1f} M/s")

    # Test 3: Multiple operations with ColorValues
    neutral_values = ColorValues()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iterations):
        data.color(neutral_values, inplace=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    neutral_color_time = elapsed / n_iterations * 1000
    print("\nColorValues neutral (all skipped):")
    print(f"  Time: {neutral_color_time:.4f} ms")

    # Test 4: Full ColorValues
    full_values = ColorValues(
        brightness=1.2,
        contrast=1.1,
        gamma=1.05,
        saturation=1.3,
        temperature=0.2,
        shadows=0.1,
        highlights=-0.05,
    )

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iterations):
        data.color(full_values, inplace=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    full_time = elapsed / n_iterations * 1000
    full_throughput = n_gaussians / (elapsed / n_iterations) / 1e6
    print("\nFull ColorValues (7 operations):")
    print(f"  Time: {full_time:.4f} ms")
    print(f"  Throughput: {full_throughput:.1f} M/s")

    # Summary
    print(f"\nEarly return speedup: {active_time/neutral_time:.1f}x")
    print(f"Skip optimization speedup: {full_time/neutral_color_time:.1f}x")

    return {
        "neutral": neutral_time,
        "active": active_time,
        "neutral_color": neutral_color_time,
        "full": full_time,
    }


def benchmark_learnable(n_gaussians: int, n_iterations: int = 100):
    """Benchmark LearnableColor with skip optimizations."""
    print(f"\n{'='*60}")
    print(f"LearnableColor Benchmark ({n_gaussians:,} Gaussians)")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Create test data
    sh0 = torch.rand(n_gaussians, 3, device=device)

    # Test 1: All parameters learnable at neutral values
    config = ColorGradingConfig()  # All learnable by default
    model = LearnableColor(config).to(device)

    # Warmup
    if device == "cuda":
        torch.cuda.synchronize()
    for _ in range(10):
        _ = model(sh0)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark all learnable (no skipping - gradients needed)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = model(sh0)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    all_learnable_time = elapsed / n_iterations * 1000
    throughput = n_gaussians / (elapsed / n_iterations) / 1e6
    print("\nAll parameters learnable (15 ops, no skipping):")
    print(f"  Time: {all_learnable_time:.3f} ms")
    print(f"  Throughput: {throughput:.1f} M/s")

    # Test 2: Only brightness learnable (others can skip)
    config2 = ColorGradingConfig(learnable=["brightness"])
    model2 = LearnableColor(config2).to(device)

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = model2(sh0)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    one_learnable_time = elapsed / n_iterations * 1000
    throughput = n_gaussians / (elapsed / n_iterations) / 1e6
    print("\nOnly brightness learnable (14 ops skipped):")
    print(f"  Time: {one_learnable_time:.3f} ms")
    print(f"  Throughput: {throughput:.1f} M/s")

    # Test 3: No parameters learnable (all at neutral = all skip)
    config3 = ColorGradingConfig(learnable=[])
    model3 = LearnableColor(config3).to(device)

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iterations):
        _ = model3(sh0)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    none_learnable_time = elapsed / n_iterations * 1000
    throughput = n_gaussians / (elapsed / n_iterations) / 1e6
    print("\nNo parameters learnable (all 15 ops skipped):")
    print(f"  Time: {none_learnable_time:.3f} ms")
    print(f"  Throughput: {throughput:.1f} M/s")

    # Test 4: Gradient computation overhead
    config4 = ColorGradingConfig(learnable=["brightness", "contrast", "saturation"])
    model4 = LearnableColor(config4).to(device)

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iterations):
        out = model4(sh0)
        loss = out.mean()
        loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    grad_time = elapsed / n_iterations * 1000
    print("\nWith gradient computation (3 learnable):")
    print(f"  Time: {grad_time:.3f} ms")

    # Summary
    print(f"\nSkip optimization speedup: {all_learnable_time/none_learnable_time:.1f}x")
    print(f"Partial learnable speedup: {all_learnable_time/one_learnable_time:.1f}x")

    return {
        "all_learnable": all_learnable_time,
        "one_learnable": one_learnable_time,
        "none_learnable": none_learnable_time,
        "with_grad": grad_time,
    }


def main():
    """Run all benchmarks."""
    print("Color Processing Optimization Benchmarks")
    print("=" * 60)
    print("Testing early returns and skip conditions for neutral values")

    # Test sizes
    sizes = [100_000, 1_000_000]

    results = {}

    for n in sizes:
        print(f"\n\n{'#'*60}")
        print(f"# Dataset Size: {n:,} Gaussians")
        print(f"{'#'*60}")

        results[n] = {
            "cpu": benchmark_cpu_color(n),
            "gpu": benchmark_gpu_tensor(n),
            "learnable": benchmark_learnable(n),
        }

    # Print summary table
    print(f"\n\n{'='*60}")
    print("SUMMARY: Optimization Impact")
    print(f"{'='*60}")

    for n in sizes:
        print(f"\n{n:,} Gaussians:")

        cpu = results[n]["cpu"]
        if cpu:
            speedup = cpu["full"] / cpu["neutral"]
            print(f"  CPU Pipeline: {speedup:.1f}x faster with neutral skip")

        gpu = results[n]["gpu"]
        if gpu:
            speedup = gpu["full"] / gpu["neutral_color"]
            print(f"  GPU Pipeline: {speedup:.1f}x faster with neutral skip")

        learn = results[n]["learnable"]
        if learn:
            speedup = learn["all_learnable"] / learn["none_learnable"]
            print(f"  LearnableColor: {speedup:.1f}x faster with all skipped")


if __name__ == "__main__":
    main()
