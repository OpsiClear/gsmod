"""Benchmark GPU operations vs CPU operations in gsmod."""

import time

import numpy as np
import torch
from gsply import GSData

from gsmod import Color, Filter, Pipeline, Transform
from gsmod.torch import ColorGPU, FilterGPU, GSTensorPro, PipelineGPU, TransformGPU


def create_test_data(n_gaussians):
    """Create test Gaussian Splatting data."""
    np.random.seed(42)

    data = GSData(
        means=np.random.randn(n_gaussians, 3).astype(np.float32),
        scales=np.random.randn(n_gaussians, 3).astype(np.float32) * 0.1,
        quats=np.random.randn(n_gaussians, 4).astype(np.float32),
        opacities=np.random.rand(n_gaussians).astype(np.float32),
        sh0=np.random.rand(n_gaussians, 3).astype(np.float32),
        shN=None,  # SH0 only for simplicity
    )

    # Normalize quaternions
    norms = np.linalg.norm(data.quats, axis=1, keepdims=True)
    data.quats /= norms

    return data


def benchmark_color_operations(data, device="cuda"):
    """Benchmark color operations on GPU vs CPU."""
    print(f"\n[COLOR OPERATIONS BENCHMARK - {len(data):,} Gaussians]")
    print("=" * 60)

    # Prepare data
    gstensor_gpu = GSTensorPro.from_gsdata(data, device=device)
    data_cpu = data.copy()

    # GPU color pipeline
    gpu_pipeline = (
        ColorGPU().brightness(1.2).contrast(1.1).saturation(1.3).gamma(0.9).temperature(0.1)
    )

    # Warmup
    _ = gpu_pipeline(gstensor_gpu.clone(), inplace=True)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark GPU
    start = time.perf_counter()
    gpu_pipeline(gstensor_gpu.clone(), inplace=True)
    if device == "cuda":
        torch.cuda.synchronize()
    gpu_time = time.perf_counter() - start

    # CPU color pipeline
    cpu_pipeline = Color().brightness(1.2).contrast(1.1).saturation(1.3).gamma(0.9).temperature(0.1)

    # Warmup
    _ = cpu_pipeline(data_cpu.copy(), inplace=True)

    # Benchmark CPU
    start = time.perf_counter()
    cpu_pipeline(data_cpu.copy(), inplace=True)
    cpu_time = time.perf_counter() - start

    # Results
    print(f"GPU ({device}): {gpu_time*1000:.2f} ms")
    print(f"CPU: {cpu_time*1000:.2f} ms")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x")

    return gpu_time, cpu_time


def benchmark_transform_operations(data, device="cuda"):
    """Benchmark transform operations on GPU vs CPU."""
    print(f"\n[TRANSFORM OPERATIONS BENCHMARK - {len(data):,} Gaussians]")
    print("=" * 60)

    # Prepare data
    gstensor_gpu = GSTensorPro.from_gsdata(data, device=device)
    data_cpu = data.copy()

    # GPU transform pipeline
    gpu_pipeline = (
        TransformGPU().translate([1.0, 0.0, 0.5]).rotate_axis_angle([0, 1, 0], np.pi / 4).scale(2.0)
    )

    # Warmup
    _ = gpu_pipeline(gstensor_gpu.clone(), inplace=True)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark GPU
    start = time.perf_counter()
    gpu_pipeline(gstensor_gpu.clone(), inplace=True)
    if device == "cuda":
        torch.cuda.synchronize()
    gpu_time = time.perf_counter() - start

    # CPU transform pipeline
    cpu_pipeline = (
        Transform().translate([1.0, 0.0, 0.5]).rotate_axis_angle([0, 1, 0], np.pi / 4).scale(2.0)
    )

    # Warmup
    _ = cpu_pipeline(data_cpu.copy(), inplace=True)

    # Benchmark CPU
    start = time.perf_counter()
    cpu_pipeline(data_cpu.copy(), inplace=True)
    cpu_time = time.perf_counter() - start

    # Results
    print(f"GPU ({device}): {gpu_time*1000:.2f} ms")
    print(f"CPU: {cpu_time*1000:.2f} ms")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x")

    return gpu_time, cpu_time


def benchmark_filter_operations(data, device="cuda"):
    """Benchmark filter operations on GPU vs CPU."""
    print(f"\n[FILTER OPERATIONS BENCHMARK - {len(data):,} Gaussians]")
    print("=" * 60)

    # Prepare data
    gstensor_gpu = GSTensorPro.from_gsdata(data, device=device)
    data_cpu = data.copy()

    # GPU filter pipeline
    gpu_pipeline = FilterGPU().within_sphere(radius=2.0).min_opacity(0.1).max_scale(0.5)

    # Warmup
    _ = gpu_pipeline.compute_mask(gstensor_gpu)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark GPU
    start = time.perf_counter()
    mask_gpu = gpu_pipeline.compute_mask(gstensor_gpu)
    filtered_gpu = gstensor_gpu[mask_gpu]
    if device == "cuda":
        torch.cuda.synchronize()
    gpu_time = time.perf_counter() - start

    # CPU filter pipeline
    cpu_pipeline = Filter().within_sphere(radius=2.0).min_opacity(0.1).max_scale(0.5)

    # Warmup
    _ = cpu_pipeline(data_cpu.copy(), inplace=False)

    # Benchmark CPU
    start = time.perf_counter()
    filtered_cpu = cpu_pipeline(data_cpu.copy(), inplace=False)
    cpu_time = time.perf_counter() - start

    # Results
    print(f"GPU ({device}): {gpu_time*1000:.2f} ms")
    print(f"CPU: {cpu_time*1000:.2f} ms")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x")
    print(f"Filtered: GPU={len(filtered_gpu):,}, CPU={len(filtered_cpu):,}")

    return gpu_time, cpu_time


def benchmark_unified_pipeline(data, device="cuda"):
    """Benchmark unified pipeline on GPU vs CPU."""
    print(f"\n[UNIFIED PIPELINE BENCHMARK - {len(data):,} Gaussians]")
    print("=" * 60)

    # Prepare data
    gstensor_gpu = GSTensorPro.from_gsdata(data, device=device)
    data_cpu = data.copy()

    # GPU unified pipeline
    gpu_pipeline = (
        PipelineGPU()
        .within_sphere(radius=3.0)
        .min_opacity(0.05)
        .translate([1.0, 0.0, 0.0])
        .rotate_axis_angle([0, 1, 0], np.pi / 6)
        .brightness(1.15)
        .saturation(1.2)
        .contrast(1.1)
    )

    # Warmup
    _ = gpu_pipeline(gstensor_gpu.clone(), inplace=False)
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark GPU
    start = time.perf_counter()
    result_gpu = gpu_pipeline(gstensor_gpu.clone(), inplace=False)
    if device == "cuda":
        torch.cuda.synchronize()
    gpu_time = time.perf_counter() - start

    # CPU unified pipeline
    cpu_pipeline = (
        Pipeline()
        .within_sphere(radius=3.0)
        .min_opacity(0.05)
        .translate([1.0, 0.0, 0.0])
        .rotate_axis_angle([0, 1, 0], np.pi / 6)
        .brightness(1.15)
        .saturation(1.2)
        .contrast(1.1)
    )

    # Warmup
    _ = cpu_pipeline(data_cpu.copy(), inplace=False)

    # Benchmark CPU
    start = time.perf_counter()
    result_cpu = cpu_pipeline(data_cpu.copy(), inplace=False)
    cpu_time = time.perf_counter() - start

    # Results
    print(f"GPU ({device}): {gpu_time*1000:.2f} ms")
    print(f"CPU: {cpu_time*1000:.2f} ms")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x")
    print(f"Output: GPU={len(result_gpu):,}, CPU={len(result_cpu):,}")

    return gpu_time, cpu_time


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 60)
    print("GSMOD GPU PERFORMANCE BENCHMARKS")
    print("=" * 60)

    # Check GPU availability
    if torch.cuda.is_available():
        device = "cuda"
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = "cpu"
        print("\nWARNING: CUDA not available. Running on CPU.")

    print(f"PyTorch Version: {torch.__version__}")

    # Test different sizes
    sizes = [10_000, 100_000, 500_000]

    if torch.cuda.is_available():
        sizes.append(1_000_000)

    results = {}

    for n_gaussians in sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {n_gaussians:,} Gaussians")
        print(f"{'='*60}")

        data = create_test_data(n_gaussians)

        # Run benchmarks
        color_gpu, color_cpu = benchmark_color_operations(data, device)
        transform_gpu, transform_cpu = benchmark_transform_operations(data, device)
        filter_gpu, filter_cpu = benchmark_filter_operations(data, device)
        unified_gpu, unified_cpu = benchmark_unified_pipeline(data, device)

        results[n_gaussians] = {
            "color": (color_gpu, color_cpu),
            "transform": (transform_gpu, transform_cpu),
            "filter": (filter_gpu, filter_cpu),
            "unified": (unified_gpu, unified_cpu),
        }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for n_gaussians in sizes:
        print(f"\n{n_gaussians:,} Gaussians:")
        for op_type, (gpu_time, cpu_time) in results[n_gaussians].items():
            speedup = cpu_time / gpu_time
            print(
                f"  {op_type.capitalize():12} - Speedup: {speedup:.1f}x "
                f"(GPU: {gpu_time*1000:.1f}ms, CPU: {cpu_time*1000:.1f}ms)"
            )

    # Average speedups
    print("\n" + "-" * 60)
    print("Average Speedups:")
    for op_type in ["color", "transform", "filter", "unified"]:
        speedups = [results[n][op_type][1] / results[n][op_type][0] for n in sizes]
        avg_speedup = sum(speedups) / len(speedups)
        print(f"  {op_type.capitalize():12} - {avg_speedup:.1f}x")


if __name__ == "__main__":
    main()
