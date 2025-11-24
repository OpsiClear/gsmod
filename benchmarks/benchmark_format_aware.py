"""Benchmark format-aware GPU operations and check equivalence with CPU."""

import logging
import time

import numpy as np
import torch
from gsply import GSData

from gsmod import Color, Filter, Transform
from gsmod.torch import ColorGPU, FilterGPU, GSTensorPro, TransformGPU

logging.basicConfig(level=logging.WARNING)


def create_test_data(n: int = 100000) -> GSData:
    """Create test data with n Gaussians."""
    np.random.seed(42)
    data = GSData(
        means=np.random.randn(n, 3).astype(np.float32) * 5,
        scales=np.random.rand(n, 3).astype(np.float32) * 0.5 + 0.1,
        quats=np.random.randn(n, 4).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32) * 0.8 + 0.1,
        sh0=np.random.rand(n, 3).astype(np.float32),
        shN=None,
    )
    # Normalize quaternions
    norms = np.linalg.norm(data.quats, axis=1, keepdims=True)
    data.quats /= norms
    return data


def benchmark_operation(
    name: str, cpu_func, gpu_func, data: GSData, device: str, warmup: int = 3, iterations: int = 10
):
    """Benchmark a single operation and check equivalence."""
    # Create GPU tensor
    gstensor = GSTensorPro.from_gsdata(data.copy(), device=device)

    # Warmup GPU
    for _ in range(warmup):
        gstensor_copy = gstensor.clone()
        gpu_func(gstensor_copy)
    torch.cuda.synchronize()

    # Benchmark GPU
    gpu_times = []
    for _ in range(iterations):
        gstensor_copy = gstensor.clone()
        torch.cuda.synchronize()
        start = time.perf_counter()
        gpu_result = gpu_func(gstensor_copy)
        torch.cuda.synchronize()
        gpu_times.append(time.perf_counter() - start)

    # Benchmark CPU
    cpu_times = []
    for _ in range(iterations):
        data_copy = data.copy()
        start = time.perf_counter()
        cpu_result = cpu_func(data_copy)
        cpu_times.append(time.perf_counter() - start)

    # Get results for equivalence check
    gpu_data = gpu_result.to_gsdata()

    # Calculate metrics
    gpu_mean = np.mean(gpu_times) * 1000
    cpu_mean = np.mean(cpu_times) * 1000
    speedup = cpu_mean / gpu_mean if gpu_mean > 0 else 0

    # Check equivalence
    try:
        # For color operations, compare sh0
        if hasattr(cpu_result, "sh0"):
            sh0_diff = np.abs(cpu_result.sh0 - gpu_data.sh0)
            max_diff = sh0_diff.max()
            mean_diff = sh0_diff.mean()
        # For transform operations, compare means and quaternions
        elif hasattr(cpu_result, "means"):
            means_diff = np.abs(cpu_result.means - gpu_data.means)
            quats_diff = np.abs(cpu_result.quats - gpu_data.quats)
            max_diff = max(means_diff.max(), quats_diff.max())
            mean_diff = (means_diff.mean() + quats_diff.mean()) / 2
        else:
            max_diff = 0
            mean_diff = 0
    except Exception:
        max_diff = -1
        mean_diff = -1

    return {
        "name": name,
        "cpu_ms": cpu_mean,
        "gpu_ms": gpu_mean,
        "speedup": speedup,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "n": len(data),
    }


def main():
    print("=" * 70)
    print("FORMAT-AWARE GPU BENCHMARK AND EQUIVALENCE CHECK")
    print("=" * 70)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print()

    # Test different data sizes
    sizes = [10000, 100000, 1000000]

    for n in sizes:
        print(f"\n{'=' * 70}")
        print(f"BENCHMARKING WITH {n:,} GAUSSIANS")
        print("=" * 70)

        data = create_test_data(n)
        results = []

        # ===== COLOR OPERATIONS =====
        print("\n[COLOR OPERATIONS]")

        # Brightness (format-agnostic)
        results.append(
            benchmark_operation(
                "brightness(1.2)",
                lambda d: Color().brightness(1.2)(d, inplace=True),
                lambda g: ColorGPU().brightness(1.2)(g, inplace=True),
                data,
                device,
            )
        )

        # Contrast (format-agnostic)
        results.append(
            benchmark_operation(
                "contrast(1.1)",
                lambda d: Color().contrast(1.1)(d, inplace=True),
                lambda g: ColorGPU().contrast(1.1)(g, inplace=True),
                data,
                device,
            )
        )

        # Gamma (format-agnostic)
        results.append(
            benchmark_operation(
                "gamma(0.8)",
                lambda d: Color().gamma(0.8)(d, inplace=True),
                lambda g: ColorGPU().gamma(0.8)(g, inplace=True),
                data,
                device,
            )
        )

        # Saturation (requires RGB)
        results.append(
            benchmark_operation(
                "saturation(1.3)",
                lambda d: Color().saturation(1.3)(d, inplace=True),
                lambda g: ColorGPU().saturation(1.3)(g, inplace=True),
                data,
                device,
            )
        )

        # Temperature (requires RGB)
        results.append(
            benchmark_operation(
                "temperature(0.2)",
                lambda d: Color().temperature(0.2)(d, inplace=True),
                lambda g: ColorGPU().temperature(0.2)(g, inplace=True),
                data,
                device,
            )
        )

        # Combined color pipeline
        results.append(
            benchmark_operation(
                "color_pipeline",
                lambda d: Color().brightness(1.1).contrast(1.05).saturation(1.2)(d, inplace=True),
                lambda g: ColorGPU()
                .brightness(1.1)
                .contrast(1.05)
                .saturation(1.2)(g, inplace=True),
                data,
                device,
            )
        )

        # ===== TRANSFORM OPERATIONS =====
        print("\n[TRANSFORM OPERATIONS]")

        # Translation
        results.append(
            benchmark_operation(
                "translate([1,0,0])",
                lambda d: Transform().translate([1.0, 0.0, 0.0])(d, inplace=True),
                lambda g: TransformGPU().translate([1.0, 0.0, 0.0])(g, inplace=True),
                data,
                device,
            )
        )

        # Uniform scale
        results.append(
            benchmark_operation(
                "scale(2.0)",
                lambda d: Transform().scale(2.0)(d, inplace=True),
                lambda g: TransformGPU().scale(2.0)(g, inplace=True),
                data,
                device,
            )
        )

        # Rotation
        results.append(
            benchmark_operation(
                "rotate_axis_angle",
                lambda d: Transform().rotate_axis_angle([0, 1, 0], 0.5)(d, inplace=True),
                lambda g: TransformGPU().rotate_axis_angle([0, 1, 0], 0.5)(g, inplace=True),
                data,
                device,
            )
        )

        # Combined transform
        results.append(
            benchmark_operation(
                "transform_pipeline",
                lambda d: Transform()
                .translate([1, 0, 0])
                .scale(1.5)
                .rotate_axis_angle([0, 1, 0], 0.3)(d, inplace=True),
                lambda g: TransformGPU()
                .translate([1, 0, 0])
                .scale(1.5)
                .rotate_axis_angle([0, 1, 0], 0.3)(g, inplace=True),
                data,
                device,
            )
        )

        # ===== FILTER OPERATIONS =====
        print("\n[FILTER OPERATIONS]")

        # Sphere filter
        results.append(
            benchmark_operation(
                "within_sphere(0.5)",
                lambda d: Filter().within_sphere(radius=0.5)(d, inplace=False),
                lambda g: FilterGPU().within_sphere(radius=0.5)(g, inplace=False),
                data,
                device,
            )
        )

        # Opacity filter
        results.append(
            benchmark_operation(
                "min_opacity(0.3)",
                lambda d: Filter().min_opacity(0.3)(d, inplace=False),
                lambda g: FilterGPU().min_opacity(0.3)(g, inplace=False),
                data,
                device,
            )
        )

        # Print results table
        print("\n" + "-" * 70)
        print(
            f"{'Operation':<25} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10} {'Max Diff':<12}"
        )
        print("-" * 70)

        for r in results:
            max_diff_str = f"{r['max_diff']:.2e}" if r["max_diff"] >= 0 else "ERROR"
            print(
                f"{r['name']:<25} {r['cpu_ms']:<12.3f} {r['gpu_ms']:<12.3f} {r['speedup']:<10.1f}x {max_diff_str:<12}"
            )

        # Summary statistics
        valid_results = [r for r in results if r["speedup"] > 0]
        if valid_results:
            avg_speedup = np.mean([r["speedup"] for r in valid_results])
            max_speedup = max(r["speedup"] for r in valid_results)

            # Calculate throughput
            total_gpu_time = sum(r["gpu_ms"] for r in valid_results)
            throughput = (n * len(valid_results)) / (total_gpu_time / 1000)

            print("-" * 70)
            print(f"Average speedup: {avg_speedup:.1f}x")
            print(f"Max speedup: {max_speedup:.1f}x")
            print(f"GPU throughput: {throughput/1e6:.1f}M Gaussians/sec")

        # Equivalence summary
        print("\n[EQUIVALENCE CHECK]")
        equiv_pass = sum(1 for r in results if 0 <= r["max_diff"] < 1e-3)
        equiv_close = sum(1 for r in results if 1e-3 <= r["max_diff"] < 0.1)
        equiv_diff = sum(1 for r in results if r["max_diff"] >= 0.1)
        equiv_error = sum(1 for r in results if r["max_diff"] < 0)

        print(f"  Equivalent (diff < 1e-3): {equiv_pass}/{len(results)}")
        print(f"  Close (diff < 0.1): {equiv_close}/{len(results)}")
        print(f"  Different (diff >= 0.1): {equiv_diff}/{len(results)}")
        if equiv_error > 0:
            print(f"  Errors: {equiv_error}/{len(results)}")

    # Final summary
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
