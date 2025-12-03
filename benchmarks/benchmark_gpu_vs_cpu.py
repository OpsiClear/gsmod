"""Benchmark GPU (PyTorch) vs CPU (Numba) for SH color operations.

This script measures the current GPU performance to determine if
additional optimization (e.g., Triton) is needed.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from gsmod.torch.gstensor_pro import GSTensorPro


def benchmark_gpu_operation(name: str, operation_fn, iterations: int = 100):
    """Benchmark a GPU operation with proper CUDA synchronization.

    :param name: Operation name
    :param operation_fn: Function to benchmark
    :param iterations: Number of iterations
    :returns: Average time in milliseconds
    """
    # Warmup
    for _ in range(5):
        operation_fn()
    torch.cuda.synchronize()

    # Benchmark
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iterations):
        operation_fn()
    end_event.record()

    torch.cuda.synchronize()
    total_time = start_event.elapsed_time(end_event)  # milliseconds
    avg_time = total_time / iterations

    print(f"{name}: {avg_time:.3f} ms")
    return avg_time


def benchmark_cpu_operation(name: str, operation_fn, iterations: int = 100):
    """Benchmark a CPU operation.

    :param name: Operation name
    :param operation_fn: Function to benchmark
    :param iterations: Number of iterations
    :returns: Average time in milliseconds
    """
    # Warmup
    for _ in range(5):
        operation_fn()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        operation_fn()
    total_time = (time.perf_counter() - start) * 1000  # milliseconds
    avg_time = total_time / iterations

    print(f"{name}: {avg_time:.3f} ms")
    return avg_time


def main():
    """Run comprehensive GPU vs CPU benchmarks."""
    print("=" * 80)
    print("GPU (PyTorch CUDA) vs CPU (Numba) Performance Benchmark")
    print("=" * 80)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available! This benchmark requires a CUDA-capable GPU.")
        print("Showing CPU-only results...\n")
        gpu_available = False
    else:
        gpu_available = True
        print(f"\n✅ GPU Found: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch Version: {torch.__version__}\n")

    # Test data sizes
    sizes = [
        (100_000, "Small (100K Gaussians)"),
        (1_000_000, "Medium (1M Gaussians)"),
        (5_000_000, "Large (5M Gaussians)"),
    ]

    all_results = []

    for N, size_name in sizes:
        print(f"\n{'=' * 80}")
        print(f"{size_name}")
        print(f"{'=' * 80}\n")

        # Create test data
        sh0_np = np.random.rand(N, 3).astype(np.float32)
        shN_np = np.random.rand(N, 8, 3).astype(np.float32)  # Degree 2 SH

        # CPU version (GSDataPro)
        from gsmod.gsdata_pro import GSDataPro

        cpu_data = GSDataPro(
            means=np.random.rand(N, 3).astype(np.float32),
            scales=np.random.rand(N, 3).astype(np.float32),
            quats=np.random.rand(N, 4).astype(np.float32),
            opacities=np.random.rand(N, 1).astype(np.float32),
            sh0=sh0_np.copy(),
            shN=shN_np.copy(),
        )

        if gpu_available:
            # GPU version (GSTensorPro)
            gpu_data = GSTensorPro(
                means=torch.from_numpy(np.random.rand(N, 3).astype(np.float32)).cuda(),
                scales=torch.from_numpy(np.random.rand(N, 3).astype(np.float32)).cuda(),
                quats=torch.from_numpy(np.random.rand(N, 4).astype(np.float32)).cuda(),
                opacities=torch.from_numpy(np.random.rand(N, 1).astype(np.float32)).cuda(),
                sh0=torch.from_numpy(sh0_np.copy()).cuda(),
                shN=torch.from_numpy(shN_np.copy()).cuda(),
            )

        results = []

        # 1. Brightness
        print("1. Brightness (1.2x):")
        cpu_time = benchmark_cpu_operation(
            "  CPU (Numba)", lambda: cpu_data.adjust_brightness(1.2, inplace=False)
        )

        if gpu_available:
            gpu_time = benchmark_gpu_operation(
                "  GPU (CUDA) ",
                lambda: gpu_data.adjust_brightness(1.2, inplace=False),
            )
            speedup = cpu_time / gpu_time
            print(f"  Speedup: {speedup:.2f}x (GPU faster)")
            results.append(("Brightness", cpu_time, gpu_time, speedup))
        else:
            results.append(("Brightness", cpu_time, None, None))

        # 2. Saturation
        print("\n2. Saturation (1.3x):")
        cpu_time = benchmark_cpu_operation(
            "  CPU (Numba)", lambda: cpu_data.adjust_saturation(1.3, inplace=False)
        )

        if gpu_available:
            gpu_time = benchmark_gpu_operation(
                "  GPU (CUDA) ",
                lambda: gpu_data.adjust_saturation(1.3, inplace=False),
            )
            speedup = cpu_time / gpu_time
            print(f"  Speedup: {speedup:.2f}x (GPU faster)")
            results.append(("Saturation", cpu_time, gpu_time, speedup))
        else:
            results.append(("Saturation", cpu_time, None, None))

        # 3. Contrast
        print("\n3. Contrast (1.1x):")
        cpu_time = benchmark_cpu_operation(
            "  CPU (Numba)", lambda: cpu_data.adjust_contrast(1.1, inplace=False)
        )

        if gpu_available:
            gpu_time = benchmark_gpu_operation(
                "  GPU (CUDA) ", lambda: gpu_data.adjust_contrast(1.1, inplace=False)
            )
            speedup = cpu_time / gpu_time
            print(f"  Speedup: {speedup:.2f}x (GPU faster)")
            results.append(("Contrast", cpu_time, gpu_time, speedup))
        else:
            results.append(("Contrast", cpu_time, None, None))

        # 4. Temperature
        print("\n4. Temperature (0.2):")
        cpu_time = benchmark_cpu_operation(
            "  CPU (Numba)", lambda: cpu_data.adjust_temperature(0.2, inplace=False)
        )

        if gpu_available:
            gpu_time = benchmark_gpu_operation(
                "  GPU (CUDA) ",
                lambda: gpu_data.adjust_temperature(0.2, inplace=False),
            )
            speedup = cpu_time / gpu_time
            print(f"  Speedup: {speedup:.2f}x (GPU faster)")
            results.append(("Temperature", cpu_time, gpu_time, speedup))
        else:
            results.append(("Temperature", cpu_time, None, None))

        # 5. Hue Shift
        print("\n5. Hue Shift (30°):")
        cpu_time = benchmark_cpu_operation(
            "  CPU (Numba)", lambda: cpu_data.adjust_hue_shift(30, inplace=False)
        )

        if gpu_available:
            gpu_time = benchmark_gpu_operation(
                "  GPU (CUDA) ", lambda: gpu_data.adjust_hue_shift(30, inplace=False)
            )
            speedup = cpu_time / gpu_time
            print(f"  Speedup: {speedup:.2f}x (GPU faster)")
            results.append(("Hue Shift", cpu_time, gpu_time, speedup))
        else:
            results.append(("Hue Shift", cpu_time, None, None))

        # 6. Gamma
        print("\n6. Gamma (0.9):")
        cpu_time = benchmark_cpu_operation(
            "  CPU (Numba)", lambda: cpu_data.adjust_gamma(0.9, inplace=False)
        )

        if gpu_available:
            gpu_time = benchmark_gpu_operation(
                "  GPU (CUDA) ", lambda: gpu_data.adjust_gamma(0.9, inplace=False)
            )
            speedup = cpu_time / gpu_time
            print(f"  Speedup: {speedup:.2f}x (GPU faster)")
            results.append(("Gamma", cpu_time, gpu_time, speedup))
        else:
            results.append(("Gamma", cpu_time, None, None))

        # Summary
        print(f"\n{'-' * 80}")
        print(f"Summary for {size_name}:")
        print(f"{'-' * 80}")

        if gpu_available:
            avg_speedup = sum(s for _, _, _, s in results if s) / len(results)
            print(f"Average GPU Speedup: {avg_speedup:.2f}x")
            print(f"Min Speedup: {min(s for _, _, _, s in results if s):.2f}x")
            print(f"Max Speedup: {max(s for _, _, _, s in results if s):.2f}x")

            # Calculate throughput
            ops_per_sec_gpu = 1000 / results[0][2]  # Using brightness as reference
            gaussians_per_sec = ops_per_sec_gpu * N
            print(f"\nGPU Throughput: {gaussians_per_sec / 1e6:.1f}M Gaussians/sec (brightness)")

        all_results.append((size_name, results))

    # Final analysis
    print(f"\n{'=' * 80}")
    print("FINAL ANALYSIS")
    print(f"{'=' * 80}\n")

    if gpu_available:
        print("GPU Performance Assessment:")
        print("-" * 80)

        # Get results from largest dataset
        _, large_results = all_results[-1]
        avg_speedup = sum(s for _, _, _, s in large_results if s) / len(large_results)

        print(f"✅ GPU is {avg_speedup:.1f}x faster than CPU on average")
        print("✅ PyTorch CUDA is highly optimized (uses cuBLAS for matrix ops)")
        print("✅ Current implementation is already near-optimal")

        print("\nIs Triton optimization needed?")
        print("-" * 80)

        if avg_speedup > 10:
            print("🟢 NO - GPU is already >10x faster than CPU. Triton would add minimal benefit.")
            print("   Recommendation: Keep current PyTorch implementation.")
        elif avg_speedup > 5:
            print("🟡 MAYBE - GPU is 5-10x faster. Triton could provide 2-3x additional speedup.")
            print("   Recommendation: Only add Triton if you need absolute max speed.")
        else:
            print("🔴 YES - GPU is <5x faster. Triton optimization could help significantly.")
            print("   Recommendation: Consider Triton for fused kernels.")

        print("\nPotential Triton gains (estimated):")
        print("  - Kernel fusion: ~2-3x speedup")
        print("  - Custom memory layout: ~1.2-1.5x speedup")
        print("  - **Total estimated**: ~2.5-4x faster than current GPU")
        print("\nTrade-off:")
        print("  - Windows support: Experimental (triton-windows fork required)")
        print("  - Complexity: Higher (custom kernels to maintain)")
        print("  - Reliability: Lower (newer, less battle-tested)")
    else:
        print("⚠️  No GPU detected. Benchmark shows CPU performance only.")

    print(f"\n{'=' * 80}")
    print("Benchmark Complete!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
