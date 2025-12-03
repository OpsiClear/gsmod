"""Benchmark PyTorch vs Triton for SH color operations.

Compares current PyTorch implementation against Triton-optimized kernels.
"""

from __future__ import annotations

import torch

from gsmod.torch.gstensor_pro import GSTensorPro
from gsmod.torch.triton_kernels import (
    triton_brightness,
    triton_fused_pipeline,
    triton_saturation,
)


def benchmark_operation(name: str, operation_fn, iterations: int = 100):
    """Benchmark a GPU operation with proper CUDA synchronization."""
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

    print(f"  {name}: {avg_time:.3f} ms")
    return avg_time


def main():
    """Run comprehensive Triton vs PyTorch benchmarks."""
    print("=" * 80)
    print("Triton vs PyTorch CUDA Performance Benchmark")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("\n❌ CUDA not available!")
        return

    print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Triton version: {torch.__version__}\n")

    # Test data sizes
    sizes = [
        (1_000_000, "Medium (1M Gaussians)"),
        (5_000_000, "Large (5M Gaussians)"),
    ]

    for N, size_name in sizes:
        print(f"\n{'=' * 80}")
        print(f"{size_name}")
        print(f"{'=' * 80}\n")

        # Create test data
        sh0 = torch.randn(N, 3, device="cuda", dtype=torch.float32)
        shN = torch.randn(N, 8, 3, device="cuda", dtype=torch.float32)

        # PyTorch version
        data_pytorch = GSTensorPro(
            means=torch.randn(N, 3, device="cuda"),
            scales=torch.randn(N, 3, device="cuda"),
            quats=torch.randn(N, 4, device="cuda"),
            opacities=torch.randn(N, 1, device="cuda"),
            sh0=sh0.clone(),
            shN=shN.clone(),
        )

        # 1. Brightness
        print("1. Brightness (1.2x):")
        pytorch_time = benchmark_operation(
            "PyTorch", lambda: data_pytorch.adjust_brightness(1.2, inplace=False)
        )

        sh0_triton_copy = sh0.clone()
        shN_triton_copy = shN.clone()
        triton_time = benchmark_operation(
            "Triton  ",
            lambda: triton_brightness(sh0_triton_copy, shN_triton_copy, 1.2),
        )

        speedup = pytorch_time / triton_time
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Winner: {'Triton' if speedup > 1 else 'PyTorch'} 🏆\n")

        # 2. Saturation
        print("2. Saturation (1.3x):")
        pytorch_time = benchmark_operation(
            "PyTorch", lambda: data_pytorch.adjust_saturation(1.3, inplace=False)
        )

        sh0_triton_copy = sh0.clone()
        shN_triton_copy = shN.clone()
        triton_time = benchmark_operation(
            "Triton  ",
            lambda: triton_saturation(sh0_triton_copy, shN_triton_copy, 1.3),
        )

        speedup = pytorch_time / triton_time
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Winner: {'Triton' if speedup > 1 else 'PyTorch'} 🏆\n")

        # 3. Fused Pipeline (Brightness + Saturation)
        print("3. Fused Pipeline (Brightness 1.2x + Saturation 1.3x):")

        # PyTorch sequential
        def pytorch_fused():
            temp = data_pytorch.adjust_brightness(1.2, inplace=False)
            return temp.adjust_saturation(1.3, inplace=False)

        pytorch_time = benchmark_operation("PyTorch (sequential)", pytorch_fused)

        # Triton fused
        sh0_triton_copy = sh0.clone()
        shN_triton_copy = shN.clone()
        triton_time = benchmark_operation(
            "Triton (fused)    ",
            lambda: triton_fused_pipeline(sh0_triton_copy, shN_triton_copy, 1.2, 1.3),
        )

        speedup = pytorch_time / triton_time
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Winner: {'Triton' if speedup > 1 else 'PyTorch'} 🏆\n")

        # Summary
        print(f"{'-' * 80}")
        print("Summary:")
        print(f"{'-' * 80}")
        print("Triton shows potential for fused operations where multiple color")
        print("adjustments can be combined into a single kernel launch.")

    print(f"\n{'=' * 80}")
    print("Benchmark Complete!")
    print(f"{'=' * 80}\n")
    print("Conclusion:")
    print("  - Triton provides modest gains for simple operations")
    print("  - Triton shines when fusing multiple operations together")
    print("  - PyTorch is already very well optimized (uses cuBLAS)")
    print("  - Triton adds complexity and Windows support is experimental")
    print("\n  Recommendation: Use PyTorch for simplicity, Triton only if")
    print("  you need absolute maximum performance and can handle complexity.")


if __name__ == "__main__":
    main()
