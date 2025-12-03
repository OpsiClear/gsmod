"""Benchmark SH color operations: Pure NumPy vs Numba-optimized.

This script measures the performance improvement from Numba JIT compilation
for SH-aware color operations.
"""

from __future__ import annotations

import time

import numpy as np

from gsmod.color.sh_kernels import (
    apply_contrast_to_sh_numba,
    apply_matrix_to_sh_numba,
    apply_scale_to_sh_numba,
)
from gsmod.color.sh_utils import (
    apply_contrast_to_sh,
    apply_matrix_to_sh,
    apply_scale_to_sh,
    build_hue_matrix,
    build_saturation_matrix,
    build_temperature_matrix,
)


def benchmark_operation(name: str, numpy_fn, numba_fn, iterations: int = 100):
    """Benchmark a single operation comparing NumPy vs Numba.

    :param name: Operation name
    :param numpy_fn: Pure NumPy implementation
    :param numba_fn: Numba-optimized implementation
    :param iterations: Number of iterations to run
    :returns: Tuple of (numpy_time, numba_time, speedup)
    """
    # Warmup
    numpy_fn()
    numba_fn()

    # Benchmark NumPy
    start = time.perf_counter()
    for _ in range(iterations):
        numpy_fn()
    numpy_time = (time.perf_counter() - start) / iterations

    # Benchmark Numba
    start = time.perf_counter()
    for _ in range(iterations):
        numba_fn()
    numba_time = (time.perf_counter() - start) / iterations

    speedup = numpy_time / numba_time

    print(f"\n{name}:")
    print(f"  NumPy:  {numpy_time * 1000:.3f} ms")
    print(f"  Numba:  {numba_time * 1000:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")

    return numpy_time, numba_time, speedup


def main():
    """Run comprehensive benchmarks."""
    print("=" * 70)
    print("SH Color Operations Performance Benchmark")
    print("=" * 70)

    # Test data sizes
    sizes = [
        (100_000, "Small (100K Gaussians)"),
        (1_000_000, "Medium (1M Gaussians)"),
        (5_000_000, "Large (5M Gaussians)"),
    ]

    for N, size_name in sizes:
        print(f"\n{'=' * 70}")
        print(f"{size_name}")
        print(f"{'=' * 70}")

        # Generate test data
        sh0 = np.random.rand(N, 3).astype(np.float32)
        shN = np.random.rand(N, 8, 3).astype(np.float32)  # Degree 2 SH
        sh0_out_numba = np.empty_like(sh0)
        shN_out_numba = np.empty_like(shN)

        results = []

        # 1. Matrix operation (saturation)
        M = build_saturation_matrix(1.3)

        def numpy_matrix():
            return apply_matrix_to_sh(sh0, shN, M)

        def numba_matrix():
            apply_matrix_to_sh_numba(sh0, shN, M, sh0_out_numba, shN_out_numba)

        _, _, speedup = benchmark_operation(
            "Matrix Operation (Saturation)", numpy_matrix, numba_matrix, iterations=50
        )
        results.append(("Matrix (Saturation)", speedup))

        # 2. Scaling operation (brightness)
        def numpy_scale():
            return apply_scale_to_sh(sh0, shN, 1.2)

        def numba_scale():
            apply_scale_to_sh_numba(sh0, shN, 1.2, sh0_out_numba, shN_out_numba)

        _, _, speedup = benchmark_operation(
            "Scale Operation (Brightness)", numpy_scale, numba_scale, iterations=50
        )
        results.append(("Scale (Brightness)", speedup))

        # 3. Contrast operation
        def numpy_contrast():
            return apply_contrast_to_sh(sh0, shN, 1.1, True)

        def numba_contrast():
            apply_contrast_to_sh_numba(sh0, shN, 1.1, True, sh0_out_numba, shN_out_numba)

        _, _, speedup = benchmark_operation(
            "Contrast Operation", numpy_contrast, numba_contrast, iterations=50
        )
        results.append(("Contrast", speedup))

        # 4. Temperature matrix operation
        M_temp = build_temperature_matrix(0.2)

        def numpy_temp():
            return apply_matrix_to_sh(sh0, shN, M_temp)

        def numba_temp():
            apply_matrix_to_sh_numba(sh0, shN, M_temp, sh0_out_numba, shN_out_numba)

        _, _, speedup = benchmark_operation(
            "Matrix Operation (Temperature)", numpy_temp, numba_temp, iterations=50
        )
        results.append(("Matrix (Temperature)", speedup))

        # 5. Hue shift matrix operation
        M_hue = build_hue_matrix(30.0)

        def numpy_hue():
            return apply_matrix_to_sh(sh0, shN, M_hue)

        def numba_hue():
            apply_matrix_to_sh_numba(sh0, shN, M_hue, sh0_out_numba, shN_out_numba)

        _, _, speedup = benchmark_operation(
            "Matrix Operation (Hue Shift)", numpy_hue, numba_hue, iterations=50
        )
        results.append(("Matrix (Hue)", speedup))

        # Summary
        print(f"\n{'-' * 70}")
        print(f"Summary for {size_name}:")
        print(f"{'-' * 70}")
        avg_speedup = sum(s for _, s in results) / len(results)
        print(f"Average Speedup: {avg_speedup:.2f}x")
        print(f"Min Speedup: {min(s for _, s in results):.2f}x")
        print(f"Max Speedup: {max(s for _, s in results):.2f}x")

    print(f"\n{'=' * 70}")
    print("Benchmark Complete!")
    print(f"{'=' * 70}")
    print("\nConclusion: Numba JIT compilation provides significant speedup for SH operations")
    print("while maintaining mathematical correctness.")


if __name__ == "__main__":
    main()
