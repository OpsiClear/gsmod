"""Simple benchmark to test GPU vs CPU performance."""

import time
import numpy as np
import torch
from gsply import GSData
from gsmod.torch import GSTensorPro


def create_test_data(n_gaussians):
    """Create test Gaussian Splatting data."""
    np.random.seed(42)

    data = GSData(
        means=np.random.randn(n_gaussians, 3).astype(np.float32),
        scales=np.random.randn(n_gaussians, 3).astype(np.float32) * 0.1,
        quats=np.random.randn(n_gaussians, 4).astype(np.float32),
        opacities=np.random.rand(n_gaussians).astype(np.float32),
        sh0=np.random.rand(n_gaussians, 3).astype(np.float32),
        shN=None,
    )

    # Normalize quaternions
    norms = np.linalg.norm(data.quats, axis=1, keepdims=True)
    data.quats /= norms

    return data


def benchmark_basic_operations():
    """Benchmark basic GPU operations."""
    print("\nGPU PERFORMANCE TEST")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Running on CPU (no GPU available)")

    # Test different sizes
    sizes = [10_000, 100_000, 500_000, 1_000_000]

    for n in sizes:
        print(f"\n{n:,} Gaussians:")
        data = create_test_data(n)

        # Load to GPU
        start = time.perf_counter()
        gstensor = GSTensorPro.from_gsdata(data, device=device)
        load_time = time.perf_counter() - start
        print(f"  Load to {device}: {load_time*1000:.1f} ms")

        # Brightness adjustment
        start = time.perf_counter()
        gstensor.adjust_brightness(1.2, inplace=True)
        if device == 'cuda':
            torch.cuda.synchronize()
        brightness_time = time.perf_counter() - start
        print(f"  Brightness: {brightness_time*1000:.2f} ms")

        # Translation
        start = time.perf_counter()
        gstensor.translate([1.0, 0.0, 0.5], inplace=True)
        if device == 'cuda':
            torch.cuda.synchronize()
        translate_time = time.perf_counter() - start
        print(f"  Translate: {translate_time*1000:.2f} ms")

        # Filtering
        start = time.perf_counter()
        mask = gstensor.filter_within_sphere(radius=2.0)
        filtered = gstensor[mask]
        if device == 'cuda':
            torch.cuda.synchronize()
        filter_time = time.perf_counter() - start
        print(f"  Filter: {filter_time*1000:.2f} ms (kept {len(filtered):,})")

        # Total throughput
        total_time = brightness_time + translate_time + filter_time
        throughput = n / total_time / 1e6
        print(f"  Throughput: {throughput:.1f} M Gaussians/sec")


if __name__ == "__main__":
    benchmark_basic_operations()