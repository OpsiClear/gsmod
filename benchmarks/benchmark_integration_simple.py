"""Benchmark GSTensorPro integration with current gsply."""

import time
from pathlib import Path
import numpy as np
import torch

# Import gsply components
from gsply import GSData, plywrite, plyread
from gsply.torch import GSTensor

# Import our enhanced GSTensorPro
from gsmod.torch import GSTensorPro, PipelineGPU


def create_test_data(n_gaussians):
    """Create test Gaussian Splatting data."""
    np.random.seed(42)

    data = GSData(
        means=np.random.randn(n_gaussians, 3).astype(np.float32) * 2,
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


def benchmark_comparison():
    """Compare GSTensor vs GSTensorPro."""
    print("\n" + "=" * 60)
    print("GSTENSOR vs GSTENSORPRO COMPARISON")
    print("=" * 60)

    sizes = [10_000, 100_000, 500_000]

    for n in sizes:
        print(f"\n[{n:,} Gaussians]")
        print("-" * 40)

        # Create test data
        data = create_test_data(n)

        # Load to GPU with both methods
        start = time.perf_counter()
        gstensor = GSTensor.from_gsdata(data, device='cuda')
        torch.cuda.synchronize()
        gs_load_time = time.perf_counter() - start

        start = time.perf_counter()
        gstensor_pro = GSTensorPro.from_gsdata(data, device='cuda')
        torch.cuda.synchronize()
        pro_load_time = time.perf_counter() - start

        print(f"Load time:")
        print(f"  GSTensor:    {gs_load_time*1000:.2f} ms")
        print(f"  GSTensorPro: {pro_load_time*1000:.2f} ms")

        # Test basic operations
        print(f"\nOperations:")

        # GSTensor doesn't have these operations
        print("  GSTensor:    No color/transform ops")

        # GSTensorPro operations
        start = time.perf_counter()
        gstensor_pro.adjust_brightness(1.2, inplace=True)
        gstensor_pro.adjust_saturation(1.3, inplace=True)
        gstensor_pro.translate([1, 0, 0], inplace=True)
        mask = gstensor_pro.filter_within_sphere(radius=2.0)
        filtered = gstensor_pro[mask]
        torch.cuda.synchronize()
        pro_ops_time = time.perf_counter() - start

        print(f"  GSTensorPro: {pro_ops_time*1000:.2f} ms")
        print(f"  Filtered:    {len(filtered):,} Gaussians")


def benchmark_pipeline():
    """Benchmark full pipeline processing."""
    print("\n" + "=" * 60)
    print("PIPELINE PROCESSING BENCHMARK")
    print("=" * 60)

    # Test with real data if available
    data_dir = Path("D:/4D/all_plys")
    if data_dir.exists():
        ply_files = list(data_dir.glob("*.ply"))[:3]
    else:
        # Create test files
        print("Creating test files...")
        ply_files = []
        for i, n in enumerate([100_000, 200_000, 400_000]):
            data = create_test_data(n)
            filename = f"test_{i}.ply"
            plywrite(filename, data)
            ply_files.append(Path(filename))

    total_time = 0
    total_gaussians = 0

    for file_path in ply_files:
        print(f"\n[{file_path.name}]")
        print("-" * 40)

        # Load file
        start_total = time.perf_counter()

        data = plyread(file_path)
        gstensor = GSTensorPro.from_gsdata(data, device='cuda')

        print(f"Gaussians: {len(gstensor):,}")

        # Create comprehensive pipeline
        pipeline = (
            PipelineGPU()
            # Filtering
            .within_sphere(radius=5.0)
            .min_opacity(0.01)
            .max_scale(0.5)
            # Transforms
            .center_at_origin()
            .normalize_scale(2.0)
            .rotate_axis_angle([0, 1, 0], np.pi/8)
            # Colors
            .brightness(1.1)
            .contrast(1.05)
            .saturation(1.25)
            .temperature(0.05)
            .vibrance(1.1)
        )

        # Process
        result = pipeline(gstensor, inplace=False)
        torch.cuda.synchronize()

        total_elapsed = time.perf_counter() - start_total

        print(f"Output: {len(result):,}")
        print(f"Total time: {total_elapsed*1000:.1f} ms")
        print(f"Throughput: {len(data)/total_elapsed/1e6:.1f} M/sec")

        total_time += total_elapsed
        total_gaussians += len(data)

        # Clean up test files if we created them
        if not data_dir.exists():
            file_path.unlink()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files processed: {len(ply_files)}")
    print(f"Total Gaussians: {total_gaussians:,}")
    print(f"Total time: {total_time:.2f} sec")
    print(f"Average throughput: {total_gaussians/total_time/1e6:.1f} M Gaussians/sec")


def benchmark_memory():
    """Benchmark GPU memory usage."""
    print("\n" + "=" * 60)
    print("GPU MEMORY USAGE")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    torch.cuda.reset_peak_memory_stats()

    sizes = [100_000, 500_000, 1_000_000]

    for n in sizes:
        # Clear GPU memory
        torch.cuda.empty_cache()

        print(f"\n[{n:,} Gaussians]")

        # Create and load data
        data = create_test_data(n)
        gstensor = GSTensorPro.from_gsdata(data, device='cuda')

        # Check memory
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2    # MB

        print(f"  Allocated: {allocated:.1f} MB")
        print(f"  Reserved:  {reserved:.1f} MB")
        print(f"  Per Gaussian: {allocated*1024/n:.1f} KB")

        # Run operations to see peak usage
        pipeline = PipelineGPU().brightness(1.2).translate([1, 0, 0])
        result = pipeline(gstensor, inplace=False)

        peak = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"  Peak usage: {peak:.1f} MB")

        del gstensor, result
        torch.cuda.empty_cache()


def main():
    """Run all benchmarks."""
    print("\n" + "=" * 60)
    print("GSTENSORPRO INTEGRATION BENCHMARKS")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__}")
    else:
        print("\nWARNING: CUDA not available, running on CPU")

    # Run benchmarks
    benchmark_comparison()
    benchmark_pipeline()
    benchmark_memory()

    print("\n" + "=" * 60)
    print("BENCHMARKS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()