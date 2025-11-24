"""Benchmark GSTensorPro with latest gsply features."""

import time
from pathlib import Path

import numpy as np
import torch

# Import gsply components
from gsply import GSData, plyread, plywrite
from gsply.torch import GSTensor, plyread_gpu

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


def benchmark_gpu_io(data, temp_file="temp_test.ply"):
    """Benchmark GPU I/O operations."""
    print("\n[GPU I/O BENCHMARK]")
    print("=" * 60)

    # Save test file
    plywrite(temp_file, data, compressed=True)

    # Benchmark CPU load -> GPU transfer
    start = time.perf_counter()
    data_cpu = plyread(temp_file)
    GSTensor.from_gsdata(data_cpu, device="cuda")
    torch.cuda.synchronize()
    cpu_path_time = time.perf_counter() - start

    # Benchmark direct GPU load
    start = time.perf_counter()
    plyread_gpu(temp_file, device="cuda")
    torch.cuda.synchronize()
    gpu_path_time = time.perf_counter() - start

    # Benchmark GSTensorPro.load
    start = time.perf_counter()
    gstensor_pro = GSTensorPro.load(temp_file, device="cuda")
    torch.cuda.synchronize()
    pro_load_time = time.perf_counter() - start

    print(f"CPU load + GPU transfer: {cpu_path_time*1000:.2f} ms")
    print(f"Direct GPU load (gsply): {gpu_path_time*1000:.2f} ms")
    print(f"GSTensorPro.load: {pro_load_time*1000:.2f} ms")
    print(f"Speedup: {cpu_path_time/gpu_path_time:.1f}x")

    # Clean up
    Path(temp_file).unlink()

    return gstensor_pro


def benchmark_mask_layers(gstensor):
    """Benchmark mask layer operations."""
    print("\n[MASK LAYER BENCHMARK]")
    print("=" * 60)

    # Create multiple mask layers
    start = time.perf_counter()

    # Add mask layers
    gstensor.add_mask_layer("opacity_high", gstensor.opacities > 0.5)
    gstensor.add_mask_layer("near_origin", torch.norm(gstensor.means, dim=1) < 2.0)
    gstensor.add_mask_layer("small_scale", torch.max(gstensor.scales, dim=1)[0] < 0.1)

    # Combine masks
    gstensor.combine_masks(mode="and")

    torch.cuda.synchronize()
    mask_create_time = time.perf_counter() - start

    # Apply masks
    start = time.perf_counter()
    filtered = gstensor.apply_masks(mode="and", inplace=False)
    torch.cuda.synchronize()
    mask_apply_time = time.perf_counter() - start

    print(f"Create 3 masks: {mask_create_time*1000:.2f} ms")
    print(f"Apply masks: {mask_apply_time*1000:.2f} ms")
    print(f"Filtered: {len(gstensor):,} -> {len(filtered):,}")

    return filtered


def benchmark_format_conversions(gstensor):
    """Benchmark format conversion operations."""
    print("\n[FORMAT CONVERSION BENCHMARK]")
    print("=" * 60)

    # Clone for testing
    test_tensor = gstensor.clone()

    # to_rgb conversion
    start = time.perf_counter()
    test_tensor.to_rgb(inplace=True)
    torch.cuda.synchronize()
    to_rgb_time = time.perf_counter() - start

    # normalize (to runtime format)
    start = time.perf_counter()
    test_tensor.normalize(inplace=True)
    torch.cuda.synchronize()
    normalize_time = time.perf_counter() - start

    # denormalize (to PLY format)
    start = time.perf_counter()
    test_tensor.denormalize(inplace=True)
    torch.cuda.synchronize()
    denormalize_time = time.perf_counter() - start

    print(f"to_rgb: {to_rgb_time*1000:.3f} ms")
    print(f"normalize: {normalize_time*1000:.3f} ms")
    print(f"denormalize: {denormalize_time*1000:.3f} ms")


def benchmark_gpu_compression(gstensor, temp_file="temp_compressed.ply"):
    """Benchmark GPU compression."""
    print("\n[GPU COMPRESSION BENCHMARK]")
    print("=" * 60)

    # GPU compression via GSTensorPro.save
    start = time.perf_counter()
    gstensor.save(temp_file, compressed=True)
    torch.cuda.synchronize()
    gpu_compress_time = time.perf_counter() - start

    file_size = Path(temp_file).stat().st_size / 1024**2  # MB

    # CPU compression comparison
    data = gstensor.to_gsdata()
    start = time.perf_counter()
    plywrite(temp_file, data, compressed=True)
    cpu_compress_time = time.perf_counter() - start

    print(f"GPU compression: {gpu_compress_time*1000:.2f} ms")
    print(f"CPU compression: {cpu_compress_time*1000:.2f} ms")
    print(f"Speedup: {cpu_compress_time/gpu_compress_time:.1f}x")
    print(f"File size: {file_size:.1f} MB")
    print(f"Throughput: {len(gstensor)/gpu_compress_time/1e6:.1f} M Gaussians/sec")

    # Clean up
    Path(temp_file).unlink()


def benchmark_full_pipeline(gstensor):
    """Benchmark full processing pipeline."""
    print("\n[FULL PIPELINE BENCHMARK]")
    print("=" * 60)

    pipeline = (
        PipelineGPU()
        # Add mask layers
        .filter_within_sphere(radius=3.0, save_mask="sphere")
        .filter_min_opacity(0.01, save_mask="opacity")
        # Transforms
        .center_at_origin()
        .normalize_scale(2.0)
        # Colors
        .brightness(1.1)
        .saturation(1.2)
        .color_preset("cinematic", strength=0.3)
    )

    # Warmup
    _ = pipeline(gstensor.clone(), inplace=False)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    result = pipeline(gstensor.clone(), inplace=False)
    torch.cuda.synchronize()
    pipeline_time = time.perf_counter() - start

    print(f"Pipeline time: {pipeline_time*1000:.2f} ms")
    print(f"Input: {len(gstensor):,} Gaussians")
    print(f"Output: {len(result):,} Gaussians")
    print(f"Throughput: {len(gstensor)/pipeline_time/1e6:.1f} M Gaussians/sec")

    return result


def main():
    """Run comprehensive benchmarks."""
    print("\n" + "=" * 60)
    print("GSPRO + GSPLY INTEGRATION BENCHMARKS")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    # Test sizes
    sizes = [100_000, 500_000, 1_000_000]

    for n in sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {n:,} Gaussians")
        print(f"{'='*60}")

        # Create test data
        data = create_test_data(n)

        # GPU I/O benchmarks
        gstensor = benchmark_gpu_io(data)

        # Mask layer benchmarks
        benchmark_mask_layers(gstensor)

        # Format conversions
        benchmark_format_conversions(gstensor)

        # GPU compression
        benchmark_gpu_compression(gstensor)

        # Full pipeline
        result = benchmark_full_pipeline(gstensor)

    # Test with real data if available
    data_dir = Path("D:/4D/all_plys")
    if data_dir.exists():
        print(f"\n{'='*60}")
        print("REAL DATA TEST")
        print(f"{'='*60}")

        ply_files = list(data_dir.glob("*.ply"))[:1]  # Test first file
        if ply_files:
            file_path = ply_files[0]
            print(f"\nFile: {file_path.name}")

            # Load with GSTensorPro
            start = time.perf_counter()
            gstensor = GSTensorPro.load(file_path, device="cuda")
            torch.cuda.synchronize()
            load_time = time.perf_counter() - start

            print(f"Load time: {load_time*1000:.1f} ms")
            print(f"Gaussians: {len(gstensor):,}")

            # Run pipeline
            result = benchmark_full_pipeline(gstensor)

            # Save result
            output_file = "output_processed.ply"
            start = time.perf_counter()
            result.save(output_file, compressed=True)
            torch.cuda.synchronize()
            save_time = time.perf_counter() - start

            print(f"\nSaved to {output_file}: {save_time*1000:.1f} ms")

            # Clean up
            Path(output_file).unlink()


if __name__ == "__main__":
    main()
