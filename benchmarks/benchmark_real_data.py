"""Benchmark GPU operations with real PLY data."""

import time
import os
from pathlib import Path
import numpy as np
import torch
import gsply
from gsmod.torch import GSTensorPro, PipelineGPU


def benchmark_real_file(file_path):
    """Benchmark operations on a real PLY file."""
    print(f"\n[PROCESSING: {file_path.name}]")
    print("=" * 60)

    # Load data
    start = time.perf_counter()
    data = gsply.plyread(file_path)
    load_time = time.perf_counter() - start
    print(f"Load from disk: {load_time*1000:.1f} ms")
    print(f"Gaussians: {len(data):,}")

    # Check GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Transfer to GPU
    start = time.perf_counter()
    gstensor = GSTensorPro.from_gsdata(data, device=device)
    gpu_transfer_time = time.perf_counter() - start
    print(f"Transfer to {device}: {gpu_transfer_time*1000:.1f} ms")

    # Create processing pipeline
    pipeline = (
        PipelineGPU()
        # Filter operations
        .within_sphere(radius=5.0)
        .min_opacity(0.01)
        .max_scale(0.5)
        # Transform operations
        .center_at_origin()
        .normalize_scale(target_size=2.0)
        # Color operations
        .brightness(1.1)
        .contrast(1.05)
        .saturation(1.2)
        .color_preset("cinematic", strength=0.5)
    )

    # Warmup
    _ = pipeline(gstensor.clone(), inplace=False)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark pipeline
    start = time.perf_counter()
    result = pipeline(gstensor.clone(), inplace=False)
    if device == 'cuda':
        torch.cuda.synchronize()
    pipeline_time = time.perf_counter() - start

    print(f"Pipeline processing: {pipeline_time*1000:.2f} ms")
    print(f"Output Gaussians: {len(result):,}")

    # Throughput
    throughput = len(data) / pipeline_time / 1e6
    print(f"Throughput: {throughput:.1f} M Gaussians/sec")

    # Memory usage
    if device == 'cuda':
        mem_allocated = torch.cuda.memory_allocated() / 1024**2
        mem_reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU Memory: {mem_allocated:.1f} MB allocated, {mem_reserved:.1f} MB reserved")

    return {
        'file': file_path.name,
        'gaussians': len(data),
        'output': len(result),
        'time': pipeline_time,
        'throughput': throughput
    }


def main():
    """Process real PLY files."""
    print("\n" + "=" * 60)
    print("REAL DATA GPU BENCHMARKS")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("\nRunning on CPU")

    # Look for PLY files
    data_dir = Path("D:/4D/all_plys")

    if not data_dir.exists():
        print(f"\nData directory not found: {data_dir}")
        print("Using sample data instead...")

        # Create a sample file for testing
        import gsply
        from gsply import GSData

        n = 500000
        np.random.seed(42)
        sample_data = GSData(
            means=np.random.randn(n, 3).astype(np.float32) * 2,
            scales=np.random.randn(n, 3).astype(np.float32) * 0.1,
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.rand(n).astype(np.float32),
            sh0=np.random.rand(n, 3).astype(np.float32),
            shN=None,
        )
        # Normalize quaternions
        norms = np.linalg.norm(sample_data.quats, axis=1, keepdims=True)
        sample_data.quats /= norms

        # Save and process
        sample_file = Path("sample_scene.ply")
        gsply.plywrite(sample_file, sample_data)
        benchmark_real_file(sample_file)
        sample_file.unlink()  # Clean up
        return

    # Find PLY files
    ply_files = list(data_dir.glob("*.ply"))[:5]  # Process first 5 files

    if not ply_files:
        print(f"No PLY files found in {data_dir}")
        return

    print(f"\nFound {len(ply_files)} PLY files")

    # Process each file
    results = []
    for file_path in ply_files:
        try:
            result = benchmark_real_file(file_path)
            results.append(result)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")

    # Summary
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        total_gaussians = sum(r['gaussians'] for r in results)
        total_output = sum(r['output'] for r in results)
        total_time = sum(r['time'] for r in results)
        avg_throughput = total_gaussians / total_time / 1e6

        print(f"\nProcessed files: {len(results)}")
        print(f"Total Gaussians: {total_gaussians:,}")
        print(f"Total output: {total_output:,}")
        print(f"Total time: {total_time*1000:.1f} ms")
        print(f"Average throughput: {avg_throughput:.1f} M Gaussians/sec")

        print("\nPer-file results:")
        for r in results:
            print(f"  {r['file']:30} {r['gaussians']:10,} -> {r['output']:10,} "
                  f"({r['time']*1000:6.1f} ms, {r['throughput']:5.1f} M/s)")


if __name__ == "__main__":
    main()