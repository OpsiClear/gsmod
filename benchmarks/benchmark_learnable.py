"""Benchmark learnable module performance."""

import time
import torch
import numpy as np

# Check CUDA availability
if not torch.cuda.is_available():
    print("CUDA not available. Benchmarks require GPU.")
    exit(1)

from gsmod.torch.learn import (
    ColorGradingConfig,
    LearnableColor,
    LearnableFilter,
    LearnableFilterConfig,
    LearnableGSTensor,
    LearnableTransform,
    TransformConfig,
)


def create_test_tensors(n: int, device: str = 'cuda'):
    """Create test tensors for benchmarking."""
    means = torch.randn(n, 3, device=device, dtype=torch.float32)
    scales = torch.rand(n, 3, device=device, dtype=torch.float32) * 0.1
    quats = torch.randn(n, 4, device=device, dtype=torch.float32)
    quats = quats / quats.norm(dim=1, keepdim=True)
    opacities = torch.rand(n, device=device, dtype=torch.float32)
    sh0 = torch.rand(n, 3, device=device, dtype=torch.float32)
    return means, scales, quats, opacities, sh0


def benchmark_forward(func, name, warmup=10, iterations=100):
    """Benchmark forward pass."""
    # Warmup
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iterations) * 1000
    return avg_ms


def benchmark_forward_backward(forward_func, backward_func, name, warmup=10, iterations=100):
    """Benchmark forward + backward pass."""
    # Warmup
    for _ in range(warmup):
        output = forward_func()
        backward_func(output)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        output = forward_func()
        backward_func(output)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iterations) * 1000
    return avg_ms


def run_benchmarks():
    """Run all benchmarks."""
    print("=" * 70)
    print("LEARNABLE MODULE PERFORMANCE BENCHMARKS")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Test sizes
    sizes = [10_000, 100_000, 1_000_000]

    results = []

    for n in sizes:
        print(f"\n{'='*70}")
        print(f"Dataset size: {n:,} Gaussians")
        print("=" * 70)

        means, scales, quats, opacities, sh0 = create_test_tensors(n)
        sh0_grad = sh0.clone().requires_grad_(True)

        # =====================================================================
        # LearnableColor Benchmarks
        # =====================================================================
        print("\n[LearnableColor]")

        config = ColorGradingConfig(learnable=['brightness', 'saturation', 'contrast'])
        model = LearnableColor(config).cuda()

        # Forward only
        def color_forward():
            return model(sh0)

        fwd_ms = benchmark_forward(color_forward, "LearnableColor forward")
        throughput = (n / fwd_ms) * 1000 / 1e6
        print(f"  Forward:          {fwd_ms:.3f} ms ({throughput:.1f}M Gaussians/sec)")

        # Forward + Backward
        def color_forward_grad():
            sh0_grad.grad = None
            return model(sh0_grad)

        def color_backward(output):
            loss = output.sum()
            loss.backward()

        fwd_bwd_ms = benchmark_forward_backward(color_forward_grad, color_backward, "LearnableColor fwd+bwd")
        throughput = (n / fwd_bwd_ms) * 1000 / 1e6
        print(f"  Forward+Backward: {fwd_bwd_ms:.3f} ms ({throughput:.1f}M Gaussians/sec)")

        results.append(('LearnableColor', n, fwd_ms, fwd_bwd_ms))

        # =====================================================================
        # LearnableTransform Benchmarks
        # =====================================================================
        print("\n[LearnableTransform]")

        config = TransformConfig(learnable=['translation', 'scale'])
        model = LearnableTransform(config).cuda()

        means_grad = means.clone().requires_grad_(True)

        # Forward only
        def transform_forward():
            return model(means, scales, quats)

        fwd_ms = benchmark_forward(transform_forward, "LearnableTransform forward")
        throughput = (n / fwd_ms) * 1000 / 1e6
        print(f"  Forward:          {fwd_ms:.3f} ms ({throughput:.1f}M Gaussians/sec)")

        # Forward + Backward
        def transform_forward_grad():
            means_grad.grad = None
            return model(means_grad, scales, quats)

        def transform_backward(output):
            new_means, new_scales, new_quats = output
            loss = new_means.sum() + new_scales.sum()
            loss.backward()

        fwd_bwd_ms = benchmark_forward_backward(transform_forward_grad, transform_backward, "LearnableTransform fwd+bwd")
        throughput = (n / fwd_bwd_ms) * 1000 / 1e6
        print(f"  Forward+Backward: {fwd_bwd_ms:.3f} ms ({throughput:.1f}M Gaussians/sec)")

        results.append(('LearnableTransform', n, fwd_ms, fwd_bwd_ms))

        # =====================================================================
        # LearnableFilter Benchmarks
        # =====================================================================
        print("\n[LearnableFilter]")

        config = LearnableFilterConfig(
            opacity_threshold=0.3,
            sphere_radius=5.0,
            learnable=['opacity_threshold', 'sphere_radius']
        )
        model = LearnableFilter(config).cuda()

        # Forward only
        def filter_forward():
            return model(means, opacities, scales)

        fwd_ms = benchmark_forward(filter_forward, "LearnableFilter forward")
        throughput = (n / fwd_ms) * 1000 / 1e6
        print(f"  Forward:          {fwd_ms:.3f} ms ({throughput:.1f}M Gaussians/sec)")

        # Forward + Backward
        means_grad = means.clone().requires_grad_(True)

        def filter_forward_grad():
            means_grad.grad = None
            return model(means_grad, opacities, scales)

        def filter_backward(output):
            loss = output.sum()
            loss.backward()

        fwd_bwd_ms = benchmark_forward_backward(filter_forward_grad, filter_backward, "LearnableFilter fwd+bwd")
        throughput = (n / fwd_bwd_ms) * 1000 / 1e6
        print(f"  Forward+Backward: {fwd_bwd_ms:.3f} ms ({throughput:.1f}M Gaussians/sec)")

        results.append(('LearnableFilter', n, fwd_ms, fwd_bwd_ms))

        # =====================================================================
        # LearnableGSTensor Chained Operations
        # =====================================================================
        print("\n[LearnableGSTensor Chained]")

        data = LearnableGSTensor(means, scales, quats, opacities, sh0)
        color_model = LearnableColor().cuda()
        transform_model = LearnableTransform().cuda()

        # Forward only
        def chain_forward():
            return data.apply_transform(transform_model).apply_color(color_model)

        fwd_ms = benchmark_forward(chain_forward, "Chained forward")
        throughput = (n / fwd_ms) * 1000 / 1e6
        print(f"  Forward:          {fwd_ms:.3f} ms ({throughput:.1f}M Gaussians/sec)")

        # Forward + Backward
        sh0_grad = sh0.clone().requires_grad_(True)
        data_grad = LearnableGSTensor(means, scales, quats, opacities, sh0_grad)

        def chain_forward_grad():
            sh0_grad.grad = None
            return data_grad.apply_transform(transform_model).apply_color(color_model)

        def chain_backward(output):
            loss = output.sh0.sum()
            loss.backward()

        fwd_bwd_ms = benchmark_forward_backward(chain_forward_grad, chain_backward, "Chained fwd+bwd")
        throughput = (n / fwd_bwd_ms) * 1000 / 1e6
        print(f"  Forward+Backward: {fwd_bwd_ms:.3f} ms ({throughput:.1f}M Gaussians/sec)")

        results.append(('Chained (Transform+Color)', n, fwd_ms, fwd_bwd_ms))

        # =====================================================================
        # Training Step (with optimizer)
        # =====================================================================
        print("\n[Training Step]")

        sh0_grad = sh0.clone().requires_grad_(True)
        data_grad = LearnableGSTensor(means, scales, quats, opacities, sh0_grad)
        target = torch.rand(n, 3, device='cuda')

        config = ColorGradingConfig(learnable=['brightness', 'contrast', 'saturation'])
        model = LearnableColor(config).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        def training_step():
            optimizer.zero_grad()
            result = data_grad.apply_color(model)
            loss = torch.nn.functional.mse_loss(result.sh0, target)
            loss.backward()
            optimizer.step()
            return loss

        # Warmup
        for _ in range(10):
            training_step()
        torch.cuda.synchronize()

        # Benchmark
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            training_step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        step_ms = (elapsed / iterations) * 1000
        throughput = (n / step_ms) * 1000 / 1e6
        print(f"  Training step:    {step_ms:.3f} ms ({throughput:.1f}M Gaussians/sec)")

        results.append(('Training Step', n, step_ms, step_ms))

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Operation':<30} {'Size':>10} {'Forward':>12} {'Fwd+Bwd':>12}")
    print("-" * 70)

    for name, size, fwd, fwd_bwd in results:
        print(f"{name:<30} {size:>10,} {fwd:>10.3f}ms {fwd_bwd:>10.3f}ms")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_benchmarks()
