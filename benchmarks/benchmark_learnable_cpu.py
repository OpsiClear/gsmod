"""Benchmark learnable module performance on CPU."""

import logging
import time

import torch

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

from gsmod.torch.learn import (
    ColorGradingConfig,
    LearnableColor,
    LearnableFilter,
    LearnableFilterConfig,
    LearnableTransform,
    TransformConfig,
)


def create_test_tensors(n: int, device: str = 'cpu'):
    """Create test tensors."""
    means = torch.randn(n, 3, device=device, dtype=torch.float32)
    scales = torch.rand(n, 3, device=device, dtype=torch.float32) * 0.1
    quats = torch.randn(n, 4, device=device, dtype=torch.float32)
    quats = quats / quats.norm(dim=1, keepdim=True)
    opacities = torch.rand(n, device=device, dtype=torch.float32)
    sh0 = torch.rand(n, 3, device=device, dtype=torch.float32)
    return means, scales, quats, opacities, sh0


def benchmark(func, warmup=5, iterations=50):
    """Benchmark a function."""
    for _ in range(warmup):
        func()

    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000


def run_benchmarks():
    """Run CPU benchmarks."""
    logger.info("=" * 70)
    logger.info("LEARNABLE MODULE CPU BENCHMARKS")
    logger.info("=" * 70)
    logger.info("")

    sizes = [10_000, 100_000, 1_000_000]

    for n in sizes:
        logger.info(f"\nDataset size: {n:,} Gaussians")
        logger.info("-" * 70)

        means, scales, quats, opacities, sh0 = create_test_tensors(n, device='cpu')

        # LearnableColor
        config = ColorGradingConfig(
            learnable=['brightness', 'saturation', 'contrast'],
            device='cpu'
        )
        model = LearnableColor(config)

        fwd_ms = benchmark(lambda: model(sh0))
        throughput = (n / fwd_ms) * 1000 / 1e6
        logger.info(f"LearnableColor:    {fwd_ms:.2f} ms ({throughput:.1f}M/sec)")

        # LearnableTransform
        config = TransformConfig(
            learnable=['translation', 'scale', 'rotation'],
            device='cpu'
        )
        model = LearnableTransform(config)

        fwd_ms = benchmark(lambda: model(means, scales, quats))
        throughput = (n / fwd_ms) * 1000 / 1e6
        logger.info(f"LearnableTransform: {fwd_ms:.2f} ms ({throughput:.1f}M/sec)")

        # LearnableFilter
        config = LearnableFilterConfig(
            opacity_threshold=0.3,
            sphere_radius=5.0,
            learnable=['opacity_threshold', 'sphere_radius'],
            device='cpu'
        )
        model = LearnableFilter(config)

        fwd_ms = benchmark(lambda: model(means, opacities, scales))
        throughput = (n / fwd_ms) * 1000 / 1e6
        logger.info(f"LearnableFilter:   {fwd_ms:.2f} ms ({throughput:.1f}M/sec)")

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    run_benchmarks()
