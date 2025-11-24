"""Benchmark axis_angle vs 6D rotation representations."""

import logging
import time

import torch

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

if not torch.cuda.is_available():
    logger.error("CUDA not available. Benchmarks require GPU.")
    exit(1)

from gsmod.torch.learn import LearnableTransform, TransformConfig


def create_test_tensors(n: int, device: str = 'cuda'):
    """Create test tensors."""
    means = torch.randn(n, 3, device=device, dtype=torch.float32)
    scales = torch.rand(n, 3, device=device, dtype=torch.float32) * 0.1
    quats = torch.randn(n, 4, device=device, dtype=torch.float32)
    quats = quats / quats.norm(dim=1, keepdim=True)
    return means, scales, quats


def benchmark(func, warmup=20, iterations=200):
    """Benchmark a function."""
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        func()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000


def run_benchmarks():
    """Run rotation representation benchmarks."""
    logger.info("=" * 70)
    logger.info("ROTATION REPRESENTATION BENCHMARKS")
    logger.info("=" * 70)
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("")

    n = 1_000_000
    means, scales, quats = create_test_tensors(n)

    # Axis-angle representation
    config_aa = TransformConfig(
        rotation_repr='axis_angle',
        learnable=['translation', 'scale', 'rotation']
    )
    model_aa = LearnableTransform(config_aa).cuda()

    # 6D representation
    config_6d = TransformConfig(
        rotation_repr='6d',
        learnable=['translation', 'scale', 'rotation']
    )
    model_6d = LearnableTransform(config_6d).cuda()

    # Forward benchmarks
    logger.info(f"Dataset size: {n:,} Gaussians")
    logger.info("")

    aa_ms = benchmark(lambda: model_aa(means, scales, quats))
    aa_throughput = (n / aa_ms) * 1000 / 1e6
    logger.info(f"Axis-angle: {aa_ms:.3f} ms ({aa_throughput:.1f}M/sec)")

    six_ms = benchmark(lambda: model_6d(means, scales, quats))
    six_throughput = (n / six_ms) * 1000 / 1e6
    logger.info(f"6D repr:    {six_ms:.3f} ms ({six_throughput:.1f}M/sec)")

    ratio = aa_ms / six_ms
    if ratio > 1:
        logger.info(f"\n6D is {ratio:.2f}x faster than axis-angle")
    else:
        logger.info(f"\nAxis-angle is {1/ratio:.2f}x faster than 6D")

    # Forward + Backward benchmarks
    logger.info("\n" + "-" * 70)
    logger.info("Forward + Backward")
    logger.info("-" * 70)

    means_grad = means.clone().requires_grad_(True)

    def aa_fwd_bwd():
        means_grad.grad = None
        out = model_aa(means_grad, scales, quats)
        loss = out[0].sum()
        loss.backward()

    def six_fwd_bwd():
        means_grad.grad = None
        out = model_6d(means_grad, scales, quats)
        loss = out[0].sum()
        loss.backward()

    aa_ms = benchmark(aa_fwd_bwd)
    aa_throughput = (n / aa_ms) * 1000 / 1e6
    logger.info(f"Axis-angle: {aa_ms:.3f} ms ({aa_throughput:.1f}M/sec)")

    six_ms = benchmark(six_fwd_bwd)
    six_throughput = (n / six_ms) * 1000 / 1e6
    logger.info(f"6D repr:    {six_ms:.3f} ms ({six_throughput:.1f}M/sec)")

    ratio = aa_ms / six_ms
    if ratio > 1:
        logger.info(f"\n6D is {ratio:.2f}x faster than axis-angle")
    else:
        logger.info(f"\nAxis-angle is {1/ratio:.2f}x faster than 6D")

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    run_benchmarks()
