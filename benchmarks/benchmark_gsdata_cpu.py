"""Benchmark GSDataPro CPU operations."""

import logging
import time

import numpy as np

from gsmod import ColorValues, FilterValues, GSDataPro, TransformValues

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def create_test_data(n: int) -> GSDataPro:
    """Create test GSDataPro."""
    np.random.seed(42)

    data = GSDataPro.__new__(GSDataPro)
    data.means = np.random.randn(n, 3).astype(np.float32)
    data.scales = (np.random.rand(n, 3) * 0.5 + 0.1).astype(np.float32)
    data.quats = np.random.randn(n, 4).astype(np.float32)
    data.quats /= np.linalg.norm(data.quats, axis=1, keepdims=True)
    data.opacities = np.random.rand(n).astype(np.float32)
    data.sh0 = np.random.rand(n, 3).astype(np.float32)
    data.shN = None

    return data


def benchmark(func, warmup=3, iterations=20):
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
    logger.info("GSDATA CPU BENCHMARKS")
    logger.info("=" * 70)
    logger.info("")

    sizes = [10_000, 100_000, 1_000_000]

    for n in sizes:
        logger.info(f"\nDataset size: {n:,} Gaussians")
        logger.info("-" * 70)

        # Color operations
        data = create_test_data(n)
        values = ColorValues(brightness=1.2, saturation=1.3, contrast=1.1)

        def color_op():
            d = data.clone()
            d.color(values, inplace=True)

        fwd_ms = benchmark(color_op)
        throughput = (n / fwd_ms) * 1000 / 1e6
        logger.info(f"Color:     {fwd_ms:.2f} ms ({throughput:.1f}M/sec)")

        # Transform operations
        data = create_test_data(n)
        values = TransformValues.from_translation(1.0, 0.0, 0.5)

        def transform_op():
            d = data.clone()
            d.transform(values, inplace=True)

        fwd_ms = benchmark(transform_op)
        throughput = (n / fwd_ms) * 1000 / 1e6
        logger.info(f"Transform: {fwd_ms:.2f} ms ({throughput:.1f}M/sec)")

        # Filter operations (FilterValues)
        data = create_test_data(n)
        values = FilterValues(min_opacity=0.3, sphere_radius=5.0)

        def filter_op():
            d = data.clone()
            d.filter(values, inplace=True)

        fwd_ms = benchmark(filter_op)
        throughput = (n / fwd_ms) * 1000 / 1e6
        logger.info(f"Filter:    {fwd_ms:.2f} ms ({throughput:.1f}M/sec)")

    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    run_benchmarks()
