"""
Run all performance benchmarks.

Benchmarks included:
- benchmark_color.py: Color processing (LUT + saturation/shadows/highlights)
- benchmark_transform.py: 3D Gaussian transforms (translate/rotate/scale)
- benchmark_filter.py: Gaussian filtering (volume/opacity/scale filters)
"""

import subprocess
import sys
from pathlib import Path

# Benchmark files to run
benchmarks = [
    "benchmark_color.py",
    "benchmark_transform.py",
    "benchmark_filter.py",
]

print("=" * 80)
print("GSPRO PERFORMANCE BENCHMARKS")
print("=" * 80)

# Get benchmark directory
benchmark_dir = Path(__file__).parent

failed_benchmarks = []

for benchmark in benchmarks:
    print(f"\n{'=' * 80}")
    print(f"Running {benchmark}...")
    print("=" * 80)

    benchmark_path = benchmark_dir / benchmark
    result = subprocess.run(
        [sys.executable, str(benchmark_path)], capture_output=False, cwd=str(benchmark_dir)
    )

    if result.returncode != 0:
        print(f"\n[FAIL] {benchmark} failed with exit code {result.returncode}")
        failed_benchmarks.append(benchmark)
    else:
        print(f"\n[OK] {benchmark} completed successfully")

print("\n" + "=" * 80)
if failed_benchmarks:
    print(f"FAILED: {len(failed_benchmarks)}/{len(benchmarks)} benchmarks")
    for bench in failed_benchmarks:
        print(f"  - {bench}")
    print("=" * 80)
    sys.exit(1)
else:
    print(f"SUCCESS: All {len(benchmarks)} benchmarks completed")
    print("=" * 80)
