# gsmod Benchmarks

Performance benchmarks for high-performance CPU processing library.

## Quick Start

```bash
# Run all benchmarks
cd benchmarks
uv run run_all_benchmarks.py

# Run individual benchmarks
uv run benchmark_color.py
uv run benchmark_transform.py
uv run benchmark_filter.py
uv run benchmark_mask_strategies.py

# Run optimization benchmarks (used in CI/CD)
uv run benchmark_optimizations.py
uv run benchmark_filter_micro.py

# Run large-scale benchmarks (1M+ Gaussians)
uv run benchmark_large_scale.py
```

## Active Benchmarks

### 1. Color Processing (`benchmark_color.py`)

Tests color adjustment performance using fused Numba kernels.

**Features:**
- Interleaved LUT layout for cache locality
- Fused Phase 1 (LUT) + Phase 2 (saturation/shadows/highlights)
- Branchless Phase 2 operations (1.8x speedup)
- Zero-copy API with pre-allocated buffers
- 7 color operations: temperature, brightness, contrast, gamma, saturation, shadows, highlights

**Tests:**
- Multiple API variants (apply, apply_numpy, apply_numpy_inplace)
- Batch size scaling (1K to 1M colors)
- Speedup comparisons

**Expected Results:**
- Zero-copy (apply_numpy_inplace): 877-1,011 M colors/sec
- With allocation (apply_numpy): 195 M colors/sec
- Standard API (apply): 134 M colors/sec

### 2. 3D Transform (`benchmark_transform.py`)

Tests 3D Gaussian transform performance with fused Numba kernel.

**Features:**
- Fused kernel: Single parallel loop combining all operations
- Custom matrix multiply (9x faster than BLAS for 3x3 matrices)
- Memory locality: Process each Gaussian completely
- Pre-allocated output buffers (zero-copy)

**Tests:**
- 1M Gaussians with 100 iterations
- Batch size scaling (10K to 2M)
- Real-world use cases (animation, real-time rendering)

**Expected Results:**
- Time: 1.479ms for 1M Gaussians
- Throughput: 676-1,111 M Gaussians/sec
- Real-time: 676 FPS (11x faster than 60 FPS target)

### 3. Filtering (`benchmark_filter.py`)

Tests Gaussian filtering performance with Numba-optimized kernels.

**Features:**
- Numba JIT compilation with parallel execution
- Volume filters: sphere and cuboid spatial selection
- Attribute filters: opacity and scale thresholds with fused kernel
- Combined filtering with AND logic
- Scene bounds and auto-threshold calculation
- Optimizations: fastmath=True, fused opacity+scale kernel (1.95x speedup)

**Tests:**
- Individual filters (opacity, scale, sphere, cuboid)
- Fused filters (opacity+scale combined)
- Full attribute filtering (filter_gaussians)
- Batch size scaling (10K to 2M Gaussians)

**Expected Results:**
- Individual filters: 304-733 M Gaussians/sec
- Fused opacity+scale: 416 M Gaussians/sec
- Full filtering (filter_gaussians): 54 M Gaussians/sec mean, 77 M/s best case
- 5x speedup from parallel scatter pattern

## Performance Summary

### Color Processing (100K colors)

| API | Time | Throughput | Speedup |
|-----|------|------------|---------|
| apply() | 0.747 ms | 134 M/s | 1.00x |
| apply_numpy() | 0.514 ms | 195 M/s | 1.45x |
| **apply_numpy_inplace()** | **0.099 ms** | **1,011 M/s** | **7.55x** |

**Batch scaling (apply_numpy_inplace):**
```
N=    1,000:   0.016 ms (   62 M colors/s)
N=   10,000:   0.022 ms (  458 M colors/s)
N=  100,000:   0.100 ms (1,005 M colors/s)
N=1,000,000:   1.141 ms (  877 M colors/s)
```

### 3D Transform (1M Gaussians)

```
Time:       1.479 ms
Throughput: 676 M Gaussians/sec
Max FPS:    676 FPS
```

**Batch scaling:**
```
N=   10,000:   0.013 ms (  769 M G/s)
N=  100,000:   0.090 ms (1,111 M G/s)
N=  500,000:   0.621 ms (  805 M G/s)
N=1,000,000:   1.479 ms (  676 M G/s)
N=2,000,000:   3.154 ms (  634 M G/s)
```

### Filtering (1M Gaussians)

| Operation | Time | Throughput |
|-----------|------|------------|
| Scene bounds (one-time) | 1.4 ms | 733 M/s |
| Recommended scale (one-time) | 6.4 ms | 156 M/s |
| Sphere filter (with fastmath) | 3.3 ms | 304 M/s |
| Cuboid filter (with fastmath) | 2.6 ms | 385 M/s |
| Opacity filter (with fastmath) | 2.3 ms | 444 M/s |
| Scale filter (with fastmath) | 2.5 ms | 409 M/s |
| Opacity + Scale (fused kernel) | 2.4 ms | 416 M/s |
| Full filtering (sphere + fused) | 4.3 ms | 234 M/s |
| Scatter kernel (parallel) | 4.0 ms | 246 M/s |
| **filter_gaussians (mean)** | **18.6 ms** | **54 M/s** |
| **filter_gaussians (best)** | **13.0 ms** | **77 M/s** |

### 4. Mask Combination (`benchmark_mask_strategies.py`)

Tests multi-layer boolean mask combination strategies for complex filtering scenarios.

**Features:**
- Numba JIT compilation with parallel execution
- Adaptive strategy: NumPy for 1 layer, Numba for 2+ layers
- Short-circuit evaluation for early exit
- 6 combination strategies compared: numpy.all(), numpy.reduce(), bitwise.reduce(), manual_loop, numba_parallel, numba_serial

**Tests:**
- Combination method comparison (6 strategies)
- Layer scaling (1, 2, 5, 10, 20 layers)
- Data size scaling (10K to 1M Gaussians)
- Selectivity impact (10% to 90% pass rate)
- Full pipeline breakdown (combine + filter)

**Expected Results:**
- Numba parallel: 0.026ms for 100K Gaussians, 5 layers (55x faster than numpy.all())
- Layer scaling: 75-122x speedup for 2+ layers
- Data size scaling: 34-74x speedup across all sizes
- Full pipeline: Mask combination only 3.8% of total time (negligible overhead)

**Benchmark Results (100K Gaussians, 5 layers):**
```
numpy.all():       1.447 ms  (72 M/s)
numba_parallel:    0.026 ms  (3,800 M/s) - 55x faster!

Layer Scaling:
1 layer:   numpy 0.006ms, numba 0.017ms (numpy better for 1 layer)
2 layers:  numpy 2.187ms, numba 0.029ms (75x faster)
5 layers:  numpy 2.317ms, numba 0.028ms (82x faster)
10 layers: numpy 2.344ms, numba 0.019ms (122x faster)

Full Pipeline (100K Gaussians, 3 layers):
Total:         0.968 ms
Combine masks: 0.037 ms (3.8%)
Apply filter:  0.799 ms (82.6%)
Overhead:      0.132 ms (13.6%)
```

## Key Optimizations

**Color Processing:**
- Interleaved LUT: 1.73x speedup from cache locality
- Fused pipeline: Phase 1 + Phase 2 in single kernel
- Branchless Phase 2: Eliminates branch misprediction
- Zero-copy API: Eliminates 80% allocation overhead

**3D Transform:**
- Fused kernel: 4-5x faster than separate operations
- Custom matmul: 9x faster than BLAS for small matrices
- Single parallel loop: Better memory locality
- Pre-allocated buffers: Zero allocation overhead

**Filtering:**
- Numba JIT compilation: ~5x speedup from parallel scatter pattern
- Fused opacity+scale kernel: 1.95x speedup (combines two filters)
- fastmath optimization: 5-10% speedup on all kernels
- Parallel scatter pattern: Lock-free parallel writes via prefix sum
- Parallel execution: prange for multi-core utilization
- Optimized bounds calculation: Single-pass min/max

**Mask Combination:**
- Adaptive strategy: NumPy for 1 layer (lower overhead), Numba for 2+ layers
- Numba parallel JIT: 55-122x speedup for multi-layer masks
- Short-circuit evaluation: Early exit when all values false
- Negligible overhead: Only 3.8% of full filtering pipeline time

## System Requirements

- Python 3.10+
- NumPy 1.24+
- Numba 0.59+ (required)
- Multi-core CPU recommended
- 8GB+ RAM for large batches

## Archived Files

Historical development files (analysis scripts, old iterations, debug scripts) are in `archive/` directory. See `archive/README.md` for details.

**Production use**: Only use benchmarks in this directory, not archived files.

## Documentation

See project root for detailed documentation:
- `README.md`: Main project documentation
- `OPTIMIZATION_COMPLETE_SUMMARY.md`: Complete optimization history
- `AUDIT_FIXES_SUMMARY.md`: Bug fixes and validation
- `.github/WORKFLOWS.md`: CI/CD pipeline documentation

## Contributing

To add a new benchmark:

1. Create `benchmark_<feature>.py` in this directory
2. Follow existing patterns:
   - Warmup iterations (20+)
   - Multiple batch sizes
   - Clear formatted output
   - Pre-allocated buffers when applicable
3. Add to `run_all_benchmarks.py`
4. Update this README

## License

MIT License - see parent directory LICENSE file.
