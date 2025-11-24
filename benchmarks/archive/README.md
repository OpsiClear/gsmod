# Archived Benchmark Files

This directory contains historical development artifacts from the gsmod optimization process. These files are preserved for reference but are not actively maintained.

## Archive Date
2025-11-13

## File Categories

### Analysis Scripts
Scripts used to analyze performance bottlenecks during development:
- `analyze_color_bottlenecks.py` - Color processing bottleneck analysis
- `analyze_cpu_bottlenecks.py` - CPU-specific bottleneck analysis
- `analyze_further_optimizations.py` - Additional optimization opportunities
- `analyze_ultra_optimization.py` - Ultra-optimization analysis
- `color_optimization_analysis.py` - Detailed color optimization study

### Old Benchmark Iterations
Previous versions of benchmarks superseded by current production benchmarks:
- `benchmark_color.py` - Old color benchmark (used PyTorch, superseded by ../benchmark_color.py)
- `benchmark_color_fused.py` - Fused color kernel benchmark iteration
- `benchmark_comprehensive.py` - Comprehensive benchmark (unclear scope)
- `benchmark_cpu_ultra_fused.py` - CPU ultra-fused kernel benchmark
- `benchmark_final_comparison.py` - Final comparison before release
- `benchmark_final_optimizations.py` - Final optimization benchmark
- `benchmark_pure_numpy_api.py` - Pure NumPy API benchmark (superseded)
- `benchmark_ultra_optimizations.py` - Ultra-optimization benchmark iteration

### Debug Scripts
Scripts used to debug specific issues during development:
- `debug_color_difference.py` - Debug color processing differences
- `debug_fused_performance.py` - Debug fused kernel performance
- `debug_specific_pixels.py` - Debug specific pixel processing issues

### Profile Scripts
Profiling scripts used to identify hotspots:
- `profile_real_bottleneck.py` - Profile real bottlenecks

### Verification Scripts
Scripts used to verify correctness of optimizations:
- `verify_color_fused_correctness.py` - Verify fused color kernel correctness
- `verify_ultra_optimizations_correctness.py` - Verify ultra-optimized kernel correctness

## Current Production Benchmarks

The active, maintained benchmarks are in the parent directory:
- `../benchmark_color.py` - Color processing benchmark (NumPy/Numba)
- `../benchmark_transform.py` - 3D transform benchmark (fused Numba kernel)
- `../run_all_benchmarks.py` - Benchmark runner

## Why Archived?

These files were part of an iterative optimization process where multiple approaches were tested and compared. The final, optimal implementations are now in production. These files are kept for:

1. **Historical reference**: Understanding the optimization journey
2. **Learning resource**: Seeing what approaches were tried and why
3. **Comparison**: Verifying that current implementations are best

## Should I Use These?

**No.** Use the production benchmarks in the parent directory instead.

If you need to understand the optimization history or reproduce specific experiments, these files may be useful for reference.
