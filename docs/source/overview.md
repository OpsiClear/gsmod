# Overview

gsmod (Gaussian Splatting Processing) is a Python library providing ultra-fast processing operations for 3D Gaussian Splatting data.

## Key Features

### Color Processing
- LUT-based color adjustments at 1,091M Gaussians/sec
- Temperature, brightness, contrast, gamma, saturation, vibrance
- Shadows, highlights, fade, hue shift
- Split toning (shadow/highlight tinting)

### Opacity Adjustment
- Format-aware opacity scaling (PLY logit and linear formats)
- Fade (reduce opacity) and boost (increase opacity) operations
- Multiplicative composition for combining adjustments
- Factory methods: OpacityValues.fade(), OpacityValues.boost()

### Geometric Transforms
- Translate, rotate, scale operations
- Fused 4x4 matrix operations for 2-3x speedup
- Multiple rotation formats: quaternion, euler, axis-angle, matrix
- Numba-optimized quaternion multiplication (200x faster)
- Shared rotation utilities for CPU/GPU code reuse

### Spatial Filtering
- Sphere and box volume filters
- Opacity and scale threshold filters
- Boolean operators: AND (&), OR (|), NOT (~)

### Unified Processing
- GaussianProcessor auto-dispatches between CPU and GPU
- Single API for GSData/GSDataPro (CPU) and GSTensor/GSTensorPro (GPU)
- Methods: color(), transform(), filter(), opacity()
- Batch processing with process() method

### GPU Acceleration
- PyTorch backend with up to 183x speedup
- 1.09B Gaussians/sec on RTX 3090 Ti
- Same API as CPU operations

## Installation

```bash
pip install gsmod
```

For GPU support:
```bash
pip install gsmod torch
```

## Quick Start

```python
from gsmod import GSDataPro, ColorValues, FilterValues, TransformValues, OpacityValues

# Load data
data = GSDataPro.from_ply("scene.ply")

# Apply operations
data.filter(FilterValues(min_opacity=0.1, sphere_radius=5.0))
data.transform(TransformValues.from_translation(1, 0, 0))
data.color(ColorValues(brightness=1.2, saturation=1.3))
data.opacity(OpacityValues.fade(0.8))  # Fade to 80%

# Save result
data.to_ply("output.ply")
```

### Unified Processing (Auto-dispatch CPU/GPU)

```python
from gsmod import GaussianProcessor, ColorValues, OpacityValues
from gsmod import GSDataPro
from gsmod.torch import GSTensorPro

# Single processor works with both CPU and GPU
processor = GaussianProcessor()

# CPU processing
cpu_data = GSDataPro.from_ply("scene.ply")
cpu_data = processor.color(cpu_data, ColorValues(brightness=1.2))
cpu_data = processor.opacity(cpu_data, OpacityValues.fade(0.8))

# GPU processing (auto-detected)
gpu_data = GSTensorPro.from_ply("scene.ply", device="cuda")
gpu_data = processor.color(gpu_data, ColorValues(brightness=1.2))
gpu_data = processor.opacity(gpu_data, OpacityValues.fade(0.8))
```

## Architecture

### Dual API Design

gsmod provides two ways to use each operation:

1. **Config Values (Simple)**: `ColorValues`, `FilterValues`, `TransformValues`, `OpacityValues`
   - Declarative, serializable, composable with `+`
   - Use with `GSDataPro.color()`, `.filter()`, `.transform()`, `.opacity()`

2. **Pipeline Classes (Advanced)**: `Color`, `Transform`, `Filter`
   - Full control over compilation and optimization
   - Method chaining for stacking operations

3. **Unified Processing**: `GaussianProcessor`
   - Auto-dispatches between CPU (NumPy/Numba) and GPU (PyTorch)
   - Single interface for all processing operations
   - Batch processing with process() method

### Performance Optimizations

- **Separated 1D LUTs**: 10x faster than 3D LUTs for color
- **Fused matrix transforms**: Single matmul vs 3 operations
- **Numba JIT compilation**: 9-200x speedup for numeric ops
- **Zero-copy processing**: In-place operations where possible
