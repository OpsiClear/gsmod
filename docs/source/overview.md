# Overview

gsmod (Gaussian Splatting Processing) is a Python library providing ultra-fast processing operations for 3D Gaussian Splatting data.

## Key Features

### Color Processing
- LUT-based color adjustments at 1,091M Gaussians/sec
- Temperature, brightness, contrast, gamma, saturation, vibrance
- Shadows, highlights, fade, hue shift
- Split toning (shadow/highlight tinting)

### Geometric Transforms
- Translate, rotate, scale operations
- Fused 4x4 matrix operations for 2-3x speedup
- Multiple rotation formats: quaternion, euler, axis-angle, matrix
- Numba-optimized quaternion multiplication (200x faster)

### Spatial Filtering
- Sphere and box volume filters
- Opacity and scale threshold filters
- Boolean operators: AND (&), OR (|), NOT (~)

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
from gsmod import GSDataPro, ColorValues, FilterValues, TransformValues

# Load data
data = GSDataPro.from_ply("scene.ply")

# Apply operations
data.filter(FilterValues(min_opacity=0.1, sphere_radius=5.0))
data.transform(TransformValues.from_translation(1, 0, 0))
data.color(ColorValues(brightness=1.2, saturation=1.3))

# Save result
data.to_ply("output.ply")
```

## Architecture

### Dual API Design

gsmod provides two ways to use each operation:

1. **Config Values (Simple)**: `ColorValues`, `FilterValues`, `TransformValues`
   - Declarative, serializable, composable with `+`
   - Use with `GSDataPro.color()`, `.filter()`, `.transform()`

2. **Pipeline Classes (Advanced)**: `Color`, `Transform`
   - Full control over compilation and optimization
   - Method chaining for stacking operations

### Performance Optimizations

- **Separated 1D LUTs**: 10x faster than 3D LUTs for color
- **Fused matrix transforms**: Single matmul vs 3 operations
- **Numba JIT compilation**: 9-200x speedup for numeric ops
- **Zero-copy processing**: In-place operations where possible
