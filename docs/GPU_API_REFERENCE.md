# GPU API Reference

This document provides a complete API reference for gsmod's GPU-accelerated operations.

## Overview

gsmod provides GPU-accelerated operations through PyTorch, achieving up to **183x speedup** and **1.09 billion Gaussians/sec** throughput on NVIDIA RTX 3090 Ti.

## Installation

GPU support requires PyTorch with CUDA:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install gsmod
```

## Quick Start

### Simplified API (Recommended)

```python
from gsmod.torch import GSTensorPro
from gsmod import ColorValues, FilterValues, TransformValues
from gsmod import CINEMATIC, STRICT_FILTER, DOUBLE_SIZE

# Load data directly to GPU
data = GSTensorPro.from_ply("scene.ply", device="cuda")

# Apply GPU-accelerated operations with method chaining
(data
    .filter(FilterValues(min_opacity=0.1, sphere_radius=0.8))
    .transform(TransformValues.from_translation(1, 0, 0))
    .color(ColorValues(brightness=1.2, saturation=1.3))
)

# Use presets
data.color(CINEMATIC)
data.filter(STRICT_FILTER)

# Save result
data.to_ply("output.ply")
```

### Legacy Pipeline API

```python
from gsmod.torch import GSTensorPro, PipelineGPU
import gsply

# Load data and convert to GPU
gsdata = gsply.plyread("scene.ply")
gstensor = GSTensorPro.from_gsdata(gsdata, device="cuda")

# Apply GPU-accelerated pipeline
pipeline = (
    PipelineGPU()
    .within_sphere(radius=0.8)
    .translate([1, 0, 0])
    .brightness(1.2)
    .saturation(1.3)
)
result = pipeline(gstensor, inplace=True)

# Convert back to CPU and save
output = result.to_gsdata()
gsply.plywrite("output.ply", output)
```

## GSTensorPro Class

GPU tensor wrapper for Gaussian Splatting data with format-aware operations.

### Constructor

```python
GSTensorPro(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor | None = None,
    masks: dict | None = None,
    _base: torch.Tensor | None = None
)
```

### Class Methods

```python
# Create from PLY file (recommended)
gstensor = GSTensorPro.from_ply(path: str, device: str = "cuda")

# Create from GSData
gstensor = GSTensorPro.from_gsdata(data: GSData, device: str = "cuda")

# Convert back to GSData
gsdata = gstensor.to_gsdata()

# Save to PLY file
gstensor.to_ply(path: str)

# Clone
clone = gstensor.clone()
```

### Simplified API Methods (Recommended)

These methods accept config value objects and return self for method chaining.

```python
from gsmod import ColorValues, FilterValues, TransformValues

# Color operations
gstensor.color(values: ColorValues, inplace: bool = True) -> GSTensorPro

# Filter operations
gstensor.filter(values: FilterValues, inplace: bool = True) -> GSTensorPro

# Transform operations
gstensor.transform(values: TransformValues, inplace: bool = True) -> GSTensorPro

# Example usage
gstensor.color(ColorValues(brightness=1.2, saturation=1.3))
gstensor.filter(FilterValues(min_opacity=0.1, sphere_radius=5.0))
gstensor.transform(TransformValues.from_translation(1, 0, 0))

# Method chaining
(gstensor
    .filter(FilterValues(min_opacity=0.1))
    .transform(TransformValues.from_scale(2.0))
    .color(ColorValues(brightness=1.2))
)

# Use presets
from gsmod import CINEMATIC, STRICT_FILTER, DOUBLE_SIZE
gstensor.color(CINEMATIC)
gstensor.filter(STRICT_FILTER)
gstensor.transform(DOUBLE_SIZE)
```

### Format Conversion Methods

```python
# Convert SH to RGB format
gstensor.to_rgb(inplace=True)  # sh0 = sh0 * 0.28209479 + 0.5

# Convert RGB back to SH format
gstensor.to_sh(inplace=True)   # sh0 = (sh0 - 0.5) / 0.28209479
```

### Color Adjustment Methods

All color methods support `inplace` parameter for zero-copy operations.

```python
# Format-agnostic (work on both SH and RGB)
gstensor.adjust_brightness(factor: float, inplace: bool = False)
gstensor.adjust_contrast(factor: float, inplace: bool = False)
gstensor.adjust_gamma(gamma: float, inplace: bool = False)
gstensor.adjust_fade(value: float, inplace: bool = False)  # Black point lift (0-1)

# Require RGB format (auto-converts if needed)
gstensor.adjust_saturation(factor: float, inplace: bool = False)
gstensor.adjust_temperature(temp: float, inplace: bool = False)  # -1 to 1
gstensor.adjust_tint(value: float, inplace: bool = False)  # -1 (green) to 1 (magenta)
gstensor.adjust_vibrance(factor: float, inplace: bool = False)
gstensor.adjust_hue_shift(degrees: float, inplace: bool = False)
```

### Transform Methods

```python
gstensor.translate(offset: list | np.ndarray | torch.Tensor, inplace: bool = False)
gstensor.scale_uniform(factor: float, inplace: bool = False)
gstensor.scale_nonuniform(factors: list | np.ndarray | torch.Tensor, inplace: bool = False)
gstensor.rotate_quaternion(quat: list | np.ndarray | torch.Tensor, inplace: bool = False)
gstensor.rotate_euler(angles: list | np.ndarray | torch.Tensor, order: str = "xyz", inplace: bool = False)
gstensor.rotate_axis_angle(axis: list | np.ndarray | torch.Tensor, angle: float, inplace: bool = False)
gstensor.transform_matrix(matrix: np.ndarray | torch.Tensor, inplace: bool = False)
gstensor.center_at_origin(inplace: bool = False)
gstensor.normalize_scale(target_size: float = 1.0, inplace: bool = False)
```

### Filter Methods

```python
# Returns boolean mask tensor
mask = gstensor.filter_within_sphere(center=None, radius: float = 1.0, save_mask: str = None)
mask = gstensor.filter_within_box(min_bounds, max_bounds)
mask = gstensor.filter_min_opacity(threshold: float)
mask = gstensor.filter_max_scale(max_scale: float)

# Apply mask to filter data
filtered = gstensor[mask]
```

## ColorGPU Pipeline

GPU-accelerated color adjustment pipeline with format-aware processing.

### Constructor and Methods

```python
pipeline = ColorGPU()

# Chainable operations
pipeline.brightness(factor: float = 1.0)     # Format-agnostic
pipeline.contrast(factor: float = 1.0)       # Format-agnostic
pipeline.gamma(value: float = 1.0)           # Format-agnostic
pipeline.saturation(factor: float = 1.0)     # Requires RGB
pipeline.temperature(temp: float = 0.0)      # Requires RGB (-1 to 1)
pipeline.tint(value: float = 0.0)            # Requires RGB (-1 to 1)
pipeline.vibrance(factor: float = 1.0)       # Requires RGB
pipeline.hue_shift(degrees: float = 0.0)     # Requires RGB
pipeline.shadows(factor: float = 0.0)        # Requires RGB
pipeline.highlights(factor: float = 0.0)     # Requires RGB
pipeline.fade(value: float = 0.0)            # Black point lift (0-1)
pipeline.shadow_tint(hue: float, sat: float) # Split toning shadows
pipeline.highlight_tint(hue: float, sat: float)  # Split toning highlights
pipeline.preset(name: str, strength: float = 1.0)

# Utility methods
pipeline.reset()           # Clear all operations
pipeline.clone()           # Copy pipeline
pipeline.requires_rgb()    # Check if RGB conversion needed
```

### Execution

```python
result = pipeline(gstensor, inplace=True, restore_format=False)
```

Parameters:
- `inplace`: Modify data in-place (True) or create copy (False)
- `restore_format`: Restore original format after processing

### Format-Aware Processing

ColorGPU automatically handles format conversion:
- Analyzes operations to determine if RGB is needed
- Converts to RGB once if any operation requires it (lazy conversion)
- Optionally restores original format after processing

```python
# Check if pipeline needs RGB
if pipeline.requires_rgb():
    print("Will convert to RGB")

# Process with format restoration
result = pipeline(gstensor, restore_format=True)  # Back to original format
```

## TransformGPU Pipeline

GPU-accelerated 3D transformation pipeline.

### Methods

```python
pipeline = TransformGPU()

# Chainable operations
pipeline.translate(offset: list | np.ndarray | torch.Tensor)
pipeline.scale(factor: float)                    # Uniform scale
pipeline.scale_nonuniform(factors: list | np.ndarray | torch.Tensor)
pipeline.rotate_quaternion(quat: list | np.ndarray | torch.Tensor)
pipeline.rotate_euler(angles: list | np.ndarray | torch.Tensor, order: str = "xyz")
pipeline.rotate_axis_angle(axis: list | np.ndarray | torch.Tensor, angle: float)
pipeline.rotate_matrix(matrix: np.ndarray | torch.Tensor)
pipeline.center_at_origin()
pipeline.normalize_scale(target_size: float = 1.0)

# Execution
result = pipeline(gstensor, inplace=True)

# Utility
pipeline.reset()
pipeline.clone()
pipeline.to_matrix()  # Compose to 4x4 matrix
```

## FilterGPU Pipeline

GPU-accelerated filtering pipeline.

### Methods

```python
pipeline = FilterGPU()

# Chainable operations
pipeline.within_sphere(center=None, radius: float = 1.0)  # Absolute radius in world units
pipeline.within_box(min_bounds, max_bounds)
pipeline.outside_sphere(center=None, radius: float = 1.0)
pipeline.min_opacity(threshold: float)
pipeline.max_opacity(threshold: float)
pipeline.min_scale(threshold: float)
pipeline.max_scale(threshold: float)

# Execution
result = pipeline(gstensor, inplace=False)  # Returns filtered GSTensorPro

# Mask computation
mask = pipeline.compute_mask(gstensor, mode="and")  # "and" or "or"

# Utility
pipeline.reset()
pipeline.clone()
```

### Radius Interpretation

Filter radius is interpreted as a factor of scene size (matching CPU behavior):
- `radius=1.0` = half of max scene dimension
- `radius=0.5` = quarter of max scene dimension

## PipelineGPU (Unified)

Combined pipeline for color, transform, and filter operations.

### Methods

```python
pipeline = PipelineGPU()

# All ColorGPU methods
pipeline.brightness(1.2).saturation(1.3)

# All TransformGPU methods
pipeline.translate([1, 0, 0]).rotate_axis_angle([0, 1, 0], 0.5)

# All FilterGPU methods
pipeline.within_sphere(radius=0.8).min_opacity(0.1)

# Execution
result = pipeline(gstensor, inplace=True)
```

## Performance Benchmarks

Tested on NVIDIA GeForce RTX 3090 Ti with 1M Gaussians:

| Operation | CPU (ms) | GPU (ms) | Speedup |
|-----------|----------|----------|---------|
| brightness | 7.16 | 0.23 | 31.2x |
| contrast | 6.19 | 0.42 | 14.9x |
| saturation | 6.58 | 0.36 | 18.5x |
| translate | 17.12 | 0.18 | **93.0x** |
| scale | 22.50 | 0.26 | **86.6x** |
| rotate | 21.45 | 2.21 | 9.7x |
| within_sphere | 105.02 | 0.57 | **183.9x** |
| min_opacity | 47.29 | 2.98 | 15.9x |

**Summary:**
- Average speedup: **43.2x**
- Max speedup: **183.9x**
- GPU throughput: **1.09 billion Gaussians/sec**

## Format Tracking

GSTensorPro tracks the format of sh0 data:

```python
# Check current format
format = gstensor._format.get("sh0")  # "sh" or "rgb"

# Convert between formats
gstensor.to_rgb(inplace=True)   # SH -> RGB
gstensor.to_sh(inplace=True)    # RGB -> SH

# Roundtrip preserves values
original = gstensor.sh0.clone()
gstensor.to_rgb(inplace=True)
gstensor.to_sh(inplace=True)
# gstensor.sh0 == original (within float tolerance)
```

## CPU vs GPU Equivalence

### Equivalent Operations

Transform and filter operations produce identical results:
- Translation: Perfect match
- Scale: Perfect match
- Rotation: Perfect match (< 1e-6 difference)
- Sphere filter: Perfect match
- Opacity filter: Perfect match

### Different by Design

Color operations follow different algorithms:
- **CPU Color**: Operates directly on SH coefficients
- **GPU ColorGPU**: Converts to RGB first (matches universal_4d_viewer)

This difference is intentional - GPU implementation mirrors universal_4d_viewer's approach for accurate color space operations.

## Examples

### Basic Color Grading

```python
from gsmod.torch import GSTensorPro
from gsmod import ColorValues, CINEMATIC

# Load directly to GPU
data = GSTensorPro.from_ply("scene.ply", device="cuda")

# Apply color grading
data.color(ColorValues(
    brightness=1.2,
    contrast=1.1,
    saturation=1.3,
    temperature=0.1
))

# Or use a preset
data.color(CINEMATIC)

# Save
data.to_ply("output.ply")
```

### Scene Transformation

```python
from gsmod.torch import GSTensorPro
from gsmod import TransformValues

# Load directly to GPU
data = GSTensorPro.from_ply("scene.ply", device="cuda")

# Transform scene
data.transform(TransformValues.from_translation(1, 0, 0))
data.transform(TransformValues.from_rotation_euler(0, 45, 0))
data.transform(TransformValues.from_scale(2.0))

# Or compose transforms
combined = (
    TransformValues.from_translation(1, 0, 0)
    + TransformValues.from_rotation_euler(0, 45, 0)
    + TransformValues.from_scale(2.0)
)
data.transform(combined)

data.to_ply("output.ply")
```

### Spatial Filtering

```python
from gsmod.torch import GSTensorPro
from gsmod import FilterValues, STRICT_FILTER

# Load directly to GPU
data = GSTensorPro.from_ply("scene.ply", device="cuda")
original_count = len(data.means)

# Filter scene
data.filter(FilterValues(
    min_opacity=0.1,
    max_scale=2.0,
    sphere_radius=5.0
))

print(f"Filtered: {len(data.means)}/{original_count} Gaussians")

# Or use a preset
data.filter(STRICT_FILTER)
```

### Full Pipeline

```python
from gsmod.torch import GSTensorPro
from gsmod import ColorValues, FilterValues, TransformValues
from gsmod import CINEMATIC

# Load directly to GPU
data = GSTensorPro.from_ply("scene.ply", device="cuda")

# Apply full pipeline with method chaining
(data
    # Filter first
    .filter(FilterValues(
        min_opacity=0.1,
        sphere_radius=5.0
    ))
    # Then transform
    .transform(TransformValues.from_translation(1, 0, 0))
    .transform(TransformValues.from_scale(1.5))
    # Finally color grade
    .color(ColorValues(brightness=1.2, saturation=1.3))
)

# Save
data.to_ply("output.ply")
```

### Legacy Pipeline Example

For advanced use cases, the legacy pipeline classes are still available:

```python
from gsmod.torch import GSTensorPro, PipelineGPU
import gsply

# Load and convert
gsdata = gsply.plyread("scene.ply")
gstensor = GSTensorPro.from_gsdata(gsdata, device="cuda")

# Combined pipeline
pipeline = (
    PipelineGPU()
    .within_sphere(radius=0.8)
    .min_opacity(0.1)
    .translate([1, 0, 0])
    .scale(1.5)
    .brightness(1.2)
    .saturation(1.3)
)
result = pipeline(gstensor, inplace=True)

# Save
gsply.plywrite("output.ply", result.to_gsdata())
```

### Format-Aware Processing

```python
from gsmod.torch import GSTensorPro, ColorGPU

# Load with default SH format
gstensor = GSTensorPro.from_gsdata(data, device="cuda")
print(f"Initial format: {gstensor._format.get('sh0', 'sh')}")

# Pipeline that needs RGB
pipeline = ColorGPU().brightness(1.2).saturation(1.3)

if pipeline.requires_rgb():
    print("Pipeline will convert to RGB")

# Process with format restoration
result = pipeline(gstensor, inplace=False, restore_format=True)
print(f"Final format: {result._format.get('sh0')}")  # Back to 'sh'
```

## Troubleshooting

### CUDA Not Available

```python
import torch
if not torch.cuda.is_available():
    print("CUDA not available. Check PyTorch installation.")
    print(f"PyTorch version: {torch.__version__}")
```

### Out of Memory

```python
# Clear GPU cache
torch.cuda.empty_cache()

# Process in batches for very large scenes
chunk_size = 1000000
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    gstensor = GSTensorPro.from_gsdata(chunk, device="cuda")
    # Process...
```

### Format Mismatch

```python
# Ensure correct format before operations
if gstensor._format.get("sh0") != "rgb":
    gstensor.to_rgb(inplace=True)

# Or let operations auto-convert
pipeline = ColorGPU().saturation(1.3)  # Auto-converts to RGB
```
