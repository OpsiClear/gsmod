<div align="center">

# gsmod

### Processing Tools for 3D Gaussian Splatting

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

[Features](#features) | [Installation](#installation) | [Quick Start](#quick-start) | [Performance](#performance) | [Documentation](#documentation)

</div>

---

## Overview

**gsmod** provides processing operations for 3D Gaussian Splatting data: color grading, 3D transforms, spatial filtering, opacity adjustment, histogram-based learning, and learnable modules for auto-adjustment. Works with both CPU (NumPy) and GPU (PyTorch) backends with unified processing interface.

**Included Features:**
- **Color Grading**: 15 adjustments (brightness, contrast, saturation, temperature, tint, gamma, shadows, highlights, fade, hue shift, split toning)
  - **39 built-in presets**: Film stocks, seasonal, time of day, artistic styles
- **Opacity Adjustment**: Format-aware opacity scaling (fade/boost) for both PLY (logit) and linear formats
  - **12 opacity presets**: Fade, boost, and special effects
- **Histogram Learning**: Learn color adjustments to match target distributions
  - Gradient-based learning with `HistogramResult.learn_from()`
  - Rule-based suggestions with `to_color_values()`
- **3D Transforms**: translate, rotate, scale with quaternion/euler/axis-angle support
- **Spatial Filtering**: sphere, box, ellipsoid, frustum volumes + opacity/scale thresholds
  - Include/Exclude modes with `invert` parameter
- **Unified Processing**: GaussianProcessor auto-dispatches between CPU and GPU backends
- **Learnable Modules**: PyTorch nn.Modules for gradient-based color/transform/filter/opacity optimization
- **Composable Pipelines**: Method chaining, built-in presets, JSON serialization
- **Scene Composition**: Concatenate, merge, deduplicate, split scenes
- **Format-Aware**: Automatic SH/RGB format tracking and conversion

---

## Features

- **Color Grading**: 15 adjustments for comprehensive color control
  - temperature, tint, brightness, contrast, gamma, saturation, vibrance
  - shadows, highlights, fade, hue_shift
  - Split toning: shadow_tint, highlight_tint
  - LUT-based processing with zero-copy API

- **Opacity Adjustment**: Format-aware opacity scaling
  - Works with both PLY (logit) and linear [0,1] opacity formats
  - Fade (reduce opacity) and boost (increase opacity) operations
  - Multiplicative composition for combining adjustments
  - Factory methods: `OpacityValues.fade()`, `OpacityValues.boost()`

- **Unified Processing**: Single API for CPU and GPU
  - `GaussianProcessor` auto-dispatches based on input type
  - Works with GSData/GSDataPro (CPU) and GSTensor/GSTensorPro (GPU)
  - Methods: `color()`, `transform()`, `filter()`, `opacity()`
  - Batch processing with `process()` method

- **3D Transforms**: Geometric operations on Gaussian positions and orientations
  - translate, rotate, scale, combined transforms
  - Rotation formats: quaternion, matrix, axis_angle, euler
  - Quaternion utilities for orientation manipulation
  - Shared rotation utilities for CPU/GPU code reuse

- **Spatial Filtering**: Volume and property-based selection
  - Volume filters: sphere, box, ellipsoid, frustum
  - Property filters: opacity and scale thresholds
  - Composable with AND/OR/NOT operators
  - Multi-layer mask management (FilterMasks API)

- **Histogram Learning**: Match target color distributions
  - `HistogramResult.learn_from()`: Gradient-based learning API
  - `to_color_values()`: Rule-based adjustments (vibrant, dramatic, bright, dark, neutral)
  - GPU acceleration support
  - Works with CPU or CUDA tensors

- **Learnable Modules**: PyTorch nn.Modules for training
  - `LearnableColor`: Gradient-based color parameter optimization
  - `LearnableOpacity`: Learnable opacity adjustment with format awareness
  - `LearnableTransform`: Learnable geometric transformations
  - `LearnableFilter`: Differentiable filtering parameters
  - Convert to/from config values for inference

- **Scene Composition**: Multi-scene operations
  - Concatenate, merge, deduplicate scenes
  - Compose with transforms (position scenes in space)
  - Split by spatial region

- **Parameterized Templates**: Reusable pipelines with named parameters
  - Efficient for parameter sweeps and animation
  - Auto-cached for repeated use

- **Pre-Activation Stage** (via gsply): Prepare log-domain GSData
  - `gsply.apply_pre_activations()` for scales, opacities, quaternions
  - Works in-place with automatic dtype/contiguity fixes

- **Built-in Presets**: Ready-to-use configurations
  - Color (39 presets): cinematic, warm, cool, neutral, vibrant, muted, dramatic, vintage
    - Film Stock: kodak_portra, fuji_velvia, kodak_ektachrome, ilford_hp5, cinestill_800t
    - Seasonal: spring_fresh, summer_bright, autumn_warm, winter_cold
    - Time of Day: sunrise, midday_sun, golden_hour, sunset, blue_hour, moonlight, overcast
    - Artistic: high_key, low_key, teal_orange, bleach_bypass, cross_process, faded_print, sepia_tone
    - Technical: lift_shadows, compress_highlights, increase_contrast, desaturate_mild, enhance_colors
  - Opacity (12 presets): fade_subtle, fade_mild, fade_moderate, fade_heavy, boost_mild, boost_moderate, boost_strong, ghost_effect, translucent, semi_transparent
  - Filter: strict, quality, cleanup
  - Transform: double_size, half_size, flip_x/y/z

- **GPU Support**: All operations available on GPU via PyTorch
- **Pure Python**: NumPy + Numba for CPU, PyTorch for GPU
- **Type-safe**: Full type hints with Python 3.12+ syntax

---

## Installation

### From PyPI

```bash
pip install gsmod
```

### From Source

```bash
git clone https://github.com/OpsiClear/gsmod.git
cd gsmod
pip install -e .
```

**Requirements:** Python >= 3.10, NumPy >= 1.24.0, Numba >= 0.59.0

**GPU Support (Optional):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Quick Start

### Simplified API (Recommended)

```python
from gsmod import GSDataPro, ColorValues, FilterValues, TransformValues, OpacityValues
from gsmod import CINEMATIC, STRICT_FILTER, DOUBLE_SIZE

# Load Gaussian splatting data
data = GSDataPro.from_ply("scene.ply")

# Apply operations with fluent chaining
data.color(ColorValues(brightness=1.2, saturation=1.3))
data.opacity(OpacityValues(scale=0.8))  # Fade to 80%
data.filter(FilterValues(min_opacity=0.1, sphere_radius=0.8))
data.transform(TransformValues.from_scale(2.0))

# Or use presets
data.color(CINEMATIC)
data.filter(STRICT_FILTER)
data.transform(DOUBLE_SIZE)

# Save result
data.to_ply("output.ply")
```

### Using Config Values

```python
from gsmod import GSDataPro, ColorValues, FilterValues, TransformValues, OpacityValues

data = GSDataPro.from_ply("scene.ply")

# Color adjustments
data.color(ColorValues(
    brightness=1.2,
    contrast=1.1,
    saturation=1.3,
    temperature=0.6,  # 0=neutral, positive=warm, negative=cool
    gamma=1.05,
    shadows=0.1,
    highlights=-0.05
))

# Opacity adjustments (format-aware: works with PLY logit and linear)
data.opacity(OpacityValues(scale=0.5))  # Fade to 50%
data.opacity(OpacityValues.fade(0.7))   # Factory method: fade to 70%
data.opacity(OpacityValues.boost(1.5))  # Boost opacity (move toward 1.0)

# Spatial filtering
# Include mode (default): keep only what matches
data.filter(FilterValues(
    min_opacity=0.1,
    max_scale=2.5,
    sphere_radius=0.8,
    sphere_center=(0, 0, 0)
))

# Exclude mode: remove what matches, keep everything outside
data.filter(FilterValues(
    sphere_radius=5.0,
    invert=True  # Keep points OUTSIDE the sphere
))

# Chain multiple filters with different modes
data.filter(FilterValues(sphere_radius=3.0, invert=False))  # Keep inside outer sphere
data.filter(FilterValues(sphere_radius=1.5, invert=True))   # Remove inside inner sphere
# Result: Hollow shell between r=1.5 and r=3.0

# Complex filtering with method chaining
(data
    .filter(FilterValues(sphere_radius=5.0, invert=False))  # Include: inside sphere
    .filter(FilterValues(min_opacity=0.5, invert=False))     # Include: high opacity
    .filter(FilterValues(box_min=(-1,-1,-1), box_max=(1,1,1), invert=True))  # Exclude: box
)

# 3D transforms
data.transform(TransformValues.from_translation(1.0, 0.0, 0.0))
data.transform(TransformValues.from_rotation_euler(0, 45, 0))  # degrees
data.transform(TransformValues.from_scale(1.5))

data.to_ply("output.ply")
```

### Composing Values

```python
from gsmod import GSDataPro, ColorValues, WARM, CINEMATIC

data = GSDataPro.from_ply("scene.ply")

# Compose presets with custom values using +
warm_bright = WARM + ColorValues(brightness=1.2)
data.color(warm_bright)

# Compose multiple presets
cinematic_warm = CINEMATIC + WARM
data.color(cinematic_warm)
```

### Unified Processing (Auto-dispatch CPU/GPU)

```python
from gsmod import GaussianProcessor, ColorValues, TransformValues, OpacityValues
from gsmod import GSDataPro
from gsmod.torch import GSTensorPro

# Create processor (works with both CPU and GPU data)
processor = GaussianProcessor()

# CPU processing
cpu_data = GSDataPro.from_ply("scene.ply")
cpu_data = processor.color(cpu_data, ColorValues(brightness=1.2))
cpu_data = processor.opacity(cpu_data, OpacityValues.fade(0.8))

# GPU processing (automatically detected)
gpu_data = GSTensorPro.from_ply("scene.ply", device="cuda")
gpu_data = processor.color(gpu_data, ColorValues(brightness=1.2))
gpu_data = processor.opacity(gpu_data, OpacityValues.fade(0.8))

# Batch processing with process() method
result = processor.process(
    cpu_data,
    color=ColorValues(brightness=1.2),
    transform=TransformValues.from_scale(2.0),
    opacity_values=OpacityValues.fade(0.8)
)
```

### GPU Pipeline

```python
from gsmod.torch import GSTensorPro
from gsmod import ColorValues, FilterValues, TransformValues, OpacityValues

# Load data to GPU
data = GSTensorPro.from_ply("scene.ply", device="cuda")

# Apply operations
data.filter(FilterValues(min_opacity=0.1, sphere_radius=0.8))
data.transform(TransformValues.from_translation(1, 0, 0))
data.color(ColorValues(brightness=1.2, saturation=1.3))
data.opacity(OpacityValues.fade(0.7))

# Save result
data.to_ply("output.ply")
```

### Method Chaining

```python
from gsmod import GSDataPro, ColorValues, FilterValues, TransformValues

data = GSDataPro.from_ply("scene.ply")

# Chain operations (all return self for fluent API)
(data
    .filter(FilterValues(min_opacity=0.1))
    .transform(TransformValues.from_scale(2.0))
    .color(ColorValues(brightness=1.2))
)

data.to_ply("output.ply")
```

### Unified Pipeline Class

The Pipeline class provides a fluent interface for building complex processing workflows. It matches the PipelineGPU interface for consistency:

```python
from gsmod import Pipeline, GSDataPro

data = GSDataPro.from_ply("scene.ply")

# Build pipeline with method chaining
pipe = (Pipeline()
    .brightness(1.2)
    .saturation(1.3)
    .contrast(1.1)
    .translate([1, 0, 0])
    .scale(2.0)
    .min_opacity(0.1))

# Apply pipeline to data
result = pipe(data, inplace=True)

# Reuse pipeline on different data
data2 = GSDataPro.from_ply("scene2.ply")
result2 = pipe(data2, inplace=False)
```

The Pipeline class supports all color, transform, and filter operations as methods. Operations are accumulated and executed in order when the pipeline is called.

### Copy vs Inplace

```python
from gsmod import GSDataPro, ColorValues

data = GSDataPro.from_ply("scene.ply")

# Inplace (default) - modifies data directly
data.color(ColorValues(brightness=1.2))

# Copy - returns new instance, original unchanged
result = data.color(ColorValues(brightness=1.2), inplace=False)
```

### GSData Pre-Activation (Optional)

When your training or authoring pipeline stores Gaussians in log-space (log scales, logit opacities, non-normalized quats), fuse the conversion into a single CPU pass before uploading to the renderer:

```python
import gsply

data = gsply.plyread("scene_raw_logits.ply")

gsply.apply_pre_activations(
    data,
    min_scale=1e-4,
    max_scale=100.0,
    min_quat_norm=1e-8,
    inplace=True,
)
```

`gsply.apply_pre_activations` exponentiates + clamps the scales, runs a numerically stable sigmoid on logit opacities, and normalizes quaternions—processing ~1M Gaussians in ≈1.3 ms (≈750M/sec). The helper automatically ensures float32 and contiguous buffers, so it pairs nicely with the zero-copy arrays returned by `gsply.plyread`.

### Histogram-Based Learning

Learn color adjustments to match a target histogram distribution using gradient descent:

```python
from gsmod import GSDataPro, ColorValues
import torch

# Step 1: Get target histogram from reference scene
reference = GSDataPro.from_ply("reference.ply")
target_hist = reference.histogram_colors()

# Step 2: Load source data
source = GSDataPro.from_ply("source.ply")
source_colors = torch.tensor(source.sh0)  # Works with CPU or CUDA

# Step 3: Learn color adjustment to match target (CONVENIENT API!)
learned = target_hist.learn_from(
    source_colors,
    params=["brightness", "contrast", "saturation", "gamma"],
    n_epochs=100,
    lr=0.02,
    verbose=True  # Print progress
)

# Step 4: Apply learned values
source.color(learned)
source.to_ply("matched.ply")
```

**Rule-based adjustments** (quick heuristics without learning):

```python
# Analyze histogram and suggest adjustments
result = data.histogram_colors()

adjustment = result.to_color_values("vibrant")   # Boost saturation, contrast
# adjustment = result.to_color_values("dramatic")  # Strong contrast, dark shadows
# adjustment = result.to_color_values("bright")    # Shift distribution higher
# adjustment = result.to_color_values("dark")      # Shift distribution lower
# adjustment = result.to_color_values("neutral")   # Flatten toward uniform

data.color(adjustment)
```

**GPU acceleration** for histogram learning:

```python
from gsmod.torch import GSTensorPro

# Load to GPU
source_gpu = GSTensorPro.from_ply("source.ply", device="cuda")

# Learn on GPU (faster)
learned = target_hist.learn_from(
    source_gpu.sh0,  # Already on CUDA
    params=["brightness", "contrast"],
    n_epochs=200,
    lr=0.05
)

source_gpu.color(learned)
```

### Using Presets

```python
from gsmod import GSDataPro, OpacityValues
from gsmod import (
    # Color presets - Basic
    WARM, COOL, NEUTRAL, CINEMATIC, VIBRANT, MUTED, DRAMATIC, VINTAGE, GOLDEN_HOUR, MOONLIGHT,
    # Color presets - Film Stock
    KODAK_PORTRA, FUJI_VELVIA, KODAK_EKTACHROME, ILFORD_HP5, CINESTILL_800T,
    # Color presets - Seasonal
    SPRING_FRESH, SUMMER_BRIGHT, AUTUMN_WARM, WINTER_COLD,
    # Color presets - Time of Day
    SUNRISE, MIDDAY_SUN, SUNSET, BLUE_HOUR, OVERCAST,
    # Color presets - Artistic
    HIGH_KEY, LOW_KEY, TEAL_ORANGE, BLEACH_BYPASS, CROSS_PROCESS, FADED_PRINT, SEPIA_TONE,
    # Color presets - Technical
    LIFT_SHADOWS, COMPRESS_HIGHLIGHTS, INCREASE_CONTRAST, DESATURATE_MILD, ENHANCE_COLORS,
    # Opacity presets
    FADE_MILD, FADE_MODERATE, BOOST_MILD, BOOST_MODERATE, GHOST_EFFECT, TRANSLUCENT,
    # Filter presets
    STRICT_FILTER, QUALITY_FILTER, CLEANUP_FILTER,
    # Transform presets
    DOUBLE_SIZE, HALF_SIZE, FLIP_X, FLIP_Y, FLIP_Z,
)

data = GSDataPro.from_ply("scene.ply")

# Apply built-in color grading presets
data.color(CINEMATIC)         # Classic cinematic look
data.color(WARM)              # Warm tones
data.color(KODAK_PORTRA)      # Film stock emulation
data.color(GOLDEN_HOUR)       # Golden hour lighting
data.color(TEAL_ORANGE)       # Popular teal/orange look

# Opacity presets
data.opacity(FADE_MODERATE)   # Fade to 70%
data.opacity(GHOST_EFFECT)    # Semi-transparent 20%
data.opacity(BOOST_MILD)      # Boost opacity by 1.2x

# Filter presets
data.filter(STRICT_FILTER)    # min_opacity=0.5, max_scale=1.0, sphere_radius=10.0
data.filter(QUALITY_FILTER)   # min_opacity=0.3, max_scale=2.0

# Transform presets
data.transform(DOUBLE_SIZE)   # scale 2x
data.transform(FLIP_X)        # flip around X axis
```

### Loading from Dict/JSON

```python
from gsmod import GSDataPro
from gsmod.config.presets import (
    color_from_dict, filter_from_dict, transform_from_dict,
    load_color_json, load_filter_json, load_transform_json,
    get_color_preset, get_filter_preset, get_transform_preset, get_opacity_preset,
)

data = GSDataPro.from_ply("scene.ply")

# Load from dictionary
color_dict = {"brightness": 1.2, "saturation": 1.3}
data.color(color_from_dict(color_dict))

# Load from JSON file
data.color(load_color_json("my_color_preset.json"))

# Get preset by name
data.color(get_color_preset("cinematic"))       # Or "kodak_portra", "teal_orange", etc.
data.opacity(get_opacity_preset("fade_moderate"))  # Or "ghost_effect", "boost_mild", etc.
data.filter(get_filter_preset("strict"))
data.transform(get_transform_preset("double_size"))
```

---

## Complete API Guide

### GSDataPro API

#### Basic Usage

```python
from gsmod import GSDataPro, ColorValues, FilterValues, TransformValues

# Load Gaussian splatting data
data = GSDataPro.from_ply("scene.ply")

# Apply operations
data.color(ColorValues(
    brightness=1.2,
    contrast=1.1,
    saturation=1.3,
    temperature=0.6,
    gamma=1.05,
    shadows=0.1,
    highlights=-0.05
))

data.filter(FilterValues(
    min_opacity=0.1,
    max_scale=2.5,
    sphere_radius=0.8
))

data.transform(TransformValues.from_scale(2.0))
data.transform(TransformValues.from_translation(1, 0, 0))
data.transform(TransformValues.from_rotation_euler(0, 45, 0))

# Save result
data.to_ply("output.ply")

# Copy instead of inplace
result = data.color(ColorValues(brightness=1.2), inplace=False)
```

#### ColorValues Parameters

```python
from gsmod import ColorValues

ColorValues(
    # Basic adjustments (multipliers, 1.0=no change)
    brightness=1.0,    # Multiplier (1.0=no change)
    contrast=1.0,      # Multiplier (1.0=no change)
    gamma=1.0,         # Gamma correction (1.0=linear)
    saturation=1.0,    # Multiplier (0=grayscale, 1.0=no change)
    vibrance=1.0,      # Selective saturation (1.0=no change)

    # White balance (additive, 0.0=no change)
    temperature=0.0,   # -1=cool/blue, 0=neutral, 1=warm/orange
    tint=0.0,          # -1=green, 0=neutral, 1=magenta

    # Tone adjustments (additive, 0.0=no change)
    shadows=0.0,       # Shadow lift/crush (-1 to 1)
    highlights=0.0,    # Highlight lift/crush (-1 to 1)
    fade=0.0,          # Black point lift for film look (0 to 1)

    # Color rotation
    hue_shift=0.0,     # Hue rotation in degrees (-180 to 180)

    # Split toning (shadow/highlight tinting)
    shadow_tint_hue=0.0,     # Shadow tint hue (-180 to 180 degrees)
    shadow_tint_sat=0.0,     # Shadow tint saturation (0 to 1)
    highlight_tint_hue=0.0,  # Highlight tint hue (-180 to 180 degrees)
    highlight_tint_sat=0.0,  # Highlight tint saturation (0 to 1)
)

# Factory methods
ColorValues.from_k(4500)  # Create from color temperature in Kelvin
```

#### OpacityValues Parameters

```python
from gsmod import OpacityValues

OpacityValues(
    scale=1.0  # Opacity scale factor
               # 0.0-1.0: Multiplicative fade (reduce opacity)
               # >1.0: Boost opacity (additive in remaining headroom)
)

# Factory methods
OpacityValues.fade(0.7)    # Fade to 70% opacity
OpacityValues.boost(1.5)   # Boost opacity by 1.5x
```

#### FilterValues Parameters

```python
from gsmod import FilterValues

FilterValues(
    min_opacity=0.0,         # Minimum opacity threshold
    max_opacity=1.0,         # Maximum opacity threshold
    min_scale=0.0,           # Minimum scale threshold
    max_scale=float('inf'),  # Maximum scale threshold
    sphere_radius=float('inf'),  # Sphere filter radius
    sphere_center=(0, 0, 0), # Sphere filter center
    box_min=None,            # Box filter min corner (x, y, z)
    box_max=None,            # Box filter max corner (x, y, z)
    invert=False,            # Include (False) or Exclude (True) mode
)

# Include mode (invert=False, default): Keep only what matches
# Exclude mode (invert=True): Remove what matches, keep everything outside
```

#### TransformValues Parameters

```python
from gsmod import TransformValues

TransformValues(
    scale=1.0,                        # Uniform scale
    rotation=(1.0, 0.0, 0.0, 0.0),    # Quaternion (w, x, y, z)
    translation=(0.0, 0.0, 0.0),      # Translation vector
)

# Factory methods
TransformValues.from_scale(2.0)
TransformValues.from_translation(1.0, 0.0, 0.0)
TransformValues.from_rotation_euler(roll, pitch, yaw)  # degrees
TransformValues.from_euler_rad(roll, pitch, yaw)       # radians
TransformValues.from_rotation_axis_angle(axis, angle)  # degrees
TransformValues.from_axis_angle_rad(axis, angle)       # radians
```

### Transform API

#### Basic Transform Operations

```python
from gsmod import GSDataPro, TransformValues

# Load Gaussian splatting data
data = GSDataPro.from_ply("scene.ply")

# Individual operations
data.transform(TransformValues.from_translation(1, 0, 0))
data.transform(TransformValues.from_rotation_euler(0, 45, 0))
data.transform(TransformValues.from_scale(2.0))

# Save result
data.to_ply("transformed.ply")
```

#### Composed Transforms

```python
from gsmod import GSDataPro, TransformValues

data = GSDataPro.from_ply("scene.ply")

# Compose multiple transforms (matrix multiplication)
combined = (
    TransformValues.from_translation(1.0, 0.0, 0.0)
    + TransformValues.from_rotation_euler(0, 45, 0)
    + TransformValues.from_scale(2.0)
)

data.transform(combined)
data.to_ply("output.ply")
```

#### Rotation Format Examples

```python
from gsmod import TransformValues

# From euler angles (degrees)
TransformValues.from_rotation_euler(0, 45, 0)  # pitch 45 degrees

# From euler angles (radians)
TransformValues.from_euler_rad(0, 0.785, 0)

# From axis-angle (degrees)
TransformValues.from_rotation_axis_angle((0, 0, 1), 90)  # 90 degrees around Z

# From axis-angle (radians)
TransformValues.from_axis_angle_rad((0, 0, 1), 1.57)

# Direct quaternion
TransformValues(rotation=(0.9239, 0, 0, 0.3827))  # w, x, y, z
```

### Filtering API

#### Basic Filtering

```python
from gsmod import GSDataPro, FilterValues

# Load data
data = GSDataPro.from_ply("scene.ply")

# Apply filters
data.filter(FilterValues(
    min_opacity=0.1,     # Remove low-opacity Gaussians
    max_scale=2.5,       # Remove large-scale outliers
    sphere_radius=0.8,   # Keep 80% of scene
    sphere_center=(0, 0, 0)
))

data.to_ply("filtered.ply")
```

#### Filtering Options

```python
from gsmod import GSDataPro, FilterValues

data = GSDataPro.from_ply("scene.ply")

# Option 1: Sphere filtering
data.filter(FilterValues(
    sphere_radius=0.8,
    sphere_center=(0, 0, 0)
))

# Option 2: Box filtering
data.filter(FilterValues(
    box_min=(-0.5, -0.5, -0.5),
    box_max=(0.5, 0.5, 0.5)
))

# Option 3: Property filtering only
data.filter(FilterValues(
    min_opacity=0.05,
    max_scale=2.5
))

# Combined filtering
data.filter(FilterValues(
    min_opacity=0.1,
    max_scale=2.5,
    sphere_radius=0.8
))
```

#### Composed Filters

```python
from gsmod import GSDataPro, FilterValues, STRICT_FILTER

data = GSDataPro.from_ply("scene.ply")

# Compose filters (stricter values win)
combined = FilterValues(min_opacity=0.3) + FilterValues(min_opacity=0.5)
# combined.min_opacity == 0.5  (stricter)

# Combine preset with custom
custom = STRICT_FILTER + FilterValues(sphere_radius=5.0)
data.filter(custom)
```

#### Atomic Filter API

The Filter class provides an atomic, composable approach to filtering. Each filter is a single operation that can be combined using Python operators:

```python
from gsmod import Filter

# Create atomic filters using factory methods
sphere = Filter.sphere(center=(0,0,0), radius=5.0)
box = Filter.box(size=(3,3,3), rotation=(0, 0, 0.5))
ellipsoid = Filter.ellipsoid(radii=(2,3,4))
frustum = Filter.frustum(position=(0,0,10), fov=60)
opacity = Filter.min_opacity(0.1)
scale = Filter.max_scale(3.0)

# Combine with Python operators
combined = sphere & opacity & scale      # AND logic
alternative = sphere | box               # OR logic
inverted = ~sphere                       # NOT logic
complex_filter = (sphere & opacity) | box

# Apply to data
filtered = combined(data, inplace=False)

# Get mask only (fast - no data copying)
mask = combined.get_mask(data)
print(f"Keeping {mask.sum()}/{len(mask)} Gaussians")

# Utility methods
print(sphere.summary(data))  # "850/1000 (85.0%)"
count = sphere.count(data)   # 850
```

#### Combining Filters

```python
from gsmod import Filter

# Individual filter masks
sphere_mask = Filter.sphere(radius=5.0).get_mask(data)
opacity_mask = Filter.min_opacity(0.1).get_mask(data)
scale_mask = Filter.max_scale(3.0).get_mask(data)

# Combine masks with boolean logic
combined_and = sphere_mask & opacity_mask & scale_mask  # All must pass
combined_or = sphere_mask | opacity_mask                # Any must pass
inverse = ~sphere_mask                                  # Outside sphere

# Apply combined mask
filtered = data[combined_and]

# Or combine filters directly (more readable)
combined = Filter.sphere(radius=5.0) & Filter.min_opacity(0.1) & Filter.max_scale(3.0)
filtered = combined(data, inplace=False)
```

#### Volume Filters

```python
from gsmod import Filter

# Sphere filter
sphere = Filter.sphere(center=(0,0,0), radius=5.0)

# Box filter with optional rotation (axis-angle in radians)
box = Filter.box(center=(0,0,0), size=(3,3,3), rotation=(0, 0, 0.5))

# Ellipsoid filter with optional rotation
ellipsoid = Filter.ellipsoid(center=(0,0,0), radii=(2,3,4), rotation=(0.1, 0, 0))

# Camera frustum filter
frustum = Filter.frustum(
    position=(0, 0, 10),
    rotation=(0, 0, 0),  # axis-angle
    fov=60,              # degrees
    aspect=1.0,
    near=0.1,
    far=100.0
)
```

#### Multi-Layer Mask Management (FilterMasks)

For complex filtering scenarios requiring multiple independent mask layers with different combination strategies, use the `FilterMasks` API for managing named mask layers:

```python
from gsmod import GSDataPro, Filter
from gsmod.filter import FilterMasks

data = GSDataPro.from_ply("scene.ply")

# Create FilterMasks manager
masks = FilterMasks(data)

# Add multiple named mask layers using atomic filters
masks.add("opacity", Filter.min_opacity(0.3))
masks.add("sphere", Filter.sphere(radius=5.0))
masks.add("scale", Filter.max_scale(2.0))

# Inspect mask layers
masks.summary()
# Output:
# opacity: 45231/100000 (45.2%)
# sphere: 67890/100000 (67.9%)
# scale: 89123/100000 (89.1%)

# Combine masks with AND logic (all conditions must pass)
combined_and = masks.combine(mode="and")
print(f"{combined_and.sum():,} Gaussians pass all filters")

# Combine masks with OR logic (any condition passes)
combined_or = masks.combine(mode="or")
print(f"{combined_or.sum():,} Gaussians pass any filter")

# Apply masks directly to filter data
filtered = masks.apply(mode="and", inplace=False)

# Access individual mask layers
opacity_mask = masks["opacity"]
```

**Performance:** FilterMasks uses Numba-optimized mask combination with automatic strategy selection:
- **1 layer**: NumPy (0.006ms, lower overhead)
- **2+ layers**: Numba parallel (0.026ms vs 1.447ms numpy, 55x faster)
- **Large-scale (1M Gaussians, 5 layers)**: 0.425ms vs 14.587ms (34x faster)

The mask combination overhead is negligible (3.8% of total filtering time) thanks to Numba optimization, making multi-layer filtering practical for interactive applications.

#### Using Low-Level Utilities

```python
from gsmod.filter import (
    calculate_scene_bounds,
    calculate_recommended_max_scale
)
from gsmod import GSDataPro, FilterValues

data = GSDataPro.from_ply("scene.ply")

# Calculate scene bounds (for reference)
bounds = calculate_scene_bounds(data.means)
print(f"Scene center: {bounds.center}")
print(f"Scene size: {bounds.sizes}")

# Calculate recommended scale threshold
max_scale = calculate_recommended_max_scale(data.scales, percentile=99.5)
print(f"Recommended max_scale: {max_scale:.4f}")

# Use in filtering with absolute values
data.filter(FilterValues(
    sphere_center=tuple(bounds.center),
    sphere_radius=bounds.max_size * 0.8,  # 80% of scene size
    max_scale=max_scale
), inplace=True)
```

### Complete Processing Example

A full example combining all operations:

```python
from gsmod import GSDataPro, ColorValues, FilterValues, TransformValues
from gsmod import CINEMATIC

# Load data
data = GSDataPro.from_ply("scene.ply")

# Apply all operations with method chaining
(data
    # Filtering
    .filter(FilterValues(
        min_opacity=0.1,
        max_scale=2.5,
        sphere_radius=5.0  # Absolute radius in world units
    ))
    # Transforms
    .transform(TransformValues.from_translation(1, 0, 0))
    .transform(TransformValues.from_rotation_euler(0, 45, 0))
    .transform(TransformValues.from_scale(1.5))
    # Color grading
    .color(ColorValues(
        temperature=0.6,
        brightness=1.2,
        contrast=1.1,
        saturation=1.3
    ))
)

# Save
data.to_ply("output.ply")
```

#### Using Presets with Custom Values

```python
from gsmod import GSDataPro, ColorValues, FilterValues, TransformValues
from gsmod import CINEMATIC, WARM, STRICT_FILTER, DOUBLE_SIZE

data = GSDataPro.from_ply("scene.ply")

# Compose preset with custom adjustments
color_style = CINEMATIC + ColorValues(brightness=1.1)
data.color(color_style)

# Combine presets
warm_cinematic = WARM + CINEMATIC
data.color(warm_cinematic)

# Apply filter and transform presets
data.filter(STRICT_FILTER)
data.transform(DOUBLE_SIZE)

data.to_ply("output.ply")
```

---

## Performance

### Color Processing (100K colors)

| API | Time | Throughput | Speedup |
|-----|------|------------|---------|
| `apply()` | 0.473 ms | 211 M/s | 1.00x |
| `apply_numpy()` | 0.480 ms | 208 M/s | 0.99x |
| **`apply_numpy_inplace()`** | **0.072 ms** | **1,389 M/s** | **6.57x** |

### Color Batch Scaling (apply_numpy_inplace)

| Batch Size | Time | Throughput |
|------------|------|------------|
| 1K | 0.016 ms | 64 M/s |
| 10K | 0.022 ms | 461 M/s |
| 100K | 0.086 ms | 1,162 M/s |
| 1M | 0.581 ms | 1,722 M/s |

### 3D Transform (1M Gaussians)

| Operation | Time | Throughput |
|-----------|------|------------|
| Combined transform | 1.743 ms | 574 M G/s |

### Transform Batch Scaling

| Batch Size | Time | Throughput |
|------------|------|------------|
| 10K | 0.09 ms | 106 M G/s |
| 100K | 0.12 ms | 863 M G/s |
| 500K | 0.31 ms | 1,593 M G/s |
| 1M | 1.43 ms | 698 M G/s |
| 2M | 7.31 ms | 273 M G/s |

### Filtering Performance (1M Gaussians)

| Operation | Time | Throughput |
|-----------|------|------------|
| Scene bounds (one-time) | 35.7 ms | 28 M/s |
| Recommended scale (one-time) | 6.4 ms | 157 M/s |
| Sphere filter (nogil=True) | 2.5 ms | 405 M/s |
| Cuboid filter (nogil=True) | 4.8 ms | 207 M/s |
| Opacity filter (nogil=True) | 2.6 ms | 392 M/s |
| Scale filter (nogil=True) | 2.2 ms | 447 M/s |
| Combined filter | 3.6 ms | 276 M/s |
| **Full filtering (filter_gaussians)** | **16.1 ms** | **62.1 M/s** |

### Key Performance Highlights

**CPU Performance:**
- **Peak Color Processing**: 1,722M colors/sec (1M batch, zero-copy)
- **Peak Transform Speed**: 1,593M Gaussians/sec (500K batch)
- **Peak Filtering Speed**: 447M Gaussians/sec (scale filter)
- **Full Pipeline**: 62.1M Gaussians/sec (complete filtering)

**GPU Performance (1M Gaussians, RTX 3090 Ti):**
- **Peak Speedup**: 183.9x (sphere filter)
- **Average Speedup**: 43.2x across all operations
- **Throughput**: 1.09 billion Gaussians/sec
- **Transform Operations**: 86-93x speedup (translate, scale)
- **Color Operations**: 15-31x speedup (brightness, saturation)

- **Scalability**: Linear scaling from 1K to 2M Gaussians on both CPU and GPU
- **CPU-GPU Consistency**: All operations produce mathematically identical results
  - Rotation matrices correctly use transpose for inverse transformations
  - Filter operations (ellipsoid, box, frustum) match exactly between CPU and GPU
  - Verified via comprehensive equivalence tests

### Optimization Details

- **Zero-copy APIs**: Direct memory operations without allocation overhead (6.6x speedup)
- **LUT-based processing**: Pre-computed look-up tables for color operations
- **nogil=True**: True parallelism by releasing GIL (+15-37% performance)
- **Fused kernels**: Combined opacity+scale filtering in single pass
- **Parallel processing**: Numba JIT with `prange` for multi-core utilization
- **fastmath optimization**: Aggressive floating-point optimizations on all kernels
- **Mathematically correct rotations**: Rodrigues' formula implementation with proper axis normalization

---

## API Reference

### GSDataPro

Main data class extending GSData with processing methods.

```python
from gsmod import GSDataPro

# Loading
data = GSDataPro.from_ply("scene.ply")
data = GSDataPro.from_gsdata(gsdata)

# Methods (all return self for chaining)
data.color(values: ColorValues, inplace: bool = True) -> GSDataPro
data.filter(values: FilterValues, inplace: bool = True) -> GSDataPro
data.transform(values: TransformValues, inplace: bool = True) -> GSDataPro
data.clone() -> GSDataPro

# Saving
data.to_ply("output.ply")
```

### GSTensorPro (GPU)

GPU tensor class with same interface.

```python
from gsmod.torch import GSTensorPro

# Loading
data = GSTensorPro.from_ply("scene.ply", device="cuda")
data = GSTensorPro.from_gsdata(gsdata, device="cuda")

# Same methods as GSDataPro
data.color(values, inplace=True)
data.filter(values, inplace=True)
data.transform(values, inplace=True)

# Saving
data.to_ply("output.ply")
output = data.to_gsdata()
```

### Config Value Classes

**ColorValues** - Color adjustment parameters
```python
ColorValues(
    brightness=1.0, contrast=1.0, gamma=1.0, saturation=1.0,
    vibrance=1.0, temperature=0.0, tint=0.0, shadows=0.0, highlights=0.0,
    fade=0.0, hue_shift=0.0, shadow_tint_hue=0.0, shadow_tint_sat=0.0,
    highlight_tint_hue=0.0, highlight_tint_sat=0.0
)
ColorValues.from_k(kelvin)  # From color temperature
```

**OpacityValues** - Opacity adjustment parameters
```python
OpacityValues(scale=1.0)
OpacityValues.fade(0.7)    # Fade to 70%
OpacityValues.boost(1.5)   # Boost by 1.5x
```

**FilterValues** - Filter parameters
```python
FilterValues(
    min_opacity=0.0, max_opacity=1.0, min_scale=0.0, max_scale=inf,
    sphere_radius=inf, sphere_center=(0,0,0), box_min=None, box_max=None,
    invert=False  # Include (False) or Exclude (True) mode
)
```

**TransformValues** - Transform parameters
```python
TransformValues(scale=1.0, rotation=(1,0,0,0), translation=(0,0,0))
TransformValues.from_scale(factor)
TransformValues.from_translation(x, y, z)
TransformValues.from_rotation_euler(roll, pitch, yaw)  # degrees
TransformValues.from_euler_rad(roll, pitch, yaw)
TransformValues.from_rotation_axis_angle(axis, angle)  # degrees
TransformValues.from_axis_angle_rad(axis, angle)
```

### Histogram Analysis and Learning

**Compute Histograms:**
```python
from gsmod import GSDataPro, HistogramConfig

data = GSDataPro.from_ply("scene.ply")

# Compute color histogram
hist = data.histogram_colors(HistogramConfig(n_bins=256))
print(f"Mean RGB: {hist.mean}")
print(f"Std RGB: {hist.std}")

# Analysis methods
hist.percentile(50, channel=0)  # Median of red channel
hist.mode(channel=1)             # Most frequent green value
hist.entropy(channel=2)          # Blue channel entropy
hist.dynamic_range()             # 99th - 1st percentile
```

**Histogram-Based Learning:**
```python
import torch

# Learn color adjustments from target histogram
target_hist = reference.histogram_colors()
source_colors = torch.tensor(source.sh0)

learned = target_hist.learn_from(
    source_colors,
    params=["brightness", "contrast", "saturation", "gamma"],
    n_epochs=100,
    lr=0.02,
    verbose=True
)

source.color(learned)
```

**Rule-Based Adjustments:**
```python
# Generate ColorValues from histogram profile
hist = data.histogram_colors()

adjustment = hist.to_color_values("vibrant")   # Boost saturation, contrast
# adjustment = hist.to_color_values("dramatic")  # Strong contrast, dark shadows
# adjustment = hist.to_color_values("bright")    # Shift distribution higher
# adjustment = hist.to_color_values("dark")      # Shift distribution lower
# adjustment = hist.to_color_values("neutral")   # Flatten toward uniform

data.color(adjustment)
```

### Built-in Presets

**Color Presets (39 total):**

*Basic Looks:*
- `WARM`, `COOL`, `NEUTRAL`
- `CINEMATIC`, `VIBRANT`, `MUTED`, `DRAMATIC`
- `VINTAGE`, `GOLDEN_HOUR`, `MOONLIGHT`

*Film Stock Emulation:*
- `KODAK_PORTRA`, `FUJI_VELVIA`, `KODAK_EKTACHROME`
- `ILFORD_HP5`, `CINESTILL_800T`

*Seasonal Looks:*
- `SPRING_FRESH`, `SUMMER_BRIGHT`, `AUTUMN_WARM`, `WINTER_COLD`

*Time of Day:*
- `SUNRISE`, `MIDDAY_SUN`, `SUNSET`, `BLUE_HOUR`, `OVERCAST`

*Artistic Styles:*
- `HIGH_KEY`, `LOW_KEY`, `TEAL_ORANGE`
- `BLEACH_BYPASS`, `CROSS_PROCESS`, `FADED_PRINT`, `SEPIA_TONE`

*Technical Adjustments:*
- `LIFT_SHADOWS`, `COMPRESS_HIGHLIGHTS`, `INCREASE_CONTRAST`
- `DESATURATE_MILD`, `ENHANCE_COLORS`

**Opacity Presets (12 total):**
- Fade: `FADE_SUBTLE`, `FADE_MILD`, `FADE_MODERATE`, `FADE_HEAVY`
- Boost: `BOOST_MILD`, `BOOST_MODERATE`, `BOOST_STRONG`
- Special Effects: `GHOST_EFFECT`, `TRANSLUCENT`, `SEMI_TRANSPARENT`

**Filter Presets:**
- `STRICT_FILTER` - min_opacity=0.5, max_scale=1.0, sphere_radius=10.0
- `QUALITY_FILTER` - min_opacity=0.3, max_scale=2.0
- `CLEANUP_FILTER` - min_opacity=0.1, max_scale=5.0

**Transform Presets:**
- `DOUBLE_SIZE`, `HALF_SIZE`
- `FLIP_X`, `FLIP_Y`, `FLIP_Z`

### Preset Loading Functions

```python
from gsmod.config.presets import (
    get_color_preset, get_filter_preset, get_transform_preset, get_opacity_preset,
    color_from_dict, filter_from_dict, transform_from_dict,
    load_color_json, load_filter_json, load_transform_json,
    save_color_json, save_filter_json, save_transform_json,
)

# Load presets by name
color = get_color_preset("cinematic")
opacity = get_opacity_preset("fade_moderate")
filter_vals = get_filter_preset("strict")
transform = get_transform_preset("double_size")
```

### Advanced Classes

For advanced use cases like mask computation and fine-grained control:
```python
from gsmod import Pipeline, Color, Transform, Filter, FilterMasks
```

- **Pipeline**: Unified CPU pipeline (new in 0.1.3) - matches PipelineGPU interface
- **Filter**: Atomic filter with boolean operators (new in 0.1.3)
- **Color, Transform**: Legacy pipeline classes for backward compatibility
- **FilterMasks**: Multi-layer mask management

### Low-Level Utilities

Quaternion operations:
```python
from gsmod.transforms import (
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    axis_angle_to_quaternion,
    euler_to_quaternion,
    quaternion_to_euler
)
```

Scene bounds:
```python
from gsmod.filter.bounds import (
    calculate_scene_bounds,        # -> SceneBounds
    calculate_recommended_max_scale  # -> float
)

bounds = calculate_scene_bounds(positions)
# Returns: SceneBounds(min, max, sizes, max_size, center)

max_scale = calculate_recommended_max_scale(scales, percentile=99.5)
```

**Rotation methods:** `from_rotation_euler()`, `from_euler_rad()`, `from_rotation_axis_angle()`, `from_axis_angle_rad()`

### Scene Composition

Functions for combining and manipulating multiple Gaussian scenes:

```python
from gsmod import (
    concatenate,
    compose_with_transforms,
    deduplicate,
    merge_scenes,
    split_by_region
)

# Load multiple scenes
scene1 = GSDataPro.from_ply("scene1.ply")
scene2 = GSDataPro.from_ply("scene2.ply")

# Simple concatenation
combined = concatenate([scene1, scene2])

# Compose with transforms (position scenes in space)
transforms = [
    TransformValues.from_translation(-2, 0, 0),
    TransformValues.from_translation(2, 0, 0),
]
composed = compose_with_transforms([scene1, scene2], transforms)

# Remove duplicate Gaussians
cleaned = deduplicate(combined, threshold=0.01)

# Split by spatial region
left, right = split_by_region(combined, axis=0, threshold=0.0)
```

### Parameterized Templates

Create reusable pipeline templates with named parameters for efficient parameter sweeps and animation:

```python
from gsmod import Color, Param

# Create template with named parameters
template = Color.template(
    brightness=Param("b", default=1.2, range=(0.5, 2.0)),
    contrast=Param("c", default=1.1, range=(0.5, 2.0)),
    saturation=Param("s", default=1.3, range=(0.0, 3.0))
)

# Use with different parameters (auto-cached for performance)
result1 = template(data, params={"b": 1.5, "c": 1.2, "s": 1.4})
result2 = template(data, params={"b": 0.8, "c": 1.0, "s": 1.0})

# Animation use case - cached LUTs for efficiency
import numpy as np
for t in np.linspace(0, 1, 100):
    brightness = 1.0 + t * 1.0  # Animate 1.0 to 2.0
    result = template(data, params={"b": brightness})
```

### Learnable Modules (Training)

PyTorch nn.Module classes for gradient-based optimization:

```python
from gsmod.torch import GSTensorPro, LearnableColor, LearnableTransform, LearnableFilter
from gsmod import ColorValues
import torch

# Create learnable module from initial values
initial = ColorValues(brightness=1.0, contrast=1.0, saturation=1.0)
learnable = initial.learn("brightness", "saturation")

# Or create directly
learnable = LearnableColor(brightness=True, contrast=False, saturation=True)

# Use in training loop
optimizer = torch.optim.Adam(learnable.parameters(), lr=0.01)

for epoch in range(100):
    result = learnable(data.sh0)
    loss = compute_loss(result, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Extract learned values
learned_values = learnable.to_values()
```

### PipelineGPU (Unified GPU Pipeline)

Chain all operations in a single fluent GPU pipeline:

```python
from gsmod.torch import GSTensorPro, PipelineGPU

# Load data to GPU
data = GSTensorPro.from_ply("scene.ply", device="cuda")

# Create unified pipeline
pipeline = (
    PipelineGPU()
    .min_opacity(0.1)
    .within_sphere(radius=5.0)
    .translate([1, 0, 0])
    .rotate_euler(0, 45, 0)
    .scale(2.0)
    .brightness(1.2)
    .saturation(1.3)
)

# Apply all operations
result = pipeline(data, inplace=True)
```

## Example: Full Processing Pipeline

```python
from gsmod import GSDataPro, ColorValues, FilterValues, TransformValues
from gsmod import CINEMATIC

# Load Gaussian splatting data
data = GSDataPro.from_ply("scene.ply")

# Apply full processing pipeline
(data
    # Filtering (remove unwanted Gaussians)
    .filter(FilterValues(
        min_opacity=0.1,        # Remove low-opacity
        max_scale=2.5,          # Remove large-scale outliers
        sphere_radius=5.0       # Absolute radius in world units
    ))
    # Geometric transforms
    .transform(TransformValues.from_translation(1.0, 0.0, 0.0))
    .transform(TransformValues.from_rotation_euler(0, 45, 0))
    .transform(TransformValues.from_scale(1.5))
    # Color grading
    .color(ColorValues(
        temperature=0.6,        # Cool tones
        brightness=1.2,         # Increase brightness
        contrast=1.1,           # Boost contrast
        saturation=1.3          # Vibrant colors
    ))
)

# Save processed scene
data.to_ply("output.ply")

# Performance notes:
# - inplace=True (default): Zero-copy modification for maximum performance
# - Filtering: 62M Gaussians/sec full pipeline
# - Transforms: 698M Gaussians/sec combined operations
# - Colors: 1,389M colors/sec with LUT-based processing
```

---

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/OpsiClear/gsmod.git
cd gsmod

# Install in development mode
pip install -e .[dev]

# Set up pre-commit hooks (recommended)
pre-commit install

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=gsmod --cov-report=html
```

### Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to maintain code quality:

```bash
# Install hooks (one-time setup)
pre-commit install

# Run manually on all files
pre-commit run --all-files

# Update hook versions
pre-commit autoupdate
```

The pre-commit hooks will automatically:
- Run ruff linting with auto-fix
- Format code with ruff
- Check for common issues (trailing whitespace, YAML syntax, etc.)
- Validate Python syntax

See [.github/PRE_COMMIT_SETUP.md](.github/PRE_COMMIT_SETUP.md) for detailed setup instructions.

### Project Structure

```
gsmod/
├── src/gsmod/
│   ├── __init__.py            # Public API
│   ├── pipeline.py            # Unified Pipeline class
│   ├── compose.py             # Scene composition
│   ├── params.py              # Parameterized pipelines
│   ├── protocols.py           # Protocol definitions
│   ├── validators.py          # Input validation
│   ├── constants.py           # Global constants
│   ├── utils.py               # Shared utilities
│   ├── color/                 # Color processing
│   │   ├── pipeline.py        # Color class
│   │   ├── presets.py         # ColorPreset class
│   │   └── kernels.py         # Numba kernels
│   ├── transform/             # 3D transformations
│   │   ├── api.py             # Quaternion utilities
│   │   ├── pipeline.py        # Transform class
│   │   └── kernels.py         # Numba kernels
│   ├── filter/                # Spatial filtering
│   │   ├── api.py             # Core implementation
│   │   ├── pipeline.py        # Filter class
│   │   ├── masks.py           # FilterMasks API
│   │   ├── bounds.py          # Scene bounds
│   │   ├── config.py          # FilterConfig
│   │   └── kernels.py         # Numba kernels
│   └── torch/                 # GPU operations
│       ├── gstensor_pro.py    # GSTensorPro class
│       ├── color.py           # ColorGPU
│       ├── transform.py       # TransformGPU
│       ├── filter.py          # FilterGPU
│       └── pipeline.py        # PipelineGPU
├── tests/                     # Unit tests
├── benchmarks/                # Performance benchmarks
├── docs/                      # Documentation
│   └── GPU_API_REFERENCE.md
├── .github/workflows/         # CI/CD
├── pyproject.toml             # Package config
└── README.md                  # This file
```

---

## Benchmarking

Run performance benchmarks to measure library performance:

```bash
# Run all benchmarks
cd benchmarks
uv run run_all_benchmarks.py

# Run specific benchmark
uv run benchmark_color_lut.py

# Run large-scale benchmark (1M+ Gaussians)
uv run benchmark_large_scale.py
```

The benchmarks measure:
- **Color processing**: LUT application across different batch sizes
- **Transform performance**: Geometric operations on Gaussians
- **Filtering performance**: Spatial and property-based filtering
- **Scalability**: Performance across varying data sizes

---

## Testing

gsmod has comprehensive test coverage with passing tests:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_color_pipeline.py -v

# Run with coverage report
pytest tests/ -v --cov=gsmod --cov-report=html
```

Test categories:
- Color LUT operations (all adjustment types, caching, edge cases)
- 3D transforms (translation, rotation, scaling, combined)
- Quaternion utilities (conversions, multiplication)
- Pipeline API (composition, presets, custom operations)
- Filtering system (volume filters, property filters, combined logic)

---

## Documentation

For detailed documentation see:
- **OPTIMIZATION_COMPLETE_SUMMARY.md** - Complete optimization history and performance analysis
- **AUDIT_FIXES_SUMMARY.md** - Bug fixes and validation methodology
- **.github/WORKFLOWS.md** - CI/CD pipeline documentation
- **benchmarks/README.md** - Benchmark suite documentation

---

## CI/CD

gsmod includes a complete GitHub Actions CI/CD pipeline:

- **Multi-platform testing**: Ubuntu, Windows, macOS
- **Multi-version testing**: Python 3.10, 3.11, 3.12, 3.13
- **Automated benchmarking**: Performance tracking on PRs
- **Build verification**: Wheel building and installation testing
- **PyPI publishing**: Automated release on GitHub Release

See [.github/WORKFLOWS.md](.github/WORKFLOWS.md) for details.

---

## Architecture

**Color Processing:**
- Phase 1: LUT-capable operations (temperature, brightness, contrast, gamma) pre-compiled into 1D LUTs
- Phase 2: Dependent operations (saturation, shadows/highlights) with branchless code
- Single fused Numba kernel with interleaved LUT layout for cache locality
- Zero-copy API eliminates 80% memory allocation overhead

**3D Transforms:**
- Matrix-based operations (scale, rotate, translate)
- Numba-accelerated quaternion operations
- Fused transforms for single-pass processing
- Parallel processing with `prange`

**Filtering System:**
- Fused opacity+scale kernel for 1.95x speedup
- Parallel scatter pattern with prefix sum for lock-free writes
- fastmath optimization on all kernels (5-10% speedup)
- Numba JIT compilation with parallel execution

---

## Contributing

Contributions are welcome! Please see [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

**Quick start:**
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run tests and benchmarks
5. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Citation

If you use gsmod in your research, please cite:

```bibtex
@software{gsmod2025,
  author = {OpsiClear},
  title = {gsmod: High-Performance Processing for 3D Gaussian Splatting},
  year = {2025},
  url = {https://github.com/OpsiClear/gsmod}
}
```

---

## Related Projects

- **gsply**: Ultra-fast Gaussian Splatting PLY I/O library (required dependency)
  - v0.3.0+ adds concatenation optimizations (6.15x faster bulk merging)
  - `make_contiguous()` for manual optimization of iterative workflows (100+ operations)
  - Multi-layer mask management (used by FilterMasks in gsmod)
  - See [gsply documentation](https://github.com/OpsiClear/gsply) for details
- **gsplat**: CUDA-accelerated Gaussian Splatting rasterizer
- **nerfstudio**: NeRF training framework with Gaussian Splatting support
- **3D Gaussian Splatting**: Original paper and implementation

---

<div align="center">

**Made with Python and NumPy**

[Report Bug](https://github.com/OpsiClear/gsmod/issues) | [Request Feature](https://github.com/OpsiClear/gsmod/issues) | [Documentation](.github/WORKFLOWS.md)

</div>
