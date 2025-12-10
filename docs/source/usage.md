# Usage Guide

This guide covers common usage patterns for gsmod.

## Loading and Saving Data

```python
from gsmod import GSDataPro

# Load from PLY file
data = GSDataPro.from_ply("scene.ply")

# Save to PLY file
data.to_ply("output.ply")

# Access data arrays
print(f"Number of Gaussians: {len(data)}")
print(f"Means shape: {data.means.shape}")
print(f"Colors shape: {data.sh0.shape}")
```

## Color Adjustments

### Using ColorValues

```python
from gsmod import GSDataPro, ColorValues

data = GSDataPro.from_ply("scene.ply")

# Basic adjustments
data.color(ColorValues(
    brightness=1.2,
    contrast=1.1,
    saturation=1.3,
    gamma=1.0
))

# Temperature and tint
data.color(ColorValues(
    temperature=0.3,  # warm
    tint=0.0          # neutral
))

# Shadows and highlights
data.color(ColorValues(
    shadows=0.1,      # lift shadows
    highlights=-0.05  # reduce highlights
))
```

### Using Presets

```python
from gsmod import GSDataPro, CINEMATIC, WARM, ColorValues

data = GSDataPro.from_ply("scene.ply")

# Apply preset
data.color(CINEMATIC)

# Combine preset with custom values
data.color(WARM + ColorValues(brightness=1.1))
```

## Geometric Transforms

### Using TransformValues

The simplest way to apply transforms using config dataclasses:

```python
from gsmod import GSDataPro, TransformValues

data = GSDataPro.from_ply("scene.ply")

# Translation
data.transform(TransformValues.from_translation(1.0, 0.0, 0.0))

# Rotation (Euler angles in degrees: roll, pitch, yaw)
data.transform(TransformValues.from_rotation_euler(0, 45, 0))

# Rotation (axis-angle in degrees)
data.transform(TransformValues.from_rotation_axis_angle(
    axis=(0, 1, 0),  # Y-axis
    angle=45         # degrees
))

# Uniform scale
data.transform(TransformValues.from_scale(2.0))

# Non-uniform scale (per-axis)
data.transform(TransformValues.from_scale((2.0, 1.0, 0.5)))
```

### Center Point for Rotation/Scale

By default, rotations and scaling happen around the world origin (0, 0, 0).
To rotate/scale around a specific point (e.g., the object's centroid):

```python
from gsmod import GSDataPro, TransformValues

data = GSDataPro.from_ply("scene.ply")

# Compute object centroid
centroid = tuple(data.means.mean(axis=0))

# Rotate 45 degrees around the object's own center
data.transform(TransformValues.from_rotation_euler(0, 45, 0, center=centroid))

# Scale around object center (object stays in place)
data.transform(TransformValues.from_scale(2.0, center=centroid))
```

### Composing Transforms

Use the `+` operator to compose multiple transforms:

```python
from gsmod import TransformValues

# Create individual transforms
translate = TransformValues.from_translation(1.0, 0.0, 0.0)
rotate = TransformValues.from_rotation_euler(0, 45, 0)
scale = TransformValues.from_scale(2.0)

# Compose: translate first, then rotate, then scale
combined = translate + rotate + scale

# Apply composed transform
data.transform(combined)
```

### Rotation Formats

```python
from gsmod import TransformValues
import numpy as np

# Euler angles (degrees) - roll, pitch, yaw
t1 = TransformValues.from_rotation_euler(roll=0, pitch=45, yaw=0)

# Euler angles (radians)
t2 = TransformValues.from_euler_rad(0, np.pi/4, 0)

# Axis-angle (degrees)
t3 = TransformValues.from_rotation_axis_angle(axis=(0, 1, 0), angle=45)

# Axis-angle (radians)
t4 = TransformValues.from_axis_angle_rad(axis=(0, 1, 0), angle=np.pi/4)
```

### Transform Pipeline (Advanced)

For complex transform sequences with matrix fusion optimization:

```python
from gsmod import Transform

# Method chaining with automatic matrix fusion
pipeline = (Transform()
    .translate([1, 0, 0])
    .rotate_euler_deg([0, 45, 0])  # Degrees
    .scale(2.0))

result = pipeline(data, inplace=True)

# Rotate around a center point
centroid = data.means.mean(axis=0)
pipeline = (Transform()
    .rotate_euler_deg([0, 90, 0], center=centroid)
    .scale(0.5, center=centroid))

# Using from_srt factory (Scale-Rotate-Translate order)
pipeline = Transform.from_srt(
    scale=2.0,
    rotation=[1, 0, 0, 0],  # quaternion (w, x, y, z)
    translation=[1, 0, 0]
)
```

### Direct Transform Methods

GSDataPro and GSTensorPro provide direct methods for common operations:

```python
from gsmod import GSDataPro

data = GSDataPro.from_ply("scene.ply")

# Translation
data.translate([1.0, 0.0, 0.0])

# Uniform scale
data.scale_uniform(2.0)

# Non-uniform scale
data.scale_nonuniform([2.0, 1.0, 0.5])

# Rotation methods
data.rotate_euler([0, 45, 0])           # Euler angles (degrees)
data.rotate_quaternion([1, 0, 0, 0])    # Quaternion (w, x, y, z)
data.rotate_axis_angle([0, 1, 0], 45)   # Axis + angle (degrees)

# Apply arbitrary 4x4 matrix
import numpy as np
matrix = np.eye(4, dtype=np.float32)
matrix[0, 3] = 1.0  # translate X by 1
data.transform_matrix(matrix)
```

### Utility Methods

```python
from gsmod import GSDataPro

data = GSDataPro.from_ply("scene.ply")

# Center scene at origin
data.center_at_origin()

# Normalize scale to fit in a bounding box
data.normalize_scale(target_size=1.0)
```

## Opacity Adjustments

### Using OpacityValues

```python
from gsmod import GSDataPro, OpacityValues

data = GSDataPro.from_ply("scene.ply")

# Fade to 80% opacity
data.opacity(OpacityValues.fade(0.8))

# Boost opacity by 20%
data.opacity(OpacityValues.boost(1.2))

# Use presets
from gsmod import FADE_MODERATE, GHOST_EFFECT
data.opacity(FADE_MODERATE)  # Fade to 70%
data.opacity(GHOST_EFFECT)   # 50% opacity
```

## Filtering

### Using FilterValues

```python
from gsmod import GSDataPro, FilterValues

data = GSDataPro.from_ply("scene.ply")

# Filter by opacity and scale
data.filter(FilterValues(
    min_opacity=0.1,
    max_scale=2.5
))

# Spatial filtering
data.filter(FilterValues(
    sphere_radius=5.0,
    sphere_center=(0, 0, 0)
))
```

### Using Filter Presets

```python
from gsmod import GSDataPro, STRICT_FILTER, QUALITY_FILTER

data = GSDataPro.from_ply("scene.ply")
data.filter(STRICT_FILTER)
```

## Method Chaining

```python
from gsmod import GSDataPro, ColorValues, FilterValues, TransformValues, OpacityValues

data = GSDataPro.from_ply("scene.ply")

# Chain operations fluently
(data
    .filter(FilterValues(min_opacity=0.1))
    .transform(TransformValues.from_scale(2.0))
    .color(ColorValues(brightness=1.2))
    .opacity(OpacityValues.fade(0.8))
    .to_ply("output.ply"))
```

## GPU Acceleration

```python
from gsmod.torch import GSTensorPro
from gsmod import ColorValues, FilterValues, TransformValues, OpacityValues

# Load directly to GPU
data = GSTensorPro.from_ply("scene.ply", device="cuda")

# Same API as CPU
data.filter(FilterValues(min_opacity=0.1))
data.transform(TransformValues.from_translation(1, 0, 0))
data.color(ColorValues(brightness=1.2))
data.opacity(OpacityValues.fade(0.8))

# Save result
data.to_ply("output.ply")
```

## Scene Composition

```python
from gsmod import GSDataPro, concatenate, compose_with_transforms
from gsmod import TransformValues

# Load scenes
scene1 = GSDataPro.from_ply("scene1.ply")
scene2 = GSDataPro.from_ply("scene2.ply")

# Simple concatenation
combined = concatenate([scene1, scene2])

# Compose with transforms
transforms = [
    TransformValues.from_translation(-1, 0, 0),
    TransformValues.from_translation(1, 0, 0),
]
composed = compose_with_transforms([scene1, scene2], transforms)
```
