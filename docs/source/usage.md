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

```python
from gsmod import GSDataPro, TransformValues

data = GSDataPro.from_ply("scene.ply")

# Translation
data.transform(TransformValues.from_translation(1.0, 0.0, 0.0))

# Rotation (Euler angles in degrees)
data.transform(TransformValues.from_rotation_euler(0, 45, 0))

# Uniform scale
data.transform(TransformValues.from_scale(2.0))

# Non-uniform scale
data.transform(TransformValues.from_scale(2.0, 1.0, 0.5))
```

### Using Transform Pipeline

```python
from gsmod import Transform

# Chain operations (matrix fusion optimization)
pipeline = (Transform()
    .translate([1, 0, 0])
    .rotate_euler(0, 45, 0)
    .scale(2.0))

result = pipeline(data, inplace=True)
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
from gsmod import GSDataPro, ColorValues, FilterValues, TransformValues

data = GSDataPro.from_ply("scene.ply")

# Chain operations fluently
(data
    .filter(FilterValues(min_opacity=0.1))
    .transform(TransformValues.from_scale(2.0))
    .color(ColorValues(brightness=1.2))
    .to_ply("output.ply"))
```

## GPU Acceleration

```python
from gsmod.torch import GSTensorPro
from gsmod import ColorValues, FilterValues, TransformValues

# Load directly to GPU
data = GSTensorPro.from_ply("scene.ply", device="cuda")

# Same API as CPU
data.filter(FilterValues(min_opacity=0.1))
data.transform(TransformValues.from_translation(1, 0, 0))
data.color(ColorValues(brightness=1.2))

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
