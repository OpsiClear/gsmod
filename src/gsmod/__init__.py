"""
gsmod - Gaussian Splatting Processing

Ultra-fast CPU-optimized processing for 3D Gaussian Splatting.

Features:
- Ultra-fast RGB color adjustments via compiled LUT pipelines
- 3D geometric transformations (NumPy/Numba optimized)
- Spatial filtering (sphere, box, min_opacity, max_scale)
- GSDataPro API for direct data operations
- Zero-copy in-place processing where possible
- Adjustments: temperature, brightness, contrast, gamma, saturation, vibrance, hue_shift, shadows, highlights
- Transforms: translate, rotate_quat/euler/axis_angle/matrix, scale
- Filtering: spherical/cuboid volumes, opacity, scale thresholds
- Parameterized templates for efficient parameter variation

Performance (with inplace=True, recommended):
  - Color: 1,015M/sec (0.10ms for 100K) | Kernel: 1,091M/sec (0.092ms)
  - Transform: 698M Gaussians/sec (1.43ms for 1M)
  - Filter: 46M/sec (2.2ms for 100K)

Example - GSDataPro (Recommended):
    >>> from gsmod import GSDataPro, ColorValues, FilterValues, TransformValues
    >>>
    >>> # Load and process
    >>> data = GSDataPro.from_ply("scene.ply")
    >>> data.filter(FilterValues(min_opacity=0.1, sphere_radius=5.0))
    >>> data.transform(TransformValues.from_translation(1, 0, 0))
    >>> data.color(ColorValues(brightness=1.2, saturation=1.3))
    >>> data.to_ply("output.ply")

Example - Advanced (Low-level Pipelines):
    >>> from gsmod import Color, Transform
    >>>
    >>> # For fine-grained control over compilation/optimization
    >>> result = Transform().rotate_quat(quat).translate([1, 0, 0])(data)
    >>> result = Color().brightness(1.2).saturation(1.3)(result)

Example - Parameterized Templates:
    >>> from gsmod import Color, Param
    >>>
    >>> # Create template with parameters
    >>> template = Color.template(
    ...     brightness=Param("b", default=1.2, range=(0.5, 2.0)),
    ...     contrast=Param("c", default=1.1, range=(0.5, 2.0))
    ... )
    >>>
    >>> # Use with different parameter values (cached for performance)
    >>> result = template(data, params={"b": 1.5, "c": 1.2})
"""

__version__ = "0.1.4"

# Import GSData from gsply
from gsply import GSData

# Color processing pipeline (legacy)
from gsmod.color.pipeline import Color
from gsmod.color.presets import ColorPreset

# Scene composition utilities
from gsmod.compose import (
    compose_with_transforms,
    concatenate,
    deduplicate,
    merge_scenes,
    split_by_region,
)
from gsmod.config.presets import (
    CINEMATIC,
    CLEANUP_FILTER,
    COOL,
    # Transform presets
    DOUBLE_SIZE,
    DRAMATIC,
    # Popular new color presets
    FADE_MODERATE,
    FLIP_X,
    FLIP_Y,
    FLIP_Z,
    GHOST_EFFECT,
    GOLDEN_HOUR,
    HALF_SIZE,
    HIGH_KEY,
    KODAK_PORTRA,
    LOW_KEY,
    MOONLIGHT,
    MUTED,
    NEUTRAL,
    QUALITY_FILTER,
    # Filter presets
    STRICT_FILTER,
    SUNSET,
    TEAL_ORANGE,
    VIBRANT,
    VINTAGE,
    # Color presets
    WARM,
    color_from_dict,
    filter_from_dict,
    # Loading functions
    get_color_preset,
    get_filter_preset,
    get_opacity_preset,
    get_transform_preset,
    load_color_json,
    load_filter_json,
    load_transform_json,
    transform_from_dict,
)

# Config values and presets
from gsmod.config.values import (
    ColorValues,
    FilterValues,
    HistogramConfig,
    OpacityValues,
    TransformValues,
)

# Filtering utilities
from gsmod.filter.atomic import Filter
from gsmod.filter.bounds import (
    SceneBounds,
    calculate_recommended_max_scale,
    calculate_scene_bounds,
)

# New simplified API
from gsmod.gsdata_pro import GSDataPro

# Histogram computation
from gsmod.histogram import HistogramResult

# Parameterized pipelines
from gsmod.params import Param

# Unified pipeline
from gsmod.pipeline import Pipeline

# Unified processing
from gsmod.processing import GaussianProcessor, get_processor

# Protocols
from gsmod.protocols import (
    ColorProcessor,
    FilterProcessor,
    PipelineStage,
    TransformProcessor,
)

# Quaternion utilities
from gsmod.transform.api import (
    axis_angle_to_quaternion,
    euler_to_quaternion,
    quaternion_multiply,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)

# Transform pipeline
from gsmod.transform.pipeline import Transform

# Utility functions
from gsmod.utils import linear_interp_1d, multiply_opacity, nearest_neighbor_1d

# Verification utilities
from gsmod.verification import FormatVerifier

__all__ = [
    # Version
    "__version__",
    # Data structures
    "GSData",
    "GSDataPro",
    # Config values
    "ColorValues",
    "FilterValues",
    "TransformValues",
    "OpacityValues",
    "HistogramConfig",
    # Histogram
    "HistogramResult",
    # Color presets - Basic
    "WARM",
    "COOL",
    "NEUTRAL",
    "CINEMATIC",
    "VIBRANT",
    "MUTED",
    "DRAMATIC",
    "VINTAGE",
    "GOLDEN_HOUR",
    "MOONLIGHT",
    # Color presets - Popular additions
    "KODAK_PORTRA",
    "HIGH_KEY",
    "LOW_KEY",
    "TEAL_ORANGE",
    "SUNSET",
    # Opacity presets
    "FADE_MODERATE",
    "GHOST_EFFECT",
    # Filter presets
    "STRICT_FILTER",
    "QUALITY_FILTER",
    "CLEANUP_FILTER",
    # Transform presets
    "DOUBLE_SIZE",
    "HALF_SIZE",
    "FLIP_X",
    "FLIP_Y",
    "FLIP_Z",
    # Preset loading functions
    "get_color_preset",
    "get_filter_preset",
    "get_transform_preset",
    "get_opacity_preset",
    "color_from_dict",
    "filter_from_dict",
    "transform_from_dict",
    "load_color_json",
    "load_filter_json",
    "load_transform_json",
    # Core pipeline classes (advanced)
    "Color",
    "ColorPreset",
    "Transform",
    "Filter",
    "Pipeline",
    # Parameterization
    "Param",
    # Unified processing
    "GaussianProcessor",
    "get_processor",
    # Protocols
    "PipelineStage",
    "ColorProcessor",
    "TransformProcessor",
    "FilterProcessor",
    # Scene composition
    "concatenate",
    "compose_with_transforms",
    "deduplicate",
    "merge_scenes",
    "split_by_region",
    # Quaternion utilities
    "quaternion_multiply",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "axis_angle_to_quaternion",
    "euler_to_quaternion",
    "quaternion_to_euler",
    # Filtering utilities
    "calculate_scene_bounds",
    "calculate_recommended_max_scale",
    "SceneBounds",
    # Utils
    "linear_interp_1d",
    "nearest_neighbor_1d",
    "multiply_opacity",
    # Verification
    "FormatVerifier",
]
