"""Preset library for gsmod operations.

Provides pre-configured value objects for common use cases,
with support for loading from dict and JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from gsmod.config.values import ColorValues, FilterValues, TransformValues

logger = logging.getLogger(__name__)

# ============================================================================
# Color Presets
# ============================================================================

WARM = ColorValues.from_k(3200)
COOL = ColorValues.from_k(8500)
NEUTRAL = ColorValues()

CINEMATIC = (
    ColorValues.from_k(4500)
    + ColorValues(contrast=1.1, saturation=0.9, shadows=0.1, highlights=-0.05)
)

VIBRANT = ColorValues(saturation=1.3, vibrance=1.2, brightness=1.05)

MUTED = ColorValues(saturation=0.7, vibrance=0.8, contrast=0.95)

DRAMATIC = ColorValues(contrast=1.25, shadows=0.15, highlights=-0.1)

VINTAGE = (
    ColorValues.from_k(4000)
    + ColorValues(saturation=0.8, contrast=0.9, gamma=1.1)
)

GOLDEN_HOUR = (
    ColorValues.from_k(3000)
    + ColorValues(brightness=1.1, saturation=1.1, shadows=0.05)
)

MOONLIGHT = (
    ColorValues.from_k(9000)
    + ColorValues(brightness=0.8, contrast=1.1, saturation=0.85)
)

# ============================================================================
# Filter Presets
# ============================================================================

STRICT_FILTER = FilterValues(min_opacity=0.5, max_scale=1.0, sphere_radius=10.0)

QUALITY_FILTER = FilterValues(min_opacity=0.3, max_scale=2.0)

CLEANUP_FILTER = FilterValues(min_opacity=0.1, max_scale=5.0)

# ============================================================================
# Transform Presets
# ============================================================================

DOUBLE_SIZE = TransformValues.from_scale(2.0)
HALF_SIZE = TransformValues.from_scale(0.5)

FLIP_X = TransformValues.from_rotation_euler(0, 180, 0)
FLIP_Y = TransformValues.from_rotation_euler(180, 0, 0)
FLIP_Z = TransformValues.from_rotation_euler(0, 0, 180)

# ============================================================================
# Preset Registry
# ============================================================================

COLOR_PRESETS: dict[str, ColorValues] = {
    "warm": WARM,
    "cool": COOL,
    "neutral": NEUTRAL,
    "cinematic": CINEMATIC,
    "vibrant": VIBRANT,
    "muted": MUTED,
    "dramatic": DRAMATIC,
    "vintage": VINTAGE,
    "golden_hour": GOLDEN_HOUR,
    "moonlight": MOONLIGHT,
}

FILTER_PRESETS: dict[str, FilterValues] = {
    "strict": STRICT_FILTER,
    "quality": QUALITY_FILTER,
    "cleanup": CLEANUP_FILTER,
}

TRANSFORM_PRESETS: dict[str, TransformValues] = {
    "double_size": DOUBLE_SIZE,
    "half_size": HALF_SIZE,
    "flip_x": FLIP_X,
    "flip_y": FLIP_Y,
    "flip_z": FLIP_Z,
}

# ============================================================================
# Loading Functions
# ============================================================================


def get_color_preset(name: str) -> ColorValues:
    """Get color preset by name.

    :param name: Preset name (case-insensitive)
    :return: ColorValues preset
    :raises KeyError: If preset not found
    """
    name_lower = name.lower()
    if name_lower not in COLOR_PRESETS:
        available = ", ".join(COLOR_PRESETS.keys())
        raise KeyError(f"Unknown color preset '{name}'. Available: {available}")
    return COLOR_PRESETS[name_lower]


def get_filter_preset(name: str) -> FilterValues:
    """Get filter preset by name.

    :param name: Preset name (case-insensitive)
    :return: FilterValues preset
    :raises KeyError: If preset not found
    """
    name_lower = name.lower()
    if name_lower not in FILTER_PRESETS:
        available = ", ".join(FILTER_PRESETS.keys())
        raise KeyError(f"Unknown filter preset '{name}'. Available: {available}")
    return FILTER_PRESETS[name_lower]


def get_transform_preset(name: str) -> TransformValues:
    """Get transform preset by name.

    :param name: Preset name (case-insensitive)
    :return: TransformValues preset
    :raises KeyError: If preset not found
    """
    name_lower = name.lower()
    if name_lower not in TRANSFORM_PRESETS:
        available = ", ".join(TRANSFORM_PRESETS.keys())
        raise KeyError(f"Unknown transform preset '{name}'. Available: {available}")
    return TRANSFORM_PRESETS[name_lower]


# ============================================================================
# Dict/JSON Loading
# ============================================================================


def color_from_dict(d: dict) -> ColorValues:
    """Create ColorValues from dictionary.

    :param d: Dictionary with color parameters
    :return: ColorValues instance

    Example:
        >>> d = {"brightness": 1.2, "temperature": 0.3}
        >>> values = color_from_dict(d)
    """
    valid_fields = {
        "brightness", "contrast", "gamma", "saturation", "vibrance",
        "temperature", "shadows", "highlights", "hue_shift"
    }
    kwargs = {k: v for k, v in d.items() if k in valid_fields}
    return ColorValues(**kwargs)


def filter_from_dict(d: dict) -> FilterValues:
    """Create FilterValues from dictionary.

    :param d: Dictionary with filter parameters
    :return: FilterValues instance
    """
    valid_fields = {
        "min_opacity", "max_opacity", "min_scale", "max_scale",
        "sphere_radius", "sphere_center", "box_min", "box_max"
    }
    kwargs = {}
    for k, v in d.items():
        if k not in valid_fields:
            continue
        if k in ("sphere_center", "box_min", "box_max") and v is not None:
            kwargs[k] = tuple(v)
        else:
            kwargs[k] = v
    return FilterValues(**kwargs)


def transform_from_dict(d: dict) -> TransformValues:
    """Create TransformValues from dictionary.

    Supports both direct field assignment and factory method syntax.

    Example:
        >>> # Direct
        >>> d = {"scale": 2.0, "translation": [1, 0, 0]}
        >>> values = transform_from_dict(d)

        >>> # Factory method
        >>> d = {"from_rotation_euler": [0, 45, 0]}
        >>> values = transform_from_dict(d)
    """
    # Check for factory methods
    if "from_scale" in d:
        return TransformValues.from_scale(d["from_scale"])
    if "from_translation" in d:
        return TransformValues.from_translation(*d["from_translation"])
    if "from_rotation_euler" in d:
        return TransformValues.from_rotation_euler(*d["from_rotation_euler"])
    if "from_euler_rad" in d:
        return TransformValues.from_euler_rad(*d["from_euler_rad"])
    if "from_rotation_axis_angle" in d:
        axis, angle = d["from_rotation_axis_angle"]
        return TransformValues.from_rotation_axis_angle(tuple(axis), angle)
    if "from_axis_angle_rad" in d:
        axis, angle = d["from_axis_angle_rad"]
        return TransformValues.from_axis_angle_rad(tuple(axis), angle)

    # Direct assignment
    kwargs = {}
    if "scale" in d:
        kwargs["scale"] = d["scale"]
    if "rotation" in d:
        kwargs["rotation"] = tuple(d["rotation"])
    if "translation" in d:
        kwargs["translation"] = tuple(d["translation"])

    return TransformValues(**kwargs)


def load_color_json(path: str | Path) -> ColorValues:
    """Load ColorValues from JSON file.

    :param path: Path to JSON file
    :return: ColorValues instance
    """
    with open(path) as f:
        d = json.load(f)
    return color_from_dict(d)


def load_filter_json(path: str | Path) -> FilterValues:
    """Load FilterValues from JSON file.

    :param path: Path to JSON file
    :return: FilterValues instance
    """
    with open(path) as f:
        d = json.load(f)
    return filter_from_dict(d)


def load_transform_json(path: str | Path) -> TransformValues:
    """Load TransformValues from JSON file.

    :param path: Path to JSON file
    :return: TransformValues instance
    """
    with open(path) as f:
        d = json.load(f)
    return transform_from_dict(d)


# ============================================================================
# Saving Functions
# ============================================================================


def color_to_dict(values: ColorValues) -> dict:
    """Convert ColorValues to dictionary.

    :param values: ColorValues instance
    :return: Dictionary representation
    """
    return {
        "brightness": values.brightness,
        "contrast": values.contrast,
        "gamma": values.gamma,
        "saturation": values.saturation,
        "vibrance": values.vibrance,
        "temperature": values.temperature,
        "shadows": values.shadows,
        "highlights": values.highlights,
        "hue_shift": values.hue_shift,
    }


def filter_to_dict(values: FilterValues) -> dict:
    """Convert FilterValues to dictionary.

    :param values: FilterValues instance
    :return: Dictionary representation
    """
    d = {
        "min_opacity": values.min_opacity,
        "max_opacity": values.max_opacity,
        "min_scale": values.min_scale,
        "max_scale": values.max_scale,
        "sphere_radius": values.sphere_radius,
        "sphere_center": list(values.sphere_center),
    }
    if values.box_min is not None:
        d["box_min"] = list(values.box_min)
        d["box_max"] = list(values.box_max)
    return d


def transform_to_dict(values: TransformValues) -> dict:
    """Convert TransformValues to dictionary.

    :param values: TransformValues instance
    :return: Dictionary representation
    """
    return {
        "scale": values.scale,
        "rotation": list(values.rotation),
        "translation": list(values.translation),
    }


def save_color_json(values: ColorValues, path: str | Path) -> None:
    """Save ColorValues to JSON file.

    :param values: ColorValues instance
    :param path: Output path
    """
    with open(path, "w") as f:
        json.dump(color_to_dict(values), f, indent=2)


def save_filter_json(values: FilterValues, path: str | Path) -> None:
    """Save FilterValues to JSON file.

    :param values: FilterValues instance
    :param path: Output path
    """
    with open(path, "w") as f:
        json.dump(filter_to_dict(values), f, indent=2)


def save_transform_json(values: TransformValues, path: str | Path) -> None:
    """Save TransformValues to JSON file.

    :param values: TransformValues instance
    :param path: Output path
    """
    with open(path, "w") as f:
        json.dump(transform_to_dict(values), f, indent=2)
