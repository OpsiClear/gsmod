"""Preset library for gsmod operations.

Provides pre-configured value objects for common use cases,
with support for loading from dict and JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from gsmod.config.values import ColorValues, FilterValues, OpacityValues, TransformValues

logger = logging.getLogger(__name__)

# ============================================================================
# Color Presets - Basic
# ============================================================================

WARM = ColorValues.from_k(3200)
COOL = ColorValues.from_k(8500)
NEUTRAL = ColorValues()

CINEMATIC = ColorValues.from_k(4500) + ColorValues(
    contrast=1.1, saturation=0.9, shadows=0.1, highlights=-0.05
)

VIBRANT = ColorValues(saturation=1.3, vibrance=1.2, brightness=1.05)

MUTED = ColorValues(saturation=0.7, vibrance=0.8, contrast=0.95)

DRAMATIC = ColorValues(contrast=1.25, shadows=0.15, highlights=-0.1)

VINTAGE = ColorValues.from_k(4000) + ColorValues(saturation=0.8, contrast=0.9, gamma=1.1)

GOLDEN_HOUR = ColorValues.from_k(3000) + ColorValues(brightness=1.1, saturation=1.1, shadows=0.05)

MOONLIGHT = ColorValues.from_k(9000) + ColorValues(brightness=0.8, contrast=1.1, saturation=0.85)

# ============================================================================
# Color Presets - Film Stock Emulation
# ============================================================================

KODAK_PORTRA = ColorValues.from_k(5200) + ColorValues(
    saturation=1.05,
    contrast=0.95,
    shadows=0.08,
    highlights=-0.03,
    shadow_tint_hue=-20,
    shadow_tint_sat=0.15,
)

FUJI_VELVIA = ColorValues(
    saturation=1.4,
    vibrance=1.3,
    contrast=1.15,
    brightness=1.02,
    shadows=0.1,
)

KODAK_EKTACHROME = ColorValues.from_k(5800) + ColorValues(
    saturation=1.2,
    contrast=1.1,
    brightness=1.05,
    highlights=-0.05,
)

ILFORD_HP5 = ColorValues(
    saturation=0.0,
    contrast=1.2,
    gamma=1.05,
    shadows=0.12,
    highlights=-0.08,
)

CINESTILL_800T = ColorValues.from_k(3200) + ColorValues(
    saturation=1.15,
    contrast=0.95,
    highlights=-0.1,
    highlight_tint_hue=180,
    highlight_tint_sat=0.25,
)

# ============================================================================
# Color Presets - Seasonal Looks
# ============================================================================

SPRING_FRESH = ColorValues.from_k(6000) + ColorValues(
    brightness=1.08,
    saturation=1.15,
    vibrance=1.2,
    shadows=0.05,
    shadow_tint_hue=120,
    shadow_tint_sat=0.1,
)

SUMMER_BRIGHT = ColorValues.from_k(6500) + ColorValues(
    brightness=1.12,
    saturation=1.25,
    vibrance=1.15,
    contrast=1.05,
    fade=0.05,
)

AUTUMN_WARM = ColorValues.from_k(3500) + ColorValues(
    saturation=1.2,
    contrast=1.08,
    shadows=0.08,
    hue_shift=10,
)

WINTER_COLD = ColorValues.from_k(8000) + ColorValues(
    brightness=0.95,
    saturation=0.85,
    contrast=1.15,
    shadows=0.1,
    tint=-0.1,
)

# ============================================================================
# Color Presets - Time of Day
# ============================================================================

SUNRISE = ColorValues.from_k(2800) + ColorValues(
    brightness=1.05,
    saturation=1.15,
    shadows=0.15,
    highlights=-0.05,
    shadow_tint_hue=-140,
    shadow_tint_sat=0.2,
)

MIDDAY_SUN = ColorValues.from_k(5500) + ColorValues(
    brightness=1.1,
    contrast=1.1,
    saturation=1.05,
)

SUNSET = ColorValues.from_k(2500) + ColorValues(
    brightness=1.08,
    saturation=1.2,
    shadows=0.1,
    highlight_tint_hue=25,
    highlight_tint_sat=0.3,
)

BLUE_HOUR = ColorValues.from_k(12000) + ColorValues(
    brightness=0.85,
    saturation=1.15,
    contrast=1.05,
    tint=-0.15,
)

OVERCAST = ColorValues.from_k(7000) + ColorValues(
    saturation=0.9,
    contrast=0.95,
    brightness=0.98,
)

# ============================================================================
# Color Presets - Artistic Styles
# ============================================================================

HIGH_KEY = ColorValues(
    brightness=1.15,
    contrast=0.85,
    saturation=0.9,
    fade=0.15,
    highlights=-0.1,
)

LOW_KEY = ColorValues(
    brightness=0.85,
    contrast=1.35,
    saturation=0.95,
    shadows=0.2,
    highlights=-0.15,
)

TEAL_ORANGE = ColorValues(
    saturation=1.2,
    contrast=1.1,
    shadow_tint_hue=180,
    shadow_tint_sat=0.3,
    highlight_tint_hue=30,
    highlight_tint_sat=0.25,
)

BLEACH_BYPASS = ColorValues(
    saturation=0.5,
    contrast=1.3,
    brightness=1.05,
    highlights=-0.15,
)

CROSS_PROCESS = ColorValues(
    saturation=1.4,
    contrast=1.2,
    hue_shift=15,
    shadow_tint_hue=120,
    shadow_tint_sat=0.2,
    highlight_tint_hue=-120,
    highlight_tint_sat=0.15,
)

FADED_PRINT = ColorValues(
    saturation=0.75,
    contrast=0.9,
    fade=0.12,
    gamma=1.08,
)

SEPIA_TONE = ColorValues(
    saturation=0.2,
    temperature=0.4,
    tint=0.1,
    contrast=0.95,
    gamma=1.05,
)

# ============================================================================
# Color Presets - Technical Adjustments
# ============================================================================

LIFT_SHADOWS = ColorValues(
    shadows=0.2,
    brightness=1.02,
)

COMPRESS_HIGHLIGHTS = ColorValues(
    highlights=-0.2,
    contrast=1.05,
)

BOOST_MIDTONES = ColorValues(
    gamma=0.9,
    contrast=1.05,
)

INCREASE_CONTRAST = ColorValues(
    contrast=1.3,
    shadows=0.1,
    highlights=-0.1,
)

DECREASE_CONTRAST = ColorValues(
    contrast=0.8,
    gamma=1.1,
)

DESATURATE_MILD = ColorValues(
    saturation=0.7,
)

DESATURATE_STRONG = ColorValues(
    saturation=0.3,
)

ENHANCE_COLORS = ColorValues(
    saturation=1.25,
    vibrance=1.3,
    contrast=1.08,
)

# ============================================================================
# Opacity Presets
# ============================================================================

FADE_SUBTLE = OpacityValues.fade(0.9)
FADE_MILD = OpacityValues.fade(0.8)
FADE_MODERATE = OpacityValues.fade(0.7)
FADE_STRONG = OpacityValues.fade(0.5)
FADE_HEAVY = OpacityValues.fade(0.3)

BOOST_SUBTLE = OpacityValues.boost(1.1)
BOOST_MILD = OpacityValues.boost(1.2)
BOOST_MODERATE = OpacityValues.boost(1.3)
BOOST_STRONG = OpacityValues.boost(1.5)

GHOST_EFFECT = OpacityValues.fade(0.2)
TRANSLUCENT = OpacityValues.fade(0.6)
SEMI_TRANSPARENT = OpacityValues.fade(0.4)

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
    # Basic
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
    # Film Stock
    "kodak_portra": KODAK_PORTRA,
    "fuji_velvia": FUJI_VELVIA,
    "kodak_ektachrome": KODAK_EKTACHROME,
    "ilford_hp5": ILFORD_HP5,
    "cinestill_800t": CINESTILL_800T,
    # Seasonal
    "spring_fresh": SPRING_FRESH,
    "summer_bright": SUMMER_BRIGHT,
    "autumn_warm": AUTUMN_WARM,
    "winter_cold": WINTER_COLD,
    # Time of Day
    "sunrise": SUNRISE,
    "midday_sun": MIDDAY_SUN,
    "sunset": SUNSET,
    "blue_hour": BLUE_HOUR,
    "overcast": OVERCAST,
    # Artistic
    "high_key": HIGH_KEY,
    "low_key": LOW_KEY,
    "teal_orange": TEAL_ORANGE,
    "bleach_bypass": BLEACH_BYPASS,
    "cross_process": CROSS_PROCESS,
    "faded_print": FADED_PRINT,
    "sepia_tone": SEPIA_TONE,
    # Technical
    "lift_shadows": LIFT_SHADOWS,
    "compress_highlights": COMPRESS_HIGHLIGHTS,
    "boost_midtones": BOOST_MIDTONES,
    "increase_contrast": INCREASE_CONTRAST,
    "decrease_contrast": DECREASE_CONTRAST,
    "desaturate_mild": DESATURATE_MILD,
    "desaturate_strong": DESATURATE_STRONG,
    "enhance_colors": ENHANCE_COLORS,
}

OPACITY_PRESETS: dict[str, OpacityValues] = {
    # Fade
    "fade_subtle": FADE_SUBTLE,
    "fade_mild": FADE_MILD,
    "fade_moderate": FADE_MODERATE,
    "fade_strong": FADE_STRONG,
    "fade_heavy": FADE_HEAVY,
    # Boost
    "boost_subtle": BOOST_SUBTLE,
    "boost_mild": BOOST_MILD,
    "boost_moderate": BOOST_MODERATE,
    "boost_strong": BOOST_STRONG,
    # Effects
    "ghost_effect": GHOST_EFFECT,
    "translucent": TRANSLUCENT,
    "semi_transparent": SEMI_TRANSPARENT,
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
    :returns: ColorValues preset
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
    :returns: FilterValues preset
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
    :returns: TransformValues preset
    :raises KeyError: If preset not found
    """
    name_lower = name.lower()
    if name_lower not in TRANSFORM_PRESETS:
        available = ", ".join(TRANSFORM_PRESETS.keys())
        raise KeyError(f"Unknown transform preset '{name}'. Available: {available}")
    return TRANSFORM_PRESETS[name_lower]


def get_opacity_preset(name: str) -> OpacityValues:
    """Get opacity preset by name.

    :param name: Preset name (case-insensitive)
    :returns: OpacityValues preset
    :raises KeyError: If preset not found
    """
    name_lower = name.lower()
    if name_lower not in OPACITY_PRESETS:
        available = ", ".join(OPACITY_PRESETS.keys())
        raise KeyError(f"Unknown opacity preset '{name}'. Available: {available}")
    return OPACITY_PRESETS[name_lower]


# ============================================================================
# Dict/JSON Loading
# ============================================================================


def color_from_dict(d: dict) -> ColorValues:
    """Create ColorValues from dictionary.

    :param d: Dictionary with color parameters
    :returns: ColorValues instance

    Example:
        >>> d = {"brightness": 1.2, "temperature": 0.3}
        >>> values = color_from_dict(d)
    """
    valid_fields = {
        "brightness",
        "contrast",
        "gamma",
        "saturation",
        "vibrance",
        "temperature",
        "shadows",
        "highlights",
        "hue_shift",
    }
    kwargs = {k: v for k, v in d.items() if k in valid_fields}
    return ColorValues(**kwargs)


def filter_from_dict(d: dict) -> FilterValues:
    """Create FilterValues from dictionary.

    :param d: Dictionary with filter parameters
    :returns: FilterValues instance
    """
    valid_fields = {
        "min_opacity",
        "max_opacity",
        "min_scale",
        "max_scale",
        "sphere_radius",
        "sphere_center",
        "box_min",
        "box_max",
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
    :returns: ColorValues instance
    """
    with open(path) as f:
        d = json.load(f)
    return color_from_dict(d)


def load_filter_json(path: str | Path) -> FilterValues:
    """Load FilterValues from JSON file.

    :param path: Path to JSON file
    :returns: FilterValues instance
    """
    with open(path) as f:
        d = json.load(f)
    return filter_from_dict(d)


def load_transform_json(path: str | Path) -> TransformValues:
    """Load TransformValues from JSON file.

    :param path: Path to JSON file
    :returns: TransformValues instance
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
    :returns: Dictionary representation
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
    :returns: Dictionary representation
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
    :returns: Dictionary representation
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
