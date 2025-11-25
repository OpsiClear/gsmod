"""Configuration module for gsmod operations.

This module provides standardized parameter specifications for all
pipeline operations, ensuring consistency between CPU and GPU backends.

Usage:
    # Unified access
    from gsmod.config import CONFIG
    CONFIG.color.brightness.neutral  # 1.0
    CONFIG.filter.min_opacity.default  # 0.0

    # Backward-compatible access
    from gsmod.config import COLOR_CONFIG, FILTER_CONFIG
    COLOR_CONFIG.brightness.neutral  # 1.0
"""

from gsmod.config.color import ColorConfig
from gsmod.config.config import (
    COLOR_CONFIG,
    CONFIG,
    FILTER_CONFIG,
    TRANSFORM_CONFIG,
    GsproConfig,
)
from gsmod.config.filter import FilterConfig
from gsmod.config.operations import OperationSpec
from gsmod.config.presets import (
    # Film Stock
    AUTUMN_WARM,
    BLEACH_BYPASS,
    BLUE_HOUR,
    # Opacity presets
    BOOST_MILD,
    BOOST_MODERATE,
    CINEMATIC,
    CINESTILL_800T,
    CLEANUP_FILTER,
    # Artistic
    COMPRESS_HIGHLIGHTS,
    COOL,
    CROSS_PROCESS,
    DESATURATE_MILD,
    # Transform presets
    DOUBLE_SIZE,
    DRAMATIC,
    ENHANCE_COLORS,
    FADE_MILD,
    FADE_MODERATE,
    FADED_PRINT,
    FLIP_X,
    FLIP_Y,
    FLIP_Z,
    FUJI_VELVIA,
    GHOST_EFFECT,
    GOLDEN_HOUR,
    HALF_SIZE,
    HIGH_KEY,
    # Technical
    ILFORD_HP5,
    INCREASE_CONTRAST,
    KODAK_EKTACHROME,
    KODAK_PORTRA,
    LIFT_SHADOWS,
    LOW_KEY,
    # Time of Day
    MIDDAY_SUN,
    MOONLIGHT,
    MUTED,
    NEUTRAL,
    OPACITY_PRESETS,
    OVERCAST,
    QUALITY_FILTER,
    # Seasonal
    SEPIA_TONE,
    SPRING_FRESH,
    # Filter presets
    STRICT_FILTER,
    SUMMER_BRIGHT,
    SUNRISE,
    SUNSET,
    TEAL_ORANGE,
    TRANSLUCENT,
    VIBRANT,
    VINTAGE,
    # Color presets - Basic
    WARM,
    WINTER_COLD,
    color_from_dict,
    color_to_dict,
    filter_from_dict,
    filter_to_dict,
    # Loading functions
    get_color_preset,
    get_filter_preset,
    get_opacity_preset,
    get_transform_preset,
    load_color_json,
    load_filter_json,
    load_transform_json,
    save_color_json,
    save_filter_json,
    save_transform_json,
    transform_from_dict,
    transform_to_dict,
)
from gsmod.config.transform import TransformConfig
from gsmod.config.values import ColorValues, FilterValues, OpacityValues, TransformValues

__all__ = [
    # Core types
    "OperationSpec",
    "GsproConfig",
    "ColorConfig",
    "FilterConfig",
    "TransformConfig",
    # Value classes (with __add__ merge support)
    "ColorValues",
    "FilterValues",
    "TransformValues",
    "OpacityValues",
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
    # Color presets - Film Stock
    "KODAK_PORTRA",
    "FUJI_VELVIA",
    "KODAK_EKTACHROME",
    "ILFORD_HP5",
    "CINESTILL_800T",
    # Color presets - Seasonal
    "SPRING_FRESH",
    "SUMMER_BRIGHT",
    "AUTUMN_WARM",
    "WINTER_COLD",
    # Color presets - Time of Day
    "SUNRISE",
    "MIDDAY_SUN",
    "SUNSET",
    "BLUE_HOUR",
    "OVERCAST",
    # Color presets - Artistic
    "HIGH_KEY",
    "LOW_KEY",
    "TEAL_ORANGE",
    "BLEACH_BYPASS",
    "CROSS_PROCESS",
    "FADED_PRINT",
    "SEPIA_TONE",
    # Color presets - Technical
    "LIFT_SHADOWS",
    "COMPRESS_HIGHLIGHTS",
    "INCREASE_CONTRAST",
    "DESATURATE_MILD",
    "ENHANCE_COLORS",
    # Opacity presets
    "FADE_MILD",
    "FADE_MODERATE",
    "BOOST_MILD",
    "BOOST_MODERATE",
    "GHOST_EFFECT",
    "TRANSLUCENT",
    "OPACITY_PRESETS",
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
    # Loading functions
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
    "save_color_json",
    "save_filter_json",
    "save_transform_json",
    "color_to_dict",
    "filter_to_dict",
    "transform_to_dict",
    # Singletons
    "CONFIG",
    "COLOR_CONFIG",
    "FILTER_CONFIG",
    "TRANSFORM_CONFIG",
]
