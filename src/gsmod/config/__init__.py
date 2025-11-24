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

from gsmod.config.operations import OperationSpec
from gsmod.config.color import ColorConfig
from gsmod.config.filter import FilterConfig
from gsmod.config.transform import TransformConfig
from gsmod.config.config import (
    GsproConfig,
    CONFIG,
    COLOR_CONFIG,
    FILTER_CONFIG,
    TRANSFORM_CONFIG,
)
from gsmod.config.values import ColorValues, FilterValues, TransformValues
from gsmod.config.presets import (
    # Color presets
    WARM, COOL, NEUTRAL, CINEMATIC, VIBRANT, MUTED, DRAMATIC, VINTAGE, GOLDEN_HOUR, MOONLIGHT,
    # Filter presets
    STRICT_FILTER, QUALITY_FILTER, CLEANUP_FILTER,
    # Transform presets
    DOUBLE_SIZE, HALF_SIZE, FLIP_X, FLIP_Y, FLIP_Z,
    # Loading functions
    get_color_preset, get_filter_preset, get_transform_preset,
    color_from_dict, filter_from_dict, transform_from_dict,
    load_color_json, load_filter_json, load_transform_json,
    save_color_json, save_filter_json, save_transform_json,
    color_to_dict, filter_to_dict, transform_to_dict,
)

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
    # Color presets
    "WARM", "COOL", "NEUTRAL", "CINEMATIC", "VIBRANT", "MUTED",
    "DRAMATIC", "VINTAGE", "GOLDEN_HOUR", "MOONLIGHT",
    # Filter presets
    "STRICT_FILTER", "QUALITY_FILTER", "CLEANUP_FILTER",
    # Transform presets
    "DOUBLE_SIZE", "HALF_SIZE", "FLIP_X", "FLIP_Y", "FLIP_Z",
    # Loading functions
    "get_color_preset", "get_filter_preset", "get_transform_preset",
    "color_from_dict", "filter_from_dict", "transform_from_dict",
    "load_color_json", "load_filter_json", "load_transform_json",
    "save_color_json", "save_filter_json", "save_transform_json",
    "color_to_dict", "filter_to_dict", "transform_to_dict",
    # Singletons
    "CONFIG",
    "COLOR_CONFIG",
    "FILTER_CONFIG",
    "TRANSFORM_CONFIG",
]
