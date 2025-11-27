"""
Color processing module - RGB color adjustments via compiled LUT pipelines.

Performance (CPU, 1M colors):
  - Color pipeline: 0.92ms (1,091M colors/sec)
  - LUT compilation: 0.003ms (one-time cost)
  - Zero-copy operation with inplace=True

Example:
    >>> from gsmod.color import Color
    >>> pipeline = Color().brightness(1.2).saturation(1.3)
    >>> result = pipeline(data, inplace=True)

Auto-correction (Photoshop/Lightroom-style):
    >>> from gsmod.color import auto_enhance, auto_contrast
    >>> result = auto_enhance(data, strength=0.8)
    >>> data.color(result.to_color_values())
"""

from gsmod.color.auto import (
    AutoCorrectionResult,
    auto_contrast,
    auto_enhance,
    auto_exposure,
    auto_white_balance,
    compute_optimal_parameters,
)
from gsmod.color.pipeline import Color
from gsmod.color.presets import ColorPreset

__all__ = [
    "Color",
    "ColorPreset",
    # Auto-correction
    "AutoCorrectionResult",
    "auto_contrast",
    "auto_exposure",
    "auto_white_balance",
    "auto_enhance",
    "compute_optimal_parameters",
]
