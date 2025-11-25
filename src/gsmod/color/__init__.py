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
"""

from gsmod.color.pipeline import Color
from gsmod.color.presets import ColorPreset

__all__ = ["Color", "ColorPreset"]
