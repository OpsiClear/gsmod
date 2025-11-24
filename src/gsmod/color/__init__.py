"""
Color processing module.

Provides high-performance color adjustments with LUT compilation
and pre-configured presets for common color grading looks.
"""

from gsmod.color.pipeline import Color
from gsmod.color.presets import ColorPreset

__all__ = ["Color", "ColorPreset"]
