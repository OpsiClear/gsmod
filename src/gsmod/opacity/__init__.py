"""Opacity adjustment module for Gaussian Splatting.

Provides format-aware opacity scaling that works with both
linear [0, 1] and PLY (logit) opacity formats.
"""

from gsmod.opacity.apply import apply_opacity_values

__all__ = ["apply_opacity_values"]
