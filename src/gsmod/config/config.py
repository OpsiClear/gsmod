"""Unified gsmod configuration.

This module provides a top-level configuration dataclass that contains
all process-specific configurations (color, filter, transform) as
sub-attributes.
"""

from __future__ import annotations

from dataclasses import dataclass

from gsmod.config.color import ColorConfig
from gsmod.config.filter import FilterConfig
from gsmod.config.transform import TransformConfig


@dataclass(frozen=True)
class GsproConfig:
    """Top-level configuration containing all pipeline configurations.

    Provides hierarchical access to all operation specifications:
        CONFIG.color.brightness
        CONFIG.filter.min_opacity
        CONFIG.transform.scale_factor

    Attributes:
        color: Color pipeline operation specifications
        filter: Filter pipeline operation specifications
        transform: Transform pipeline operation specifications
    """

    color: ColorConfig = ColorConfig()
    filter: FilterConfig = FilterConfig()
    transform: TransformConfig = TransformConfig()

    def get_all_specs(self) -> dict[str, dict[str, any]]:
        """Get all operation specs organized by process.

        :return: Nested dictionary of all specifications
        """
        return {
            "color": self.color.get_all_specs(),
            "filter": self.filter.get_all_specs(),
            "transform": self.transform.get_all_specs(),
        }


# Main singleton instance
CONFIG = GsproConfig()

# Backward-compatible singleton exports
COLOR_CONFIG = CONFIG.color
FILTER_CONFIG = CONFIG.filter
TRANSFORM_CONFIG = CONFIG.transform
