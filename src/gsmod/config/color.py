"""Color operation configuration.

This module defines the standardized parameter specifications for all
color operations, ensuring consistency between CPU and GPU pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass

from gsmod.config.operations import OperationSpec


@dataclass(frozen=True)
class ColorConfig:
    """Configuration for all color pipeline operations.

    All parameters use standardized ranges that work identically
    for both CPU (Color) and GPU (ColorGPU) pipelines.
    """

    temperature: OperationSpec = OperationSpec(
        name="temperature",
        min_value=-1.0,
        max_value=1.0,
        default=0.0,
        neutral=0.0,
        composition="additive",
        description="Color temperature: -1=cool/blue, 0=neutral, 1=warm/orange",
    )

    brightness: OperationSpec = OperationSpec(
        name="brightness",
        min_value=0.0,
        max_value=5.0,
        default=1.0,
        neutral=1.0,
        composition="multiplicative",
        description="Brightness multiplier: 1.0=no change",
    )

    contrast: OperationSpec = OperationSpec(
        name="contrast",
        min_value=0.0,
        max_value=5.0,
        default=1.0,
        neutral=1.0,
        composition="multiplicative",
        description="Contrast multiplier: 1.0=no change",
    )

    gamma: OperationSpec = OperationSpec(
        name="gamma",
        min_value=0.1,
        max_value=5.0,
        default=1.0,
        neutral=1.0,
        composition="multiplicative",
        description="Gamma correction: 1.0=linear, <1=brighter, >1=darker",
    )

    saturation: OperationSpec = OperationSpec(
        name="saturation",
        min_value=0.0,
        max_value=5.0,
        default=1.0,
        neutral=1.0,
        composition="multiplicative",
        description="Saturation: 0=grayscale, 1.0=no change",
    )

    vibrance: OperationSpec = OperationSpec(
        name="vibrance",
        min_value=0.0,
        max_value=5.0,
        default=1.0,
        neutral=1.0,
        composition="multiplicative",
        description="Smart saturation: 1.0=no change",
    )

    hue_shift: OperationSpec = OperationSpec(
        name="hue_shift",
        min_value=-180.0,
        max_value=180.0,
        default=0.0,
        neutral=0.0,
        composition="additive",
        description="Hue rotation in degrees",
    )

    shadows: OperationSpec = OperationSpec(
        name="shadows",
        min_value=-1.0,
        max_value=1.0,
        default=0.0,
        neutral=0.0,
        composition="additive",
        description="Shadow adjustment: -1=darker, 0=neutral, 1=lighter",
    )

    highlights: OperationSpec = OperationSpec(
        name="highlights",
        min_value=-1.0,
        max_value=1.0,
        default=0.0,
        neutral=0.0,
        composition="additive",
        description="Highlight adjustment: -1=darker, 0=neutral, 1=lighter",
    )

    tint: OperationSpec = OperationSpec(
        name="tint",
        min_value=-1.0,
        max_value=1.0,
        default=0.0,
        neutral=0.0,
        composition="additive",
        description="Tint: -1=green, 0=neutral, 1=magenta",
    )

    fade: OperationSpec = OperationSpec(
        name="fade",
        min_value=0.0,
        max_value=1.0,
        default=0.0,
        neutral=0.0,
        composition="additive",
        description="Black point lift: 0=none, 1=full (film/matte look)",
    )

    shadow_tint_hue: OperationSpec = OperationSpec(
        name="shadow_tint_hue",
        min_value=-180.0,
        max_value=180.0,
        default=0.0,
        neutral=0.0,
        composition="additive",
        description="Shadow tint hue in degrees",
    )

    shadow_tint_sat: OperationSpec = OperationSpec(
        name="shadow_tint_sat",
        min_value=0.0,
        max_value=1.0,
        default=0.0,
        neutral=0.0,
        composition="additive",
        description="Shadow tint intensity: 0=none, 1=full",
    )

    highlight_tint_hue: OperationSpec = OperationSpec(
        name="highlight_tint_hue",
        min_value=-180.0,
        max_value=180.0,
        default=0.0,
        neutral=0.0,
        composition="additive",
        description="Highlight tint hue in degrees",
    )

    highlight_tint_sat: OperationSpec = OperationSpec(
        name="highlight_tint_sat",
        min_value=0.0,
        max_value=1.0,
        default=0.0,
        neutral=0.0,
        composition="additive",
        description="Highlight tint intensity: 0=none, 1=full",
    )

    def get_spec(self, name: str) -> OperationSpec:
        """Get operation spec by name.

        :param name: Operation name
        :return: OperationSpec for the operation
        :raises AttributeError: If operation not found
        """
        return getattr(self, name)

    def get_all_specs(self) -> dict[str, OperationSpec]:
        """Get all operation specs as a dictionary.

        :return: Dictionary mapping operation names to specs
        """
        return {
            "temperature": self.temperature,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "gamma": self.gamma,
            "saturation": self.saturation,
            "vibrance": self.vibrance,
            "hue_shift": self.hue_shift,
            "shadows": self.shadows,
            "highlights": self.highlights,
            "tint": self.tint,
            "fade": self.fade,
            "shadow_tint_hue": self.shadow_tint_hue,
            "shadow_tint_sat": self.shadow_tint_sat,
            "highlight_tint_hue": self.highlight_tint_hue,
            "highlight_tint_sat": self.highlight_tint_sat,
        }


# Singleton instance for use throughout the codebase
COLOR_CONFIG = ColorConfig()
