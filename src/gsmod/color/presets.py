"""
ColorPreset: Pre-configured color adjustment presets.

Provides commonly used color grading looks for easy application to Gaussian
Splatting data or any RGB color arrays.
"""

from __future__ import annotations

import numpy as np

from gsmod.color.pipeline import Color


class ColorPreset:
    """
    Pre-configured color adjustment presets.

    Provides commonly used color grading looks:
        preset = ColorPreset.cinematic()
        colors = preset.apply(colors)
    """

    def __init__(
        self,
        temperature: float = 0.5,
        brightness: float = 1.0,
        contrast: float = 1.0,
        gamma: float = 1.0,
        saturation: float = 1.0,
        shadows: float = 1.0,
        highlights: float = 1.0,
    ):
        """
        Create custom preset.

        :param temperature: Color temperature (0=cool, 0.5=neutral, 1=warm)
        :param brightness: Brightness multiplier
        :param contrast: Contrast multiplier
        :param gamma: Gamma correction
        :param saturation: Saturation adjustment
        :param shadows: Shadow boost/reduce
        :param highlights: Highlight boost/reduce
        """
        self.params = {
            "temperature": temperature,
            "brightness": brightness,
            "contrast": contrast,
            "gamma": gamma,
            "saturation": saturation,
            "shadows": shadows,
            "highlights": highlights,
        }
        # Create the color pipeline with the preset parameters
        self._pipeline = (
            Color()
            .temperature(temperature)
            .brightness(brightness)
            .contrast(contrast)
            .gamma(gamma)
            .saturation(saturation)
            .shadows(shadows)
            .highlights(highlights)
        )

    def apply(self, colors: np.ndarray, inplace: bool = True) -> np.ndarray:
        """
        Apply preset to colors.

        :param colors: RGB colors [N, 3] in range [0, 1]
        :param inplace: If True, modifies colors in-place
        :return: Adjusted colors [N, 3]
        """
        return self._pipeline.apply(colors, inplace=inplace)

    def to_pipeline(self) -> Color:
        """
        Convert preset to Color pipeline for further composition.

        :return: Color pipeline with this preset's adjustments
        """
        return (
            Color()
            .temperature(self.params["temperature"])
            .brightness(self.params["brightness"])
            .contrast(self.params["contrast"])
            .gamma(self.params["gamma"])
            .saturation(self.params["saturation"])
            .shadows(self.params["shadows"])
            .highlights(self.params["highlights"])
        )

    @classmethod
    def neutral(cls) -> ColorPreset:
        """Identity transformation (no changes)."""
        return cls()

    @classmethod
    def cinematic(cls) -> ColorPreset:
        """
        Cinematic look: desaturated, high contrast, teal shadows, orange highlights.
        """
        return cls(
            temperature=0.55,  # Slightly warm
            brightness=1.0,
            contrast=1.2,  # Increased contrast
            gamma=0.95,
            saturation=0.85,  # Desaturated
            shadows=1.1,  # Lift shadows (teal look)
            highlights=0.95,  # Compress highlights
        )

    @classmethod
    def warm(cls) -> ColorPreset:
        """
        Warm sunset look: orange tones, boosted brightness.
        """
        return cls(
            temperature=0.75,  # Very warm
            brightness=1.15,
            contrast=1.1,
            gamma=0.9,
            saturation=1.2,  # Boosted saturation
            shadows=1.15,
            highlights=0.9,
        )

    @classmethod
    def cool(cls) -> ColorPreset:
        """
        Cool blue look: blue tones, crisp contrast.
        """
        return cls(
            temperature=0.25,  # Very cool
            brightness=1.0,
            contrast=1.15,
            gamma=1.0,
            saturation=1.1,
            shadows=1.0,
            highlights=0.95,
        )

    @classmethod
    def vibrant(cls) -> ColorPreset:
        """
        Vibrant look: boosted saturation and contrast.
        """
        return cls(
            temperature=0.5,
            brightness=1.1,
            contrast=1.2,
            gamma=0.95,
            saturation=1.4,  # High saturation
            shadows=1.1,
            highlights=0.9,
        )

    @classmethod
    def muted(cls) -> ColorPreset:
        """
        Muted/pastel look: low saturation, lifted shadows.
        """
        return cls(
            temperature=0.5,
            brightness=1.05,
            contrast=0.95,  # Reduced contrast
            gamma=1.0,
            saturation=0.7,  # Low saturation
            shadows=1.2,  # Lifted shadows
            highlights=0.95,
        )

    @classmethod
    def dramatic(cls) -> ColorPreset:
        """
        Dramatic look: high contrast, crushed shadows, blown highlights.
        """
        return cls(
            temperature=0.5,
            brightness=1.0,
            contrast=1.4,  # Very high contrast
            gamma=0.85,
            saturation=1.1,
            shadows=0.8,  # Crushed shadows
            highlights=1.1,  # Blown highlights
        )

    def __repr__(self) -> str:
        """String representation of the preset."""
        active_params = [
            (k, v)
            for k, v in self.params.items()
            if (k != "temperature" and v != 1.0) or (k == "temperature" and v != 0.5)
        ]
        param_str = ", ".join(f"{k}={v:.2f}" for k, v in active_params)
        return f"ColorPreset({param_str})" if param_str else "ColorPreset(neutral)"
