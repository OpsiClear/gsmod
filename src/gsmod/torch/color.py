"""GPU-accelerated color processing pipeline for Gaussian Splatting."""

from __future__ import annotations

import logging

import torch

from gsmod.torch.gstensor_pro import GSTensorPro

logger = logging.getLogger(__name__)


class ColorGPU:
    """GPU-accelerated color adjustment pipeline using PyTorch.

    Provides chainable color operations that run entirely on GPU with massive
    parallelism. All operations return self for fluent chaining.

    Format Contract:
        - Input: GSTensorPro in any format (SH or RGB)
        - Processing: ALL operations run in RGB space [0, 1]
        - Output: RGB by default, or original format if restore_format=True

    This ensures CPU/GPU equivalence and correct color math.

    Performance:
        - 10-100x faster than CPU ColorLUT
        - Zero CPU-GPU transfer during processing
        - Batched operations minimize kernel launches

    Example:
        >>> from gsmod.torch import ColorGPU
        >>> pipeline = (
        ...     ColorGPU()
        ...     .brightness(1.2)
        ...     .contrast(1.1)
        ...     .saturation(1.3)
        ...     .temperature(0.1)
        ... )
        >>> result = pipeline(gstensor, inplace=True)
    """

    def __init__(self):
        """Initialize color pipeline."""
        self._operations = []

    def brightness(self, factor: float = 1.0) -> ColorGPU:
        """Add brightness adjustment.

        :param factor: Brightness factor (1.0 = no change, >1.0 = brighter, <1.0 = darker)
        :returns: Self for chaining

        Example:
            >>> pipeline = ColorGPU().brightness(1.2)
        """
        self._operations.append(("brightness", factor))
        return self

    def contrast(self, factor: float = 1.0) -> ColorGPU:
        """Add contrast adjustment.

        :param factor: Contrast factor (1.0 = no change, >1.0 = more contrast, <1.0 = less)
        :returns: Self for chaining

        Example:
            >>> pipeline = ColorGPU().contrast(1.1)
        """
        self._operations.append(("contrast", factor))
        return self

    def saturation(self, factor: float = 1.0) -> ColorGPU:
        """Add saturation adjustment.

        :param factor: Saturation factor (1.0 = no change, 0 = grayscale, >1.0 = more saturated)
        :returns: Self for chaining

        Example:
            >>> pipeline = ColorGPU().saturation(1.3)
        """
        self._operations.append(("saturation", factor))
        return self

    def gamma(self, value: float = 1.0) -> ColorGPU:
        """Add gamma correction.

        :param value: Gamma value (1.0 = no change, <1.0 = brighter, >1.0 = darker)
        :returns: Self for chaining

        Example:
            >>> pipeline = ColorGPU().gamma(0.8)
        """
        self._operations.append(("gamma", value))
        return self

    def temperature(self, temp: float = 0.0) -> ColorGPU:
        """Add temperature adjustment.

        :param temp: Temperature (-1.0 = cold/blue, 0 = neutral, 1.0 = warm/orange)
        :returns: Self for chaining

        Example:
            >>> pipeline = ColorGPU().temperature(0.2)
        """
        self._operations.append(("temperature", temp))
        return self

    def vibrance(self, factor: float = 1.0) -> ColorGPU:
        """Add vibrance adjustment.

        :param factor: Vibrance factor (1.0 = no change, >1.0 = more vibrant)
        :returns: Self for chaining

        Example:
            >>> pipeline = ColorGPU().vibrance(1.2)
        """
        self._operations.append(("vibrance", factor))
        return self

    def hue_shift(self, degrees: float = 0.0) -> ColorGPU:
        """Add hue shift.

        :param degrees: Hue shift in degrees (-180 to 180)
        :returns: Self for chaining

        Example:
            >>> pipeline = ColorGPU().hue_shift(30)
        """
        self._operations.append(("hue_shift", degrees))
        return self

    def shadows(self, factor: float = 0.0) -> ColorGPU:
        """Adjust shadows (dark areas).

        :param factor: Shadow adjustment (-1.0 = darker, 0 = no change, 1.0 = lighter)
        :returns: Self for chaining

        Example:
            >>> pipeline = ColorGPU().shadows(0.2)
        """
        self._operations.append(("shadows", factor))
        return self

    def highlights(self, factor: float = 0.0) -> ColorGPU:
        """Adjust highlights (bright areas).

        :param factor: Highlight adjustment (-1.0 = darker, 0 = no change, 1.0 = lighter)
        :returns: Self for chaining

        Example:
            >>> pipeline = ColorGPU().highlights(-0.1)
        """
        self._operations.append(("highlights", factor))
        return self

    def tint(self, value: float = 0.0) -> ColorGPU:
        """Add green/magenta tint (white balance complement to temperature).

        :param value: Tint value (-1.0 = green, 0 = neutral, 1.0 = magenta)
        :returns: Self for chaining

        Example:
            >>> pipeline = ColorGPU().temperature(0.2).tint(-0.1)  # Warm with slight green
        """
        self._operations.append(("tint", value))
        return self

    def fade(self, value: float = 0.0) -> ColorGPU:
        """Add black point lift for film/matte look.

        :param value: Fade amount (0.0 = no change, 1.0 = full lift)
        :returns: Self for chaining

        Example:
            >>> pipeline = ColorGPU().fade(0.1)  # Subtle film look
        """
        self._operations.append(("fade", value))
        return self

    def shadow_tint(self, hue: float, saturation: float) -> ColorGPU:
        """Add color tint to shadow regions (split toning).

        :param hue: Tint hue in degrees (-180 to 180)
        :param saturation: Tint intensity (0.0 = none, 1.0 = full)
        :returns: Self for chaining

        Example:
            >>> pipeline = ColorGPU().shadow_tint(220, 0.3)  # Blue shadows
        """
        self._operations.append(("shadow_tint", (hue, saturation)))
        return self

    def highlight_tint(self, hue: float, saturation: float) -> ColorGPU:
        """Add color tint to highlight regions (split toning).

        :param hue: Tint hue in degrees (-180 to 180)
        :param saturation: Tint intensity (0.0 = none, 1.0 = full)
        :returns: Self for chaining

        Example:
            >>> pipeline = ColorGPU().highlight_tint(40, 0.2)  # Warm highlights
        """
        self._operations.append(("highlight_tint", (hue, saturation)))
        return self

    def preset(self, name: str, strength: float = 1.0) -> ColorGPU:
        """Apply color preset.

        :param name: Preset name ("cinematic", "warm", "cool", "vibrant", "muted", "dramatic")
        :param strength: Preset strength (0.0 to 1.0)
        :returns: Self for chaining

        Example:
            >>> pipeline = ColorGPU().preset("cinematic", strength=0.8)
        """
        self._operations.append(("preset", (name, strength)))
        return self

    def __call__(self, data: GSTensorPro, inplace: bool = True, restore_format: bool = False) -> GSTensorPro:
        """Apply color pipeline to GSTensorPro.

        Format Contract:
            - Input: Any format (SH or RGB)
            - Processing: Always in RGB space
            - Output: RGB by default, or original format if restore_format=True

        :param data: GSTensorPro object to process
        :param inplace: If True, modify data in-place; if False, create copy
        :param restore_format: If True, restore original format after processing
        :returns: Processed GSTensorPro

        Example:
            >>> pipeline = ColorGPU().brightness(1.2).saturation(1.3)
            >>> result = pipeline(gstensor, inplace=True)
        """
        if not isinstance(data, GSTensorPro):
            raise TypeError(f"Expected GSTensorPro, got {type(data)}")

        # Fast path: no operations
        if not self._operations:
            return data if inplace else data.clone()

        # Create copy if not inplace
        if not inplace:
            data = data.clone()
            # Deep copy _format since clone() may do shallow copy
            if hasattr(data, '_format'):
                data._format = data._format.copy()

        # Track original format for potential restoration (use gsply 0.2.8 property)
        original_is_sh = data.is_sh0_sh

        # ALWAYS convert to RGB for color operations (ensures CPU/GPU equivalence)
        # Use gsply 0.2.8 format query property
        if not data.is_sh0_rgb:
            logger.debug("[ColorGPU] Converting to RGB format for color operations")
            data.to_rgb(inplace=True)

        # Apply operations in FIXED ORDER to match CPU Color pipeline
        # CPU enforces: Phase 1 (LUT) -> Phase 2 (post-LUT)
        # Phase 1: temperature, brightness, contrast, gamma
        # Phase 2: saturation, vibrance, hue_shift, shadows, highlights, preset

        # Collect operations by type
        ops_by_type = {
            "temperature": [], "tint": [], "brightness": [], "contrast": [], "gamma": [],
            "saturation": [], "vibrance": [], "hue_shift": [],
            "shadows": [], "highlights": [], "fade": [],
            "shadow_tint": [], "highlight_tint": [], "preset": []
        }
        for op_name, op_value in self._operations:
            if op_name in ops_by_type:
                ops_by_type[op_name].append(op_value)

        # Phase 1: Apply LUT operations in fixed order (temperature, tint, brightness, contrast, gamma)
        for temp in ops_by_type["temperature"]:
            data.adjust_temperature(temp, inplace=True)
        for tint_val in ops_by_type["tint"]:
            self._apply_tint(data, tint_val)
        for bright in ops_by_type["brightness"]:
            data.adjust_brightness(bright, inplace=True)
        for contrast in ops_by_type["contrast"]:
            data.adjust_contrast(contrast, inplace=True)
        for gamma in ops_by_type["gamma"]:
            data.adjust_gamma(gamma, inplace=True)

        # Phase 2: Apply post-LUT operations in fixed order
        for sat in ops_by_type["saturation"]:
            data.adjust_saturation(sat, inplace=True)
        for vib in ops_by_type["vibrance"]:
            data.adjust_vibrance(vib, inplace=True)
        for hue in ops_by_type["hue_shift"]:
            data.adjust_hue_shift(hue, inplace=True)
        for shadow in ops_by_type["shadows"]:
            self._apply_shadows(data, shadow)
        for highlight in ops_by_type["highlights"]:
            self._apply_highlights(data, highlight)
        for fade_val in ops_by_type["fade"]:
            self._apply_fade(data, fade_val)
        for tint_args in ops_by_type["shadow_tint"]:
            hue, sat = tint_args
            self._apply_shadow_tint(data, hue, sat)
        for tint_args in ops_by_type["highlight_tint"]:
            hue, sat = tint_args
            self._apply_highlight_tint(data, hue, sat)
        for preset_args in ops_by_type["preset"]:
            name, strength = preset_args
            data.apply_color_preset(name, strength, inplace=True)

        # Restore original format if requested
        if restore_format and original_is_sh:
            logger.debug("[ColorGPU] Restoring to SH format")
            data.to_sh(inplace=True)

        logger.debug(
            "[ColorGPU] Applied %d operations to %d Gaussians",
            len(self._operations), len(data)
        )

        return data

    def requires_rgb(self) -> bool:
        """Check if any operation in the pipeline requires RGB format.

        Note: All color operations now require RGB format for correct
        color math and CPU/GPU equivalence. This method always returns
        True if there are any operations.

        :returns: True if there are operations, False otherwise

        Example:
            >>> pipeline = ColorGPU().brightness(1.2).saturation(1.3)
            >>> pipeline.requires_rgb()  # True
        """
        return len(self._operations) > 0

    def _apply_shadows(self, data: GSTensorPro, factor: float):
        """Apply shadow adjustment (GPU-optimized).

        :param data: GSTensorPro to modify in-place
        :param factor: Shadow factor (-1.0 to 1.0)
        """
        # Calculate luminance
        luminance = torch.sum(
            data.sh0 * torch.tensor([0.299, 0.587, 0.114], device=data.device),
            dim=-1, keepdim=True
        )

        # Create shadow mask (inverse luminance)
        shadow_mask = 1.0 - luminance

        # Apply adjustment weighted by shadow mask
        adjustment = 1.0 + factor * 0.5 * shadow_mask
        data.sh0 *= adjustment
        data.sh0.clamp_(0, 1)

    def _apply_highlights(self, data: GSTensorPro, factor: float):
        """Apply highlight adjustment (GPU-optimized).

        :param data: GSTensorPro to modify in-place
        :param factor: Highlight factor (-1.0 to 1.0)
        """
        # Calculate luminance
        luminance = torch.sum(
            data.sh0 * torch.tensor([0.299, 0.587, 0.114], device=data.device),
            dim=-1, keepdim=True
        )

        # Create highlight mask (luminance)
        highlight_mask = luminance

        # Apply adjustment weighted by highlight mask
        adjustment = 1.0 + factor * 0.5 * highlight_mask
        data.sh0 *= adjustment
        data.sh0.clamp_(0, 1)

    def _apply_tint(self, data: GSTensorPro, value: float):
        """Apply green/magenta tint (GPU-optimized).

        :param data: GSTensorPro to modify in-place
        :param value: Tint value (-1.0 = green, 1.0 = magenta)
        """
        # Tint: negative = green boost, positive = magenta (reduce green)
        tint_offset_g = -value * 0.1
        tint_offset_rb = value * 0.05

        data.sh0[..., 0] += tint_offset_rb  # R
        data.sh0[..., 1] += tint_offset_g   # G
        data.sh0[..., 2] += tint_offset_rb  # B
        data.sh0.clamp_(0, 1)

    def _apply_fade(self, data: GSTensorPro, value: float):
        """Apply black point lift for film/matte look (GPU-optimized).

        :param data: GSTensorPro to modify in-place
        :param value: Fade amount (0.0 to 1.0)
        """
        # Fade: output = fade + input * (1 - fade)
        if value > 0:
            data.sh0.mul_(1.0 - value).add_(value)
            data.sh0.clamp_(0, 1)

    def _apply_shadow_tint(self, data: GSTensorPro, hue: float, saturation: float):
        """Apply color tint to shadow regions (GPU-optimized).

        :param data: GSTensorPro to modify in-place
        :param hue: Tint hue in degrees
        :param saturation: Tint intensity (0.0 to 1.0)
        """
        if saturation <= 0:
            return

        # Calculate luminance
        luminance = torch.sum(
            data.sh0 * torch.tensor([0.299, 0.587, 0.114], device=data.device),
            dim=-1, keepdim=True
        )

        # Shadow mask: smooth falloff (1 - lum*2) clamped
        shadow_mask = torch.clamp(1.0 - luminance * 2.0, 0.0, 1.0)

        # Convert hue to RGB offset
        import math
        hue_rad = hue * (math.pi / 180.0)
        tint_r = math.cos(hue_rad) * 0.5
        tint_g = math.cos(hue_rad - 2.0 * math.pi / 3.0) * 0.5
        tint_b = math.cos(hue_rad - 4.0 * math.pi / 3.0) * 0.5

        # Apply tint weighted by shadow mask and saturation
        tint_strength = shadow_mask * saturation
        data.sh0[..., 0] += tint_r * tint_strength.squeeze(-1)
        data.sh0[..., 1] += tint_g * tint_strength.squeeze(-1)
        data.sh0[..., 2] += tint_b * tint_strength.squeeze(-1)
        data.sh0.clamp_(0, 1)

    def _apply_highlight_tint(self, data: GSTensorPro, hue: float, saturation: float):
        """Apply color tint to highlight regions (GPU-optimized).

        :param data: GSTensorPro to modify in-place
        :param hue: Tint hue in degrees
        :param saturation: Tint intensity (0.0 to 1.0)
        """
        if saturation <= 0:
            return

        # Calculate luminance
        luminance = torch.sum(
            data.sh0 * torch.tensor([0.299, 0.587, 0.114], device=data.device),
            dim=-1, keepdim=True
        )

        # Highlight mask: smooth falloff (lum*2 - 1) clamped
        highlight_mask = torch.clamp(luminance * 2.0 - 1.0, 0.0, 1.0)

        # Convert hue to RGB offset
        import math
        hue_rad = hue * (math.pi / 180.0)
        tint_r = math.cos(hue_rad) * 0.5
        tint_g = math.cos(hue_rad - 2.0 * math.pi / 3.0) * 0.5
        tint_b = math.cos(hue_rad - 4.0 * math.pi / 3.0) * 0.5

        # Apply tint weighted by highlight mask and saturation
        tint_strength = highlight_mask * saturation
        data.sh0[..., 0] += tint_r * tint_strength.squeeze(-1)
        data.sh0[..., 1] += tint_g * tint_strength.squeeze(-1)
        data.sh0[..., 2] += tint_b * tint_strength.squeeze(-1)
        data.sh0.clamp_(0, 1)

    def reset(self) -> ColorGPU:
        """Reset pipeline, removing all operations.

        :returns: Self for chaining

        Example:
            >>> pipeline.reset().brightness(1.1)  # Clear and start fresh
        """
        self._operations = []
        return self

    def clone(self) -> ColorGPU:
        """Create a copy of this pipeline.

        :returns: New ColorGPU with same operations

        Example:
            >>> pipeline2 = pipeline1.clone()
            >>> pipeline2.contrast(1.2)  # Doesn't affect pipeline1
        """
        new_pipeline = ColorGPU()
        new_pipeline._operations = self._operations.copy()
        return new_pipeline