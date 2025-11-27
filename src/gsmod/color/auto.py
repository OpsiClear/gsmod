"""Automatic color correction algorithms.

Implements industry-standard auto-correction algorithms inspired by:
- Photoshop Auto Contrast/Color/Tone
- Lightroom Auto Settings
- iOS Photos Auto Enhance

These algorithms analyze the input image and compute optimal adjustments
without requiring a target histogram.

References:
- Adobe Photoshop Auto options: 0.1% shadow/highlight clipping
- Gray World white balance assumption
- 18% gray (0.18 linear, ~0.45 gamma) midtone target
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gsmod.gsdata_pro import GSDataPro

logger = logging.getLogger(__name__)


@dataclass
class AutoCorrectionResult:
    """Result of automatic color correction analysis.

    Contains computed adjustments that can be applied to ColorValues.
    """

    # Exposure/brightness adjustment
    exposure: float = 0.0  # EV stops (-2 to +2)
    brightness: float = 1.0  # Multiplier

    # Contrast
    contrast: float = 1.0  # Multiplier
    blacks: float = 0.0  # Black point lift
    whites: float = 0.0  # White point adjustment

    # White balance
    temperature: float = 0.0  # -1 to 1
    tint: float = 0.0  # -1 to 1

    # Per-channel gains (for white balance)
    r_gain: float = 1.0
    g_gain: float = 1.0
    b_gain: float = 1.0

    def to_color_values(self):
        """Convert to ColorValues for application.

        :returns: ColorValues with computed adjustments
        """
        from gsmod.config.values import ColorValues

        return ColorValues(
            brightness=self.brightness,
            contrast=self.contrast,
            temperature=self.temperature,
            tint=self.tint,
        )


def auto_contrast(
    data: GSDataPro,
    clip_percent: float = 0.1,
    per_channel: bool = False,
) -> AutoCorrectionResult:
    """Compute automatic contrast adjustment using percentile stretching.

    This is the algorithm behind Photoshop's Auto Contrast:
    - Clips the darkest and brightest pixels (default 0.1%)
    - Stretches the histogram to use full dynamic range

    :param data: GSDataPro with sh0 colors [N, 3]
    :param clip_percent: Percentage of pixels to clip at each end (0.1 = 0.1%)
    :param per_channel: If True, stretch each channel independently (like Auto Levels)
    :returns: AutoCorrectionResult with contrast adjustments

    Example:
        >>> result = auto_contrast(data, clip_percent=0.1)
        >>> data.color(result.to_color_values())
    """
    colors = data.sh0
    if colors is None or len(colors) == 0:
        return AutoCorrectionResult()

    # Flatten to analyze
    if per_channel:
        # Per-channel: stretch each channel independently
        # This can introduce color casts but maximizes contrast
        gains = []
        offsets = []
        for c in range(3):
            channel = colors[:, c]
            low = np.percentile(channel, clip_percent)
            high = np.percentile(channel, 100 - clip_percent)
            range_val = high - low
            if range_val > 0.01:
                gains.append(1.0 / range_val)
                offsets.append(-low / range_val)
            else:
                gains.append(1.0)
                offsets.append(0.0)

        # Convert to approximate brightness/contrast
        avg_gain = np.mean(gains)
        return AutoCorrectionResult(
            contrast=float(avg_gain),
            r_gain=float(gains[0]),
            g_gain=float(gains[1]),
            b_gain=float(gains[2]),
        )
    else:
        # Monochromatic: use luminance, preserve color relationships
        # This is what Photoshop's Auto Contrast does
        luminance = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]

        low = np.percentile(luminance, clip_percent)
        high = np.percentile(luminance, 100 - clip_percent)
        current_range = high - low

        if current_range < 0.01:
            return AutoCorrectionResult()

        # Target: stretch to [0, 1]
        # contrast = 1 / current_range
        # But we need to be careful not to over-stretch
        target_range = 0.95  # Leave some headroom
        contrast = target_range / current_range
        contrast = np.clip(contrast, 0.5, 3.0)  # Reasonable bounds

        # Black point: how much to lift
        blacks = -low * contrast

        return AutoCorrectionResult(
            contrast=float(contrast),
            blacks=float(np.clip(blacks, -0.2, 0.2)),
        )


def auto_exposure(
    data: GSDataPro,
    target_midtone: float = 0.45,
    clip_percent: float = 1.0,
) -> AutoCorrectionResult:
    """Compute automatic exposure adjustment.

    Adjusts exposure so the average midtone reaches the target value.
    Based on the 18% gray principle (0.18 linear = ~0.45 gamma-corrected).

    :param data: GSDataPro with sh0 colors [N, 3]
    :param target_midtone: Target for average luminance (default 0.45 = 18% gray)
    :param clip_percent: Exclude top/bottom percentile from average
    :returns: AutoCorrectionResult with exposure adjustment

    Example:
        >>> result = auto_exposure(data, target_midtone=0.45)
        >>> data.color(result.to_color_values())
    """
    colors = data.sh0
    if colors is None or len(colors) == 0:
        return AutoCorrectionResult()

    # Compute luminance
    luminance = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]

    # Exclude extremes (like Photoshop's clip percentage)
    low_thresh = np.percentile(luminance, clip_percent)
    high_thresh = np.percentile(luminance, 100 - clip_percent)
    mask = (luminance >= low_thresh) & (luminance <= high_thresh)

    if not np.any(mask):
        return AutoCorrectionResult()

    # Current midtone (average of non-clipped pixels)
    current_midtone = np.mean(luminance[mask])

    if current_midtone < 0.01:
        return AutoCorrectionResult()

    # Compute brightness multiplier to reach target
    brightness = target_midtone / current_midtone
    brightness = np.clip(brightness, 0.5, 2.5)  # Reasonable bounds

    # Convert to EV stops for reference
    exposure_ev = np.log2(brightness)

    return AutoCorrectionResult(
        exposure=float(exposure_ev),
        brightness=float(brightness),
    )


def auto_white_balance(
    data: GSDataPro,
    clip_percent: float = 1.0,
    method: str = "gray_world",
) -> AutoCorrectionResult:
    """Compute automatic white balance using Gray World assumption.

    The Gray World algorithm assumes the average color should be neutral gray.
    This is the basis for many auto white balance implementations.

    :param data: GSDataPro with sh0 colors [N, 3]
    :param clip_percent: Exclude extremes from average calculation
    :param method: "gray_world" or "white_patch"
    :returns: AutoCorrectionResult with white balance adjustments

    Example:
        >>> result = auto_white_balance(data)
        >>> print(f"Temperature: {result.temperature}, Tint: {result.tint}")
    """
    colors = data.sh0
    if colors is None or len(colors) == 0:
        return AutoCorrectionResult()

    # Compute luminance for clipping
    luminance = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]

    # Exclude over/underexposed pixels
    low_thresh = np.percentile(luminance, clip_percent)
    high_thresh = np.percentile(luminance, 100 - clip_percent)
    mask = (luminance >= low_thresh) & (luminance <= high_thresh)

    if not np.any(mask):
        return AutoCorrectionResult()

    if method == "gray_world":
        # Gray World: average color should be gray
        avg_r = np.mean(colors[mask, 0])
        avg_g = np.mean(colors[mask, 1])
        avg_b = np.mean(colors[mask, 2])

        # Target: all channels equal to overall average
        avg_all = (avg_r + avg_g + avg_b) / 3

        if avg_all < 0.01:
            return AutoCorrectionResult()

        # Compute gains to neutralize
        r_gain = avg_all / (avg_r + 1e-8)
        g_gain = avg_all / (avg_g + 1e-8)
        b_gain = avg_all / (avg_b + 1e-8)

        # Clip to reasonable range
        r_gain = np.clip(r_gain, 0.5, 2.0)
        g_gain = np.clip(g_gain, 0.5, 2.0)
        b_gain = np.clip(b_gain, 0.5, 2.0)

        # Convert to temperature/tint (approximate)
        # Temperature: R/B ratio, Tint: G vs R+B
        rb_ratio = r_gain / (b_gain + 1e-8)
        temperature = (rb_ratio - 1.0) * 0.5  # Map to [-1, 1] range
        temperature = np.clip(temperature, -1.0, 1.0)

        g_vs_rb = g_gain / ((r_gain + b_gain) / 2 + 1e-8)
        tint = (1.0 - g_vs_rb) * 0.5  # Map to [-1, 1] range
        tint = np.clip(tint, -1.0, 1.0)

    elif method == "white_patch":
        # White Patch: brightest pixels should be white
        bright_thresh = np.percentile(luminance, 95)
        bright_mask = luminance >= bright_thresh

        if not np.any(bright_mask):
            return AutoCorrectionResult()

        # Average of brightest pixels
        max_r = np.mean(colors[bright_mask, 0])
        max_g = np.mean(colors[bright_mask, 1])
        max_b = np.mean(colors[bright_mask, 2])

        max_val = max(max_r, max_g, max_b)
        if max_val < 0.01:
            return AutoCorrectionResult()

        r_gain = max_val / (max_r + 1e-8)
        g_gain = max_val / (max_g + 1e-8)
        b_gain = max_val / (max_b + 1e-8)

        r_gain = np.clip(r_gain, 0.5, 2.0)
        g_gain = np.clip(g_gain, 0.5, 2.0)
        b_gain = np.clip(b_gain, 0.5, 2.0)

        rb_ratio = r_gain / (b_gain + 1e-8)
        temperature = (rb_ratio - 1.0) * 0.5
        temperature = np.clip(temperature, -1.0, 1.0)

        g_vs_rb = g_gain / ((r_gain + b_gain) / 2 + 1e-8)
        tint = (1.0 - g_vs_rb) * 0.5
        tint = np.clip(tint, -1.0, 1.0)

    else:
        raise ValueError(f"Unknown method: {method}")

    return AutoCorrectionResult(
        temperature=float(temperature),
        tint=float(tint),
        r_gain=float(r_gain),
        g_gain=float(g_gain),
        b_gain=float(b_gain),
    )


def auto_enhance(
    data: GSDataPro,
    strength: float = 1.0,
    preserve_warmth: bool = True,
) -> AutoCorrectionResult:
    """Compute comprehensive automatic enhancement.

    Combines exposure, contrast, and optional white balance for
    a natural-looking enhancement. Similar to iOS Photos Auto Enhance.

    :param data: GSDataPro with sh0 colors [N, 3]
    :param strength: How strong to apply corrections (0-1)
    :param preserve_warmth: If True, don't fully neutralize warm tones
    :returns: AutoCorrectionResult with combined adjustments

    Example:
        >>> result = auto_enhance(data, strength=0.8)
        >>> data.color(result.to_color_values())
    """
    # Get individual corrections
    exposure_result = auto_exposure(data, target_midtone=0.45)
    contrast_result = auto_contrast(data, clip_percent=0.5)
    wb_result = auto_white_balance(data, method="gray_world")

    # Blend based on strength
    brightness = 1.0 + (exposure_result.brightness - 1.0) * strength
    contrast = 1.0 + (contrast_result.contrast - 1.0) * strength * 0.7  # Less aggressive

    # White balance: preserve some original warmth if requested
    if preserve_warmth:
        temperature = wb_result.temperature * strength * 0.5  # Only half correction
        tint = wb_result.tint * strength * 0.5
    else:
        temperature = wb_result.temperature * strength
        tint = wb_result.tint * strength

    return AutoCorrectionResult(
        brightness=float(np.clip(brightness, 0.5, 2.0)),
        contrast=float(np.clip(contrast, 0.7, 1.5)),
        temperature=float(temperature),
        tint=float(tint),
    )


def compute_optimal_parameters(
    data: GSDataPro,
    target_mean: float | None = None,
    target_std: float | None = None,
    preserve_contrast: bool = True,
    min_contrast_ratio: float = 0.8,
) -> AutoCorrectionResult:
    """Compute optimal color parameters to match targets while preserving quality.

    Unlike histogram matching, this finds the minimal adjustments needed
    to reach target statistics while preserving the original image character.

    :param data: GSDataPro with sh0 colors [N, 3]
    :param target_mean: Target mean luminance (None = keep original)
    :param target_std: Target std luminance (None = keep original)
    :param preserve_contrast: If True, don't reduce contrast below threshold
    :param min_contrast_ratio: Minimum allowed contrast vs original
    :returns: AutoCorrectionResult with minimal adjustments

    Example:
        >>> # Match exposure of reference while keeping contrast
        >>> result = compute_optimal_parameters(
        ...     data,
        ...     target_mean=0.5,
        ...     preserve_contrast=True
        ... )
    """
    colors = data.sh0
    if colors is None or len(colors) == 0:
        return AutoCorrectionResult()

    luminance = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
    current_mean = np.mean(luminance)
    current_std = np.std(luminance)

    brightness = 1.0
    contrast = 1.0

    # Exposure adjustment
    if target_mean is not None and current_mean > 0.01:
        brightness = target_mean / current_mean
        brightness = np.clip(brightness, 0.5, 2.0)

    # Contrast adjustment
    if target_std is not None and current_std > 0.01:
        contrast = target_std / current_std

        # Preserve contrast: don't go below minimum ratio
        if preserve_contrast:
            min_contrast = min_contrast_ratio
            contrast = max(contrast, min_contrast)

        contrast = np.clip(contrast, 0.5, 2.0)

    return AutoCorrectionResult(
        brightness=float(brightness),
        contrast=float(contrast),
    )
