"""Format utilities wrapping gsply's format API.

This module provides utilities for working with format-aware objects
using gsply's public format API instead of directly accessing _format dict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    pass


T = TypeVar("T")


def ensure_format_copied(dest: Any, src: Any) -> None:
    """Ensure format is properly copied from source to destination.

    Uses gsply's public copy_format_from() method when available.
    Falls back to direct _format dict copy for compatibility.

    :param dest: Destination object to copy format to
    :param src: Source object to copy format from

    Example:
        >>> processed = transform(data)  # Format may be lost
        >>> ensure_format_copied(processed, data)  # Restore format
    """
    if hasattr(src, "copy_format_from"):
        # Use public API (preferred)
        dest.copy_format_from(src)
    elif hasattr(src, "_format") and hasattr(dest, "_format"):
        # Fallback to direct dict copy
        dest._format = dict(src._format)


def copy_format_dict(src: Any) -> dict:
    """Get a copy of the format dict from a format-aware object.

    Uses gsply's public format_state property when available.

    :param src: Source object with format tracking
    :returns: Copy of format dict

    Example:
        >>> fmt = copy_format_dict(data)
        >>> # Later...
        >>> dest._format = fmt
    """
    if hasattr(src, "format_state"):
        return src.format_state
    if hasattr(src, "_format"):
        return dict(src._format)
    return {}


def is_ply_format(data: Any) -> bool:
    """Check if data is in PLY format (log-scales, logit-opacities).

    :param data: FormatAware object
    :returns: True if both scales and opacities are in PLY format
    """
    return (
        hasattr(data, "is_scales_ply")
        and hasattr(data, "is_opacities_ply")
        and data.is_scales_ply
        and data.is_opacities_ply
    )


def is_linear_format(data: Any) -> bool:
    """Check if data is in linear format.

    :param data: FormatAware object
    :returns: True if both scales and opacities are in linear format
    """
    return (
        hasattr(data, "is_scales_linear")
        and hasattr(data, "is_opacities_linear")
        and data.is_scales_linear
        and data.is_opacities_linear
    )


def ensure_rgb[T](data: T, inplace: bool = True) -> T:
    """Ensure data is in RGB color format.

    Converts sh0 from SH format to RGB if needed.

    :param data: FormatAware object with to_rgb() method
    :param inplace: If True, modify in-place; if False, return copy
    :returns: Data with sh0 in RGB format

    Example:
        >>> data = ensure_rgb(data)  # Now sh0 contains RGB colors
    """
    if hasattr(data, "is_sh0_rgb") and not data.is_sh0_rgb:
        if hasattr(data, "to_rgb"):
            return data.to_rgb(inplace=inplace)
    return data


def ensure_sh[T](data: T, inplace: bool = True) -> T:
    """Ensure data is in SH color format.

    Converts sh0 from RGB format to SH if needed.

    :param data: FormatAware object with to_sh() method
    :param inplace: If True, modify in-place; if False, return copy
    :returns: Data with sh0 in SH format

    Example:
        >>> data = ensure_sh(data)  # Now sh0 contains SH coefficients
    """
    if hasattr(data, "is_sh0_rgb") and data.is_sh0_rgb:
        if hasattr(data, "to_sh"):
            return data.to_sh(inplace=inplace)
    return data


def ensure_linear[T](data: T, inplace: bool = True) -> T:
    """Ensure data is in linear format (linear scales, linear opacities).

    Converts from PLY format if needed.

    :param data: FormatAware object with denormalize() method
    :param inplace: If True, modify in-place; if False, return copy
    :returns: Data with linear scales and opacities

    Example:
        >>> data = ensure_linear(data)  # Now scales/opacities are linear
    """
    if hasattr(data, "is_opacities_ply") and data.is_opacities_ply:
        if hasattr(data, "denormalize"):
            return data.denormalize(inplace=inplace)
    return data


def ensure_ply[T](data: T, inplace: bool = True) -> T:
    """Ensure data is in PLY format (log-scales, logit-opacities).

    Converts from linear format if needed.

    :param data: FormatAware object with normalize() method
    :param inplace: If True, modify in-place; if False, return copy
    :returns: Data with PLY-format scales and opacities

    Example:
        >>> data = ensure_ply(data)  # Now ready for PLY file writing
    """
    if hasattr(data, "is_opacities_linear") and data.is_opacities_linear:
        if hasattr(data, "normalize"):
            return data.normalize(inplace=inplace)
    return data


def get_opacity_for_threshold(data: Any, linear_threshold: float) -> float:
    """Get opacity threshold in the correct format for the data.

    Converts a linear threshold [0, 1] to the appropriate format
    (logit for PLY format, linear for linear format).

    :param data: FormatAware object
    :param linear_threshold: Threshold in linear [0, 1] space
    :returns: Threshold in data's opacity format

    Example:
        >>> threshold = get_opacity_for_threshold(data, 0.5)
        >>> mask = data.opacities > threshold
    """
    import numpy as np

    if hasattr(data, "is_opacities_ply") and data.is_opacities_ply:
        # Convert to logit space: logit(p) = log(p / (1-p))
        p = np.clip(linear_threshold, 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))
    return linear_threshold


def get_scale_for_threshold(data: Any, linear_threshold: float) -> float:
    """Get scale threshold in the correct format for the data.

    Converts a linear threshold to the appropriate format
    (log for PLY format, linear for linear format).

    :param data: FormatAware object
    :param linear_threshold: Threshold in linear scale space
    :returns: Threshold in data's scale format

    Example:
        >>> threshold = get_scale_for_threshold(data, 0.01)
        >>> max_scales = np.max(data.scales, axis=1)
        >>> mask = max_scales < threshold
    """
    import numpy as np

    if hasattr(data, "is_scales_ply") and data.is_scales_ply:
        # Convert to log space
        return np.log(max(linear_threshold, 1e-9))
    return linear_threshold
