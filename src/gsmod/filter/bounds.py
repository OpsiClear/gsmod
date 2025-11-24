"""
Scene bounds calculation for filtering.

Provides utilities for calculating spatial bounds of Gaussian splat scenes.
Based on Universal 4D Viewer TransformService.calculate_scene_bounds().
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from gsply import GSData


@dataclass
class SceneBounds:
    """
    Container for scene spatial bounds.

    Attributes:
        min: Minimum point [x, y, z]
        max: Maximum point [x, y, z]
        sizes: Scene dimensions [width, height, depth] (computed)
        max_size: Largest dimension (computed)
        center: Scene center point [x, y, z] (computed)
    """

    min: np.ndarray
    max: np.ndarray
    sizes: np.ndarray = field(init=False, repr=False)
    max_size: float = field(init=False)
    center: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Calculate derived attributes from min and max bounds."""
        # Ensure arrays are float32
        self.min = np.asarray(self.min, dtype=np.float32)
        self.max = np.asarray(self.max, dtype=np.float32)

        # Compute derived values
        self.sizes = self.max - self.min
        self.max_size = float(np.max(self.sizes))
        self.center = (self.min + self.max) * 0.5


def calculate_scene_bounds(data: GSData | np.ndarray) -> SceneBounds:
    """
    Calculate spatial bounds of a Gaussian splat scene.

    Uses NumPy's optimized min/max which is already highly tuned.
    Testing showed NumPy is competitive with custom Numba for this operation.

    :param data: GSData object or positions array [N, 3] in format [x, y, z]
    :return: SceneBounds object containing min, max, sizes, max_size, center

    Example:
        >>> # With GSData
        >>> from gsply import GSData
        >>> bounds = calculate_scene_bounds(data)
        >>> print(f"Scene size: {bounds.sizes}")
        >>> print(f"Center: {bounds.center}")
        >>>
        >>> # With array (backwards compatible)
        >>> positions = np.random.randn(1000, 3)
        >>> bounds = calculate_scene_bounds(positions)

    Note:
        - Input should be in world coordinates
        - Performance: ~1.5ms for 1M Gaussians
        - Returns float32 for memory efficiency
    """
    # Extract positions from GSData or use array directly
    if isinstance(data, GSData):
        positions = data.means
    else:
        positions = np.asarray(data, dtype=np.float32)

    if len(positions) == 0:
        raise ValueError("positions cannot be empty")

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be [N, 3], got shape {positions.shape}")

    # Use NumPy's optimized min/max (competitive with custom Numba)
    min_point = positions.min(axis=0)
    max_point = positions.max(axis=0)

    return SceneBounds(min_point, max_point)


def calculate_recommended_max_scale(data: GSData | np.ndarray, percentile: float = 99.5) -> float:
    """
    Calculate recommended maximum scale threshold using percentiles.

    Uses Numba-optimized max calculation for 2x speedup.

    :param data: GSData object or scales array [N, 3] in format [scale_x, scale_y, scale_z]
    :param percentile: Percentile to use (default: 99.5)
    :return: Recommended max_scale threshold

    Example:
        >>> # With GSData
        >>> from gsply import GSData
        >>> threshold = calculate_recommended_max_scale(data)
        >>> print(f"Recommended max_scale: {threshold:.4f}")
        >>>
        >>> # With array (backwards compatible)
        >>> scales = np.random.rand(1000, 3) * 2.0
        >>> threshold = calculate_recommended_max_scale(scales)

    Note:
        - Run on first frame for good defaults
        - 99.5th percentile filters top 0.5% of scales
        - Adjust percentile based on data quality
        - Performance: ~12ms for 1M Gaussians (optimized with Numba)
    """
    from gsmod.filter.kernels import calculate_max_scales_numba

    # Extract scales from GSData or use array directly
    if isinstance(data, GSData):
        scales = data.scales
    else:
        scales = np.asarray(data, dtype=np.float32)

    if len(scales) == 0:
        raise ValueError("scales cannot be empty")

    if scales.ndim != 2 or scales.shape[1] != 3:
        raise ValueError(f"scales must be [N, 3], got shape {scales.shape}")

    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be between 0 and 100, got {percentile}")

    # Get maximum scale per Gaussian with Numba optimization
    max_scales = np.empty(len(scales), dtype=np.float32)
    calculate_max_scales_numba(scales, max_scales)

    # Calculate percentile (NumPy is fast enough for percentile)
    threshold = float(np.percentile(max_scales, percentile))

    return threshold
