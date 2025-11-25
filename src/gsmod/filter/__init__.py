"""
Gaussian splat filtering module - Spatial and quality-based filtering.

Performance (CPU, 100K Gaussians):
  - Filter pipeline: 2.2ms (46M Gaussians/sec)
  - Volume filters: sphere, box, ellipsoid, frustum
  - Quality filters: opacity, scale thresholds
  - Invert mode: Include/exclude filtering support

Example:
    >>> from gsmod import GSDataPro, FilterValues
    >>>
    >>> # Include mode: keep only inside sphere
    >>> data.filter(FilterValues(
    ...     min_opacity=0.1,
    ...     sphere_radius=5.0,
    ...     max_scale=3.0,
    ...     invert=False
    ... ))
    >>>
    >>> # Exclude mode: remove inside sphere
    >>> data.filter(FilterValues(sphere_radius=2.0, invert=True))
"""

from gsmod.filter.bounds import (
    SceneBounds,
    calculate_recommended_max_scale,
    calculate_scene_bounds,
)
from gsmod.filter.config import (
    UI_RANGES,
    BoxFilter,
    EllipsoidFilter,
    FrustumFilter,
    QualityFilter,
    SphereFilter,
    VolumeFilter,
)

__all__ = [
    # Config types (for apply_geometry_filter)
    "SphereFilter",
    "BoxFilter",
    "EllipsoidFilter",
    "FrustumFilter",
    "QualityFilter",
    "VolumeFilter",
    "UI_RANGES",
    # Utilities
    "SceneBounds",
    "calculate_scene_bounds",
    "calculate_recommended_max_scale",
]
