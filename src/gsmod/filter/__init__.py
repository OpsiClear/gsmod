"""
Gaussian splat filtering module.

Filter Gaussians using FilterValues with volume, opacity, and scale criteria.

Features:
- Volume filtering (sphere, box, ellipsoid, frustum)
- Quality filtering (opacity, scale)

Example:
    >>> from gsmod import GSDataPro, FilterValues
    >>>
    >>> # Filter with FilterValues
    >>> data.filter(FilterValues(
    ...     min_opacity=0.1,
    ...     sphere_radius=5.0,
    ...     max_scale=3.0
    ... ))
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
