"""Atomic Filter class with boolean operators.

Provides a composable filter API using factory methods and operators.
Filters are immutable and can be combined with &, |, ~ operators.

Performance:
    - AND operations use fused kernel (fast): Filter.a() & Filter.b()
    - OR/NOT operations use mask combination (flexible): Filter.a() | Filter.b()

Example:
    >>> from gsmod import Filter
    >>>
    >>> # Create atomic filters
    >>> opacity = Filter.min_opacity(0.5)
    >>> sphere = Filter.sphere(radius=5.0)
    >>>
    >>> # Combine with operators (AND uses fused kernel)
    >>> combined = opacity & sphere
    >>>
    >>> # Apply to data
    >>> mask = combined.get_mask(data)
    >>> filtered = combined(data, inplace=False)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gsmod.config.values import FilterValues
    from gsmod.gsdata_pro import GSDataPro

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Filter:
    """Atomic filter with boolean operators.

    Filters are immutable and composable via operators:
    - & (AND): Both conditions must pass (uses fused kernel - fast)
    - | (OR): Either condition passes (uses mask combination)
    - ~ (NOT): Invert the filter (uses mask inversion)

    Factory methods create atomic filters with FilterValues internally,
    enabling fast fused kernel execution for AND combinations.
    """

    _values: FilterValues | None  # For filters that can use fused kernel
    _mask_fn: Callable[[GSDataPro], np.ndarray] | None  # For complex filters
    _description: str

    # ========================================================================
    # Internal helpers
    # ========================================================================

    @classmethod
    def _from_values(cls, values: FilterValues, description: str) -> Filter:
        """Create filter from FilterValues (fast path)."""
        return cls(_values=values, _mask_fn=None, _description=description)

    @classmethod
    def _from_mask_fn(cls, mask_fn: Callable[[GSDataPro], np.ndarray], description: str) -> Filter:
        """Create filter from mask function (slow path for OR/NOT)."""
        return cls(_values=None, _mask_fn=mask_fn, _description=description)

    # ========================================================================
    # Factory methods - Quality filters
    # ========================================================================

    @classmethod
    def min_opacity(cls, threshold: float) -> Filter:
        """Filter Gaussians with opacity >= threshold.

        :param threshold: Minimum opacity value [0.0, 1.0]
        :returns: Filter instance
        """
        from gsmod.config.values import FilterValues

        return cls._from_values(FilterValues(min_opacity=threshold), f"min_opacity({threshold})")

    @classmethod
    def max_opacity(cls, threshold: float) -> Filter:
        """Filter Gaussians with opacity <= threshold.

        :param threshold: Maximum opacity value [0.0, 1.0]
        :returns: Filter instance
        """
        from gsmod.config.values import FilterValues

        return cls._from_values(FilterValues(max_opacity=threshold), f"max_opacity({threshold})")

    @classmethod
    def min_scale(cls, threshold: float) -> Filter:
        """Filter Gaussians with max(scales) >= threshold.

        :param threshold: Minimum scale value
        :returns: Filter instance
        """
        from gsmod.config.values import FilterValues

        return cls._from_values(FilterValues(min_scale=threshold), f"min_scale({threshold})")

    @classmethod
    def max_scale(cls, threshold: float) -> Filter:
        """Filter Gaussians with max(scales) <= threshold.

        :param threshold: Maximum scale value
        :returns: Filter instance
        """
        from gsmod.config.values import FilterValues

        return cls._from_values(FilterValues(max_scale=threshold), f"max_scale({threshold})")

    # ========================================================================
    # Factory methods - Spatial filters (inside)
    # ========================================================================

    @classmethod
    def sphere(
        cls,
        radius: float,
        center: tuple[float, float, float] | np.ndarray | None = None,
    ) -> Filter:
        """Filter Gaussians inside sphere.

        :param radius: Sphere radius in world units
        :param center: Sphere center [x, y, z], defaults to origin
        :returns: Filter instance
        """
        from gsmod.config.values import FilterValues

        if center is None:
            center = (0.0, 0.0, 0.0)
        center_tuple = tuple(center) if not isinstance(center, tuple) else center

        return cls._from_values(
            FilterValues(sphere_radius=radius, sphere_center=center_tuple),
            f"sphere(r={radius})",
        )

    @classmethod
    def box(
        cls,
        min_corner: tuple[float, float, float] | np.ndarray,
        max_corner: tuple[float, float, float] | np.ndarray,
        rotation: tuple[float, float, float] | np.ndarray | None = None,
    ) -> Filter:
        """Filter Gaussians inside axis-aligned or rotated box.

        :param min_corner: Box minimum corner [x, y, z]
        :param max_corner: Box maximum corner [x, y, z]
        :param rotation: Optional axis-angle rotation (radians)
        :returns: Filter instance
        """
        from gsmod.config.values import FilterValues

        min_tuple = tuple(min_corner) if not isinstance(min_corner, tuple) else min_corner
        max_tuple = tuple(max_corner) if not isinstance(max_corner, tuple) else max_corner
        rot_tuple = (
            tuple(rotation)
            if rotation is not None and not isinstance(rotation, tuple)
            else rotation
        )

        desc = f"box(min={min_corner}, max={max_corner})"
        if rotation is not None:
            desc = f"rotated_{desc}"

        return cls._from_values(
            FilterValues(box_min=min_tuple, box_max=max_tuple, box_rot=rot_tuple), desc
        )

    @classmethod
    def ellipsoid(
        cls,
        center: tuple[float, float, float] | np.ndarray,
        radii: tuple[float, float, float] | np.ndarray,
        rotation: tuple[float, float, float] | np.ndarray | None = None,
    ) -> Filter:
        """Filter Gaussians inside ellipsoid.

        :param center: Ellipsoid center [x, y, z]
        :param radii: Ellipsoid radii [rx, ry, rz]
        :param rotation: Optional axis-angle rotation (radians)
        :returns: Filter instance
        """
        from gsmod.config.values import FilterValues

        center_tuple = tuple(center) if not isinstance(center, tuple) else center
        radii_tuple = tuple(radii) if not isinstance(radii, tuple) else radii
        rot_tuple = (
            tuple(rotation)
            if rotation is not None and not isinstance(rotation, tuple)
            else rotation
        )

        desc = f"ellipsoid(center={center}, radii={radii})"
        if rotation is not None:
            desc = f"rotated_{desc}"

        return cls._from_values(
            FilterValues(
                ellipsoid_center=center_tuple,
                ellipsoid_radii=radii_tuple,
                ellipsoid_rot=rot_tuple,
            ),
            desc,
        )

    @classmethod
    def frustum(
        cls,
        position: tuple[float, float, float] | np.ndarray,
        rotation: tuple[float, float, float] | np.ndarray | None,
        fov: float,
        aspect: float,
        near: float,
        far: float,
    ) -> Filter:
        """Filter Gaussians inside camera frustum.

        :param position: Camera position [x, y, z]
        :param rotation: Camera rotation as axis-angle (radians) or None
        :param fov: Vertical field of view in radians
        :param aspect: Aspect ratio (width/height)
        :param near: Near clipping plane distance
        :param far: Far clipping plane distance
        :returns: Filter instance
        """
        from gsmod.config.values import FilterValues

        pos_tuple = tuple(position) if not isinstance(position, tuple) else position
        rot_tuple = (
            tuple(rotation)
            if rotation is not None and not isinstance(rotation, tuple)
            else rotation
        )

        return cls._from_values(
            FilterValues(
                frustum_pos=pos_tuple,
                frustum_rot=rot_tuple,
                frustum_fov=fov,
                frustum_aspect=aspect,
                frustum_near=near,
                frustum_far=far,
            ),
            f"frustum(fov={np.degrees(fov):.1f}deg, near={near}, far={far})",
        )

    # ========================================================================
    # Factory methods - Spatial filters (outside) - convenience wrappers
    # ========================================================================

    @classmethod
    def outside_sphere(
        cls,
        radius: float,
        center: tuple[float, float, float] | np.ndarray | None = None,
    ) -> Filter:
        """Filter Gaussians outside sphere.

        :param radius: Sphere radius in world units
        :param center: Sphere center [x, y, z], defaults to origin
        :returns: Filter instance (inverted sphere)
        """
        return ~cls.sphere(radius, center)

    @classmethod
    def outside_box(
        cls,
        min_corner: tuple[float, float, float] | np.ndarray,
        max_corner: tuple[float, float, float] | np.ndarray,
        rotation: tuple[float, float, float] | np.ndarray | None = None,
    ) -> Filter:
        """Filter Gaussians outside box.

        :param min_corner: Box minimum corner [x, y, z]
        :param max_corner: Box maximum corner [x, y, z]
        :param rotation: Optional axis-angle rotation (radians)
        :returns: Filter instance (inverted box)
        """
        return ~cls.box(min_corner, max_corner, rotation)

    @classmethod
    def outside_ellipsoid(
        cls,
        center: tuple[float, float, float] | np.ndarray,
        radii: tuple[float, float, float] | np.ndarray,
        rotation: tuple[float, float, float] | np.ndarray | None = None,
    ) -> Filter:
        """Filter Gaussians outside ellipsoid.

        :param center: Ellipsoid center [x, y, z]
        :param radii: Ellipsoid radii [rx, ry, rz]
        :param rotation: Optional axis-angle rotation (radians)
        :returns: Filter instance (inverted ellipsoid)
        """
        return ~cls.ellipsoid(center, radii, rotation)

    @classmethod
    def outside_frustum(
        cls,
        position: tuple[float, float, float] | np.ndarray,
        rotation: tuple[float, float, float] | np.ndarray | None,
        fov: float,
        aspect: float,
        near: float,
        far: float,
    ) -> Filter:
        """Filter Gaussians outside camera frustum.

        :param position: Camera position [x, y, z]
        :param rotation: Camera rotation as axis-angle (radians) or None
        :param fov: Vertical field of view in radians
        :param aspect: Aspect ratio (width/height)
        :param near: Near clipping plane distance
        :param far: Far clipping plane distance
        :returns: Filter instance (inverted frustum)
        """
        return ~cls.frustum(position, rotation, fov, aspect, near, far)

    # ========================================================================
    # Boolean operators
    # ========================================================================

    def __and__(self, other: Filter) -> Filter:
        """AND operator: both filters must pass.

        Uses fused kernel when both filters have FilterValues (fast).
        Falls back to mask combination otherwise.

        :param other: Another Filter
        :returns: Combined filter (AND logic)
        """
        if not isinstance(other, Filter):
            return NotImplemented

        # Fast path: merge FilterValues for fused kernel
        if self._values is not None and other._values is not None:
            merged = self._values + other._values
            return Filter._from_values(merged, f"({self._description} & {other._description})")

        # Slow path: mask combination (for complex filters)
        def combined_fn(data: GSDataPro) -> np.ndarray:
            return self.get_mask(data) & other.get_mask(data)

        return Filter._from_mask_fn(combined_fn, f"({self._description} & {other._description})")

    def __or__(self, other: Filter) -> Filter:
        """OR operator: either filter passes.

        Always uses mask combination (cannot be expressed in single fused kernel).

        :param other: Another Filter
        :returns: Combined filter (OR logic)
        """
        if not isinstance(other, Filter):
            return NotImplemented

        def combined_fn(data: GSDataPro) -> np.ndarray:
            return self.get_mask(data) | other.get_mask(data)

        return Filter._from_mask_fn(combined_fn, f"({self._description} | {other._description})")

    def __invert__(self) -> Filter:
        """NOT operator: invert filter.

        Always uses mask inversion (cannot be expressed in fused kernel).

        :returns: Inverted filter
        """

        def inverted_fn(data: GSDataPro) -> np.ndarray:
            return ~self.get_mask(data)

        return Filter._from_mask_fn(inverted_fn, f"~({self._description})")

    # ========================================================================
    # Application methods
    # ========================================================================

    def get_mask(self, data: GSDataPro) -> np.ndarray:
        """Compute boolean mask from filter.

        Uses fused kernel for FilterValues-based filters (fast).

        :param data: GSDataPro instance
        :returns: Boolean mask [N] where True = keep
        """
        from gsmod.filter.apply import compute_filter_mask

        if self._values is not None:
            # Fast path: use fused kernel
            mask = compute_filter_mask(data, self._values)
        else:
            # Slow path: use mask function (for OR/NOT)
            mask = self._mask_fn(data)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Filter %s: kept %d/%d", self._description, mask.sum(), len(mask))
        return mask

    def __call__(self, data: GSDataPro, inplace: bool = False) -> GSDataPro:
        """Apply filter to data.

        :param data: GSDataPro instance
        :param inplace: If True, modify in place (reassign arrays)
        :returns: Filtered GSDataPro
        """
        from gsmod.filter.apply import apply_mask_fused

        if not inplace:
            data = data.clone()

        mask = self.get_mask(data)

        # Apply mask using fused parallel scatter (single pass)
        data.means, data.scales, data.quats, data.opacities, data.sh0, data.shN = apply_mask_fused(
            data, mask
        )

        return data

    # ========================================================================
    # Conversion methods
    # ========================================================================

    @classmethod
    def from_values(cls, values: FilterValues) -> Filter:
        """Create Filter from FilterValues.

        :param values: FilterValues instance
        :returns: Filter instance
        """
        from gsmod.config.values import FilterValues as FilterValuesType

        if not isinstance(values, FilterValuesType):
            raise TypeError(f"Expected FilterValues, got {type(values)}")

        # Build description from active filters
        parts = []
        if values.min_opacity > 0.0:
            parts.append(f"min_opacity({values.min_opacity})")
        if values.max_opacity < 1.0:
            parts.append(f"max_opacity({values.max_opacity})")
        if values.min_scale > 0.0:
            parts.append(f"min_scale({values.min_scale})")
        if values.max_scale < 100.0:
            parts.append(f"max_scale({values.max_scale})")
        if values.sphere_radius < float("inf"):
            parts.append(f"sphere(r={values.sphere_radius})")
        if values.box_min is not None:
            parts.append("box")
        if values.ellipsoid_radii is not None:
            parts.append("ellipsoid")
        if values.frustum_pos is not None:
            parts.append("frustum")

        desc = " & ".join(parts) if parts else "all"
        if values.invert:
            desc = f"~({desc})"

        return cls._from_values(values, desc)

    def to_values(self) -> FilterValues:
        """Convert Filter to FilterValues.

        Only works for filters with internal FilterValues.
        Complex filters (OR/NOT combinations) raise ValueError.

        :returns: FilterValues instance
        :raises ValueError: If filter cannot be converted
        """
        if self._values is not None:
            return self._values

        raise ValueError(f"Cannot convert complex filter to FilterValues: {self._description}")

    @property
    def can_use_fused_kernel(self) -> bool:
        """Check if filter can use fast fused kernel path.

        :returns: True if filter has internal FilterValues
        """
        return self._values is not None

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Filter({self._description})"
