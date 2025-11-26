"""Atomic Filter class with boolean operators.

Provides a composable filter API using factory methods and operators.
Filters are immutable and can be combined with &, |, ~ operators.

Example:
    >>> from gsmod import Filter
    >>>
    >>> # Create atomic filters
    >>> opacity = Filter.min_opacity(0.5)
    >>> sphere = Filter.sphere(radius=5.0)
    >>>
    >>> # Combine with operators
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
    from gsmod.gsdata_pro import GSDataPro

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Filter:
    """Atomic filter with boolean operators.

    Filters are immutable and composable via operators:
    - & (AND): Both conditions must pass
    - | (OR): Either condition passes
    - ~ (NOT): Invert the filter

    Factory methods create atomic filters that delegate to optimized
    Numba kernels for mask computation.
    """

    _mask_fn: Callable[[GSDataPro], np.ndarray]
    _description: str

    # ========================================================================
    # Factory methods - Quality filters
    # ========================================================================

    @classmethod
    def min_opacity(cls, threshold: float) -> Filter:
        """Filter Gaussians with opacity >= threshold.

        :param threshold: Minimum opacity value [0.0, 1.0]
        :returns: Filter instance
        """

        def mask_fn(data: GSDataPro) -> np.ndarray:
            from gsmod.config.values import FilterValues
            from gsmod.filter.apply import compute_filter_mask

            return compute_filter_mask(data, FilterValues(min_opacity=threshold))

        return cls(_mask_fn=mask_fn, _description=f"min_opacity({threshold})")

    @classmethod
    def max_opacity(cls, threshold: float) -> Filter:
        """Filter Gaussians with opacity <= threshold.

        :param threshold: Maximum opacity value [0.0, 1.0]
        :returns: Filter instance
        """

        def mask_fn(data: GSDataPro) -> np.ndarray:
            from gsmod.config.values import FilterValues
            from gsmod.filter.apply import compute_filter_mask

            return compute_filter_mask(data, FilterValues(max_opacity=threshold))

        return cls(_mask_fn=mask_fn, _description=f"max_opacity({threshold})")

    @classmethod
    def min_scale(cls, threshold: float) -> Filter:
        """Filter Gaussians with max(scales) >= threshold.

        :param threshold: Minimum scale value
        :returns: Filter instance
        """

        def mask_fn(data: GSDataPro) -> np.ndarray:
            from gsmod.config.values import FilterValues
            from gsmod.filter.apply import compute_filter_mask

            return compute_filter_mask(data, FilterValues(min_scale=threshold))

        return cls(_mask_fn=mask_fn, _description=f"min_scale({threshold})")

    @classmethod
    def max_scale(cls, threshold: float) -> Filter:
        """Filter Gaussians with max(scales) <= threshold.

        :param threshold: Maximum scale value
        :returns: Filter instance
        """

        def mask_fn(data: GSDataPro) -> np.ndarray:
            from gsmod.config.values import FilterValues
            from gsmod.filter.apply import compute_filter_mask

            return compute_filter_mask(data, FilterValues(max_scale=threshold))

        return cls(_mask_fn=mask_fn, _description=f"max_scale({threshold})")

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
        if center is None:
            center = (0.0, 0.0, 0.0)
        # Convert to tuple for FilterValues
        center_tuple = tuple(center) if not isinstance(center, tuple) else center

        def mask_fn(data: GSDataPro) -> np.ndarray:
            from gsmod.config.values import FilterValues
            from gsmod.filter.apply import compute_filter_mask

            return compute_filter_mask(
                data, FilterValues(sphere_radius=radius, sphere_center=center_tuple)
            )

        return cls(_mask_fn=mask_fn, _description=f"sphere(r={radius})")

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
        # Convert to tuples for FilterValues
        min_tuple = tuple(min_corner) if not isinstance(min_corner, tuple) else min_corner
        max_tuple = tuple(max_corner) if not isinstance(max_corner, tuple) else max_corner
        rot_tuple = (
            tuple(rotation)
            if rotation is not None and not isinstance(rotation, tuple)
            else rotation
        )

        def mask_fn(data: GSDataPro) -> np.ndarray:
            from gsmod.config.values import FilterValues
            from gsmod.filter.apply import compute_filter_mask

            return compute_filter_mask(
                data, FilterValues(box_min=min_tuple, box_max=max_tuple, box_rot=rot_tuple)
            )

        desc = f"box(min={min_corner}, max={max_corner})"
        if rotation is not None:
            desc = f"rotated_{desc}"
        return cls(_mask_fn=mask_fn, _description=desc)

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
        # Convert to tuples for FilterValues
        center_tuple = tuple(center) if not isinstance(center, tuple) else center
        radii_tuple = tuple(radii) if not isinstance(radii, tuple) else radii
        rot_tuple = (
            tuple(rotation)
            if rotation is not None and not isinstance(rotation, tuple)
            else rotation
        )

        def mask_fn(data: GSDataPro) -> np.ndarray:
            from gsmod.config.values import FilterValues
            from gsmod.filter.apply import compute_filter_mask

            return compute_filter_mask(
                data,
                FilterValues(
                    ellipsoid_center=center_tuple,
                    ellipsoid_radii=radii_tuple,
                    ellipsoid_rot=rot_tuple,
                ),
            )

        desc = f"ellipsoid(center={center}, radii={radii})"
        if rotation is not None:
            desc = f"rotated_{desc}"
        return cls(_mask_fn=mask_fn, _description=desc)

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
        # Convert to tuples for FilterValues
        pos_tuple = tuple(position) if not isinstance(position, tuple) else position
        rot_tuple = (
            tuple(rotation)
            if rotation is not None and not isinstance(rotation, tuple)
            else rotation
        )

        def mask_fn(data: GSDataPro) -> np.ndarray:
            from gsmod.config.values import FilterValues
            from gsmod.filter.apply import compute_filter_mask

            return compute_filter_mask(
                data,
                FilterValues(
                    frustum_pos=pos_tuple,
                    frustum_rot=rot_tuple,
                    frustum_fov=fov,
                    frustum_aspect=aspect,
                    frustum_near=near,
                    frustum_far=far,
                ),
            )

        return cls(
            _mask_fn=mask_fn,
            _description=f"frustum(fov={np.degrees(fov):.1f}deg, near={near}, far={far})",
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

        :param other: Another Filter
        :returns: Combined filter (AND logic)
        """
        if not isinstance(other, Filter):
            return NotImplemented

        self_fn = self._mask_fn
        other_fn = other._mask_fn

        def combined_fn(data: GSDataPro) -> np.ndarray:
            return self_fn(data) & other_fn(data)

        return Filter(
            _mask_fn=combined_fn,
            _description=f"({self._description} & {other._description})",
        )

    def __or__(self, other: Filter) -> Filter:
        """OR operator: either filter passes.

        :param other: Another Filter
        :returns: Combined filter (OR logic)
        """
        if not isinstance(other, Filter):
            return NotImplemented

        self_fn = self._mask_fn
        other_fn = other._mask_fn

        def combined_fn(data: GSDataPro) -> np.ndarray:
            return self_fn(data) | other_fn(data)

        return Filter(
            _mask_fn=combined_fn,
            _description=f"({self._description} | {other._description})",
        )

    def __invert__(self) -> Filter:
        """NOT operator: invert filter.

        :returns: Inverted filter
        """
        self_fn = self._mask_fn

        def inverted_fn(data: GSDataPro) -> np.ndarray:
            return ~self_fn(data)

        return Filter(
            _mask_fn=inverted_fn,
            _description=f"~({self._description})",
        )

    # ========================================================================
    # Application methods
    # ========================================================================

    def get_mask(self, data: GSDataPro) -> np.ndarray:
        """Compute boolean mask from filter.

        :param data: GSDataPro instance
        :returns: Boolean mask [N] where True = keep
        """
        mask = self._mask_fn(data)
        logger.debug(f"Filter {self._description}: kept {mask.sum()}/{len(mask)}")
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
    def from_values(cls, values) -> Filter:
        """Create Filter from FilterValues.

        Converts FilterValues into a Filter with equivalent mask logic.
        All active filter criteria are combined with AND.

        :param values: FilterValues instance
        :returns: Filter instance
        """
        from gsmod.config.values import FilterValues

        if not isinstance(values, FilterValues):
            raise TypeError(f"Expected FilterValues, got {type(values)}")

        filters: list[Filter] = []

        # Quality filters
        if values.min_opacity > 0.0:
            filters.append(cls.min_opacity(values.min_opacity))
        if values.max_opacity < 1.0:
            filters.append(cls.max_opacity(values.max_opacity))
        if values.min_scale > 0.0:
            filters.append(cls.min_scale(values.min_scale))
        if values.max_scale < 100.0:
            filters.append(cls.max_scale(values.max_scale))

        # Sphere filter
        if values.sphere_radius < float("inf"):
            filters.append(cls.sphere(values.sphere_radius, values.sphere_center))

        # Box filter
        if values.box_min is not None and values.box_max is not None:
            filters.append(cls.box(values.box_min, values.box_max, values.box_rot))

        # Ellipsoid filter
        if values.ellipsoid_radii is not None:
            filters.append(
                cls.ellipsoid(
                    values.ellipsoid_center or (0.0, 0.0, 0.0),
                    values.ellipsoid_radii,
                    values.ellipsoid_rot,
                )
            )

        # Frustum filter
        if values.frustum_pos is not None:
            filters.append(
                cls.frustum(
                    values.frustum_pos,
                    values.frustum_rot,
                    values.frustum_fov,
                    values.frustum_aspect,
                    values.frustum_near,
                    values.frustum_far,
                )
            )

        # Combine all filters with AND
        if not filters:
            # No-op filter: pass everything
            return cls(
                _mask_fn=lambda data: np.ones(len(data.means), dtype=bool), _description="all"
            )

        result = filters[0]
        for f in filters[1:]:
            result = result & f

        # Apply invert if requested
        if values.invert:
            result = ~result

        return result

    def to_values(self):
        """Convert simple Filter to FilterValues.

        Only works for simple atomic filters (single filter type).
        Complex combined filters raise ValueError.

        :returns: FilterValues instance
        :raises ValueError: If filter is too complex to convert
        """
        from gsmod.config.values import FilterValues

        desc = self._description

        # Parse simple filters
        if desc.startswith("min_opacity("):
            val = float(desc[12:-1])
            return FilterValues(min_opacity=val)
        elif desc.startswith("max_opacity("):
            val = float(desc[12:-1])
            return FilterValues(max_opacity=val)
        elif desc.startswith("min_scale("):
            val = float(desc[10:-1])
            return FilterValues(min_scale=val)
        elif desc.startswith("max_scale("):
            val = float(desc[10:-1])
            return FilterValues(max_scale=val)
        elif desc.startswith("sphere(r="):
            val = float(desc[9:-1])
            return FilterValues(sphere_radius=val)
        elif desc == "all":
            return FilterValues()
        elif desc.startswith("~("):
            # Inverted filter - try to convert inner and invert
            # This is a heuristic - may not work for complex cases
            inner_desc = desc[2:-1]
            if inner_desc.startswith("sphere(r="):
                val = float(inner_desc[9:-1])
                return FilterValues(sphere_radius=val, invert=True)
            raise ValueError(f"Cannot convert complex inverted filter to FilterValues: {desc}")
        else:
            raise ValueError(f"Cannot convert complex filter to FilterValues: {desc}")

    def __repr__(self) -> str:
        """Return string representation."""
        return f"Filter({self._description})"
