"""
Gaussian splat filtering API.

Provides volume, opacity, and scale filtering for Gaussian splats.
All filters use absolute values in world units.
"""

from __future__ import annotations

import logging

import numpy as np

from gsmod.filter.config import (
    BoxFilter,
    EllipsoidFilter,
    FrustumFilter,
    QualityFilter,
    SphereFilter,
    VolumeFilter,
)

# Import Numba kernels for optimization
from gsmod.filter.kernels import (
    ellipsoid_filter_numba,
    frustum_filter_numba,
    rotated_cuboid_filter_numba,
    sphere_filter_numba,
)

logger = logging.getLogger(__name__)


def apply_geometry_filter(
    positions: np.ndarray,
    geometry: VolumeFilter,
    quality: QualityFilter | None = None,
) -> np.ndarray:
    """
    Apply geometry filter using the config interface.

    :param positions: Gaussian positions [N, 3]
    :param geometry: Volume filter (SphereFilter, BoxFilter, EllipsoidFilter, FrustumFilter)
    :param quality: Optional quality filter (opacity/scale thresholds)
    :return: Boolean mask [N] where True = keep Gaussian

    Example:
        >>> from gsmod.filter.config import SphereFilter, FrustumFilter, QualityFilter
        >>> from gsmod.filter.api import apply_geometry_filter
        >>>
        >>> # Sphere filter with absolute radius
        >>> mask = apply_geometry_filter(
        ...     positions,
        ...     SphereFilter(center=(0, 0, 0), radius=5.0)
        ... )
        >>>
        >>> # Frustum filter with quality
        >>> mask = apply_geometry_filter(
        ...     positions,
        ...     FrustumFilter(position=(0, 0, 10), fov=1.047),
        ...     QualityFilter(min_opacity=0.1, max_scale=2.0)
        ... )
    """
    # Get quality parameters
    min_opacity = quality.min_opacity if quality else 0.0
    max_scale = quality.max_scale if quality else 10.0

    # Dispatch based on filter type
    if isinstance(geometry, SphereFilter):
        return _apply_filter(
            positions=positions,
            filter_type="sphere",
            sphere_center=geometry.center,
            sphere_radius=geometry.radius,
            opacity_threshold=min_opacity,
            max_scale=max_scale,
        )

    elif isinstance(geometry, BoxFilter):
        return _apply_filter(
            positions=positions,
            filter_type="cuboid",
            cuboid_center=geometry.center,
            cuboid_size=geometry.size,
            cuboid_rotation=geometry.rotation,
            opacity_threshold=min_opacity,
            max_scale=max_scale,
        )

    elif isinstance(geometry, EllipsoidFilter):
        return _apply_filter(
            positions=positions,
            filter_type="ellipsoid",
            ellipsoid_center=geometry.center,
            ellipsoid_radii=geometry.radii,
            ellipsoid_rotation=geometry.rotation,
            opacity_threshold=min_opacity,
            max_scale=max_scale,
        )

    elif isinstance(geometry, FrustumFilter):
        return _apply_filter(
            positions=positions,
            filter_type="frustum",
            frustum_position=geometry.position,
            frustum_rotation=geometry.rotation,
            frustum_fov=geometry.fov,
            frustum_aspect=geometry.aspect,
            frustum_near=geometry.near,
            frustum_far=geometry.far,
            opacity_threshold=min_opacity,
            max_scale=max_scale,
        )

    else:
        raise TypeError(f"Unknown geometry filter type: {type(geometry)}")


def _apply_filter(
    positions: np.ndarray,
    opacities: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    # Filter parameters
    filter_type: str = "none",
    sphere_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    sphere_radius: float = 1.0,
    cuboid_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cuboid_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cuboid_rotation: tuple[float, float, float] | None = None,
    # Ellipsoid parameters
    ellipsoid_center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ellipsoid_radii: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ellipsoid_rotation: tuple[float, float, float] | None = None,
    # Frustum parameters
    frustum_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    frustum_rotation: tuple[float, float, float] | None = None,
    frustum_fov: float = 1.047,  # 60 degrees in radians
    frustum_aspect: float = 1.0,
    frustum_near: float = 0.1,
    frustum_far: float = 100.0,
    opacity_threshold: float = 0.05,
    max_scale: float = 10.0,
) -> np.ndarray:
    """
    Apply volume, opacity, and scale filtering to Gaussian splats.

    Creates a boolean mask indicating which Gaussians pass all filter criteria.
    All filters use AND logic (all conditions must be met).
    All sizes and radii are in absolute world units.

    :param positions: Gaussian positions [N, 3]
    :param opacities: Gaussian opacities [N] in range [0, 1] (optional)
    :param scales: Gaussian scales [N, 3] (optional)
    :param filter_type: Spatial filter type ("none", "sphere", "cuboid", "ellipsoid", "frustum")
    :param sphere_center: Center point for sphere filtering [x, y, z]
    :param sphere_radius: Absolute radius in world units
    :param cuboid_center: Center point for box filtering [x, y, z]
    :param cuboid_size: Full size in each dimension [width, height, depth]
    :param cuboid_rotation: Rotation in axis-angle format (radians) or None
    :param ellipsoid_center: Center point [x, y, z]
    :param ellipsoid_radii: Radii in each axis [rx, ry, rz]
    :param ellipsoid_rotation: Rotation in axis-angle format (radians) or None
    :param frustum_position: Camera position [x, y, z]
    :param frustum_rotation: Camera rotation in axis-angle format (radians) or None
    :param frustum_fov: Vertical field of view in radians
    :param frustum_aspect: Aspect ratio (width/height)
    :param frustum_near: Near clipping plane distance
    :param frustum_far: Far clipping plane distance
    :param opacity_threshold: Minimum opacity to keep (0.0 to 1.0)
    :param max_scale: Maximum scale threshold
    :return: Boolean mask [N] where True = keep Gaussian

    Example:
        >>> # Sphere filtering with absolute radius
        >>> mask = _apply_filter(
        ...     positions,
        ...     filter_type="sphere",
        ...     sphere_center=(0.0, 0.0, 0.0),
        ...     sphere_radius=5.0
        ... )

        >>> # Box filtering
        >>> mask = _apply_filter(
        ...     positions,
        ...     filter_type="cuboid",
        ...     cuboid_center=(0.0, 0.0, 0.0),
        ...     cuboid_size=(10.0, 10.0, 10.0)
        ... )

        >>> # Combined filtering
        >>> mask = _apply_filter(
        ...     positions, opacities, scales,
        ...     filter_type="sphere",
        ...     sphere_radius=3.0,
        ...     opacity_threshold=0.05,
        ...     max_scale=2.5
        ... )
    """
    # Validate inputs
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions must be [N, 3], got shape {positions.shape}")

    n_gaussians = len(positions)
    if n_gaussians == 0:
        return np.array([], dtype=bool)

    # Start with all Gaussians included
    mask = np.ones(n_gaussians, dtype=bool)

    # === SPATIAL FILTERING ===
    if filter_type != "none":
        if filter_type == "sphere":
            mask = mask & _apply_sphere_filter(
                positions,
                sphere_center,
                sphere_radius,
            )

        elif filter_type == "ellipsoid":
            mask = mask & _apply_ellipsoid_filter(
                positions,
                ellipsoid_center,
                ellipsoid_radii,
                ellipsoid_rotation,
            )

        elif filter_type == "cuboid":
            mask = mask & _apply_rotated_cuboid_filter(
                positions,
                cuboid_center,
                cuboid_size,
                cuboid_rotation,
            )

        elif filter_type == "frustum":
            mask = mask & _apply_frustum_filter(
                positions,
                frustum_position,
                frustum_rotation,
                frustum_fov,
                frustum_aspect,
                frustum_near,
                frustum_far,
            )

        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")

    # === OPACITY + SCALE FILTERING (FUSED) ===
    needs_opacity_filter = opacities is not None and opacity_threshold > 0.0
    needs_scale_filter = scales is not None and max_scale < 10.0

    if needs_opacity_filter or needs_scale_filter:
        # Validate inputs
        if needs_opacity_filter:
            opacities = np.asarray(opacities, dtype=np.float32)
            if opacities.shape != (n_gaussians,):
                raise ValueError(
                    f"opacities shape {opacities.shape} doesn't match "
                    f"positions shape {positions.shape}"
                )

        if needs_scale_filter:
            scales = np.asarray(scales, dtype=np.float32)
            if scales.ndim != 2 or scales.shape != (n_gaussians, 3):
                raise ValueError(
                    f"scales shape {scales.shape} doesn't match positions shape {positions.shape}"
                )

        # Import fused kernel
        from gsmod.filter.kernels import opacity_scale_filter_fused

        # Apply fused opacity + scale filter in single pass
        out_mask = np.empty(n_gaussians, dtype=np.bool_)
        opacity_scale_filter_fused(
            mask,
            opacities if needs_opacity_filter else None,
            scales if needs_scale_filter else None,
            opacity_threshold,
            max_scale,
            out_mask,
        )
        mask = out_mask

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Fused filter: opacity_threshold={opacity_threshold}, "
                f"max_scale={max_scale}, kept={mask.sum()}/{n_gaussians}"
            )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Total filtered: {n_gaussians} -> {mask.sum()} Gaussians")

    return mask


def _apply_sphere_filter(
    positions: np.ndarray,
    sphere_center: tuple[float, float, float],
    sphere_radius: float,
) -> np.ndarray:
    """
    Apply sphere volume filter with Numba optimization.

    :param positions: Gaussian positions [N, 3]
    :param sphere_center: Center point
    :param sphere_radius: Absolute radius in world units
    :return: Boolean mask [N] where True = inside sphere
    """
    radius_sq = sphere_radius**2
    center = np.array(sphere_center, dtype=np.float32)

    mask = np.empty(len(positions), dtype=np.bool_)
    sphere_filter_numba(positions, center, radius_sq, mask)

    logger.debug(
        f"Sphere filter: kept {mask.sum()}/{len(mask)} "
        f"(center={center}, radius={sphere_radius:.3f})"
    )

    return mask


def _axis_angle_to_rotation_matrix(
    axis_angle: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """
    Convert axis-angle rotation to 3x3 rotation matrix.

    :param axis_angle: Rotation vector [3] where magnitude is angle in radians
    :return: Rotation matrix [3, 3]
    """
    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    angle = np.linalg.norm(axis_angle)

    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)

    # Normalize axis
    axis = axis_angle / angle

    # Rodrigues' rotation formula
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float32
    )

    # R = I + sin(angle) * K + (1 - cos(angle)) * K^2
    R = np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    return R


def _apply_ellipsoid_filter(
    positions: np.ndarray,
    ellipsoid_center: tuple[float, float, float],
    ellipsoid_radii: tuple[float, float, float],
    ellipsoid_rotation: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """
    Apply ellipsoid volume filter with Numba optimization.

    :param positions: Gaussian positions [N, 3]
    :param ellipsoid_center: Center point [x, y, z]
    :param ellipsoid_radii: Radii in each axis [rx, ry, rz]
    :param ellipsoid_rotation: Rotation in axis-angle format (radians) or None
    :return: Boolean mask [N] where True = inside ellipsoid
    """
    center = np.array(ellipsoid_center, dtype=np.float32)
    radii = np.array(ellipsoid_radii, dtype=np.float32)

    if ellipsoid_rotation is not None:
        rotation_matrix = _axis_angle_to_rotation_matrix(ellipsoid_rotation).T
    else:
        rotation_matrix = np.eye(3, dtype=np.float32)

    mask = np.empty(len(positions), dtype=np.bool_)
    ellipsoid_filter_numba(positions, center, radii, rotation_matrix, mask)

    logger.debug(
        f"Ellipsoid filter: kept {mask.sum()}/{len(mask)} (center={center}, radii={radii})"
    )

    return mask


def _apply_rotated_cuboid_filter(
    positions: np.ndarray,
    cuboid_center: tuple[float, float, float],
    cuboid_size: tuple[float, float, float],
    cuboid_rotation: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """
    Apply rotated cuboid (box) volume filter with Numba optimization.

    :param positions: Gaussian positions [N, 3]
    :param cuboid_center: Center point [x, y, z]
    :param cuboid_size: Full size in each dimension [width, height, depth]
    :param cuboid_rotation: Rotation in axis-angle format (radians) or None
    :return: Boolean mask [N] where True = inside cuboid
    """
    center = np.array(cuboid_center, dtype=np.float32)
    half_extents = np.array(cuboid_size, dtype=np.float32) * 0.5

    if cuboid_rotation is not None:
        rotation_matrix = _axis_angle_to_rotation_matrix(cuboid_rotation).T
    else:
        rotation_matrix = np.eye(3, dtype=np.float32)

    mask = np.empty(len(positions), dtype=np.bool_)
    rotated_cuboid_filter_numba(positions, center, half_extents, rotation_matrix, mask)

    logger.debug(
        f"Rotated cuboid filter: kept {mask.sum()}/{len(mask)} "
        f"(center={center}, half_extents={half_extents})"
    )

    return mask


def _apply_frustum_filter(
    positions: np.ndarray,
    frustum_position: tuple[float, float, float],
    frustum_rotation: tuple[float, float, float] | None = None,
    frustum_fov: float = 1.047,
    frustum_aspect: float = 1.0,
    frustum_near: float = 0.1,
    frustum_far: float = 100.0,
) -> np.ndarray:
    """
    Apply camera frustum volume filter with Numba optimization.

    :param positions: Gaussian positions [N, 3]
    :param frustum_position: Camera position [x, y, z]
    :param frustum_rotation: Camera rotation in axis-angle format (radians) or None
    :param frustum_fov: Vertical field of view in radians
    :param frustum_aspect: Aspect ratio width/height
    :param frustum_near: Near clipping plane distance
    :param frustum_far: Far clipping plane distance
    :return: Boolean mask [N] where True = inside frustum
    """
    camera_pos = np.array(frustum_position, dtype=np.float32)

    if frustum_rotation is not None:
        rotation_matrix = _axis_angle_to_rotation_matrix(frustum_rotation).T
    else:
        rotation_matrix = np.eye(3, dtype=np.float32)

    tan_half_fov_y = np.tan(frustum_fov / 2)
    tan_half_fov_x = tan_half_fov_y * frustum_aspect

    mask = np.empty(len(positions), dtype=np.bool_)
    frustum_filter_numba(
        positions,
        camera_pos,
        rotation_matrix,
        tan_half_fov_x,
        tan_half_fov_y,
        frustum_near,
        frustum_far,
        mask,
    )

    logger.debug(
        f"Frustum filter: kept {mask.sum()}/{len(mask)} "
        f"(pos={camera_pos}, fov={np.degrees(frustum_fov):.1f}deg, "
        f"near={frustum_near}, far={frustum_far})"
    )

    return mask
