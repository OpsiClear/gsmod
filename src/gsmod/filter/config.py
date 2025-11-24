"""
Filter configuration for Gaussian splat filtering.

Each geometry filter type has its own dataclass.
Use union types for type-safe polymorphism.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SphereFilter:
    """
    Sphere volume filter.

    Attributes:
        center: Center point [x, y, z]
        radius: Radius in world units
    """

    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 1.0

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("radius must be positive")


@dataclass
class BoxFilter:
    """
    Box (cuboid) volume filter with optional rotation.

    Attributes:
        center: Center point [x, y, z]
        size: Size in each dimension [width, height, depth]
        rotation: Rotation as axis-angle [rx, ry, rz] in radians
    """

    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rotation: tuple[float, float, float] | None = None

    def __post_init__(self):
        if any(s <= 0 for s in self.size):
            raise ValueError("size dimensions must all be positive")


@dataclass
class EllipsoidFilter:
    """
    Ellipsoid volume filter with optional rotation.

    Attributes:
        center: Center point [x, y, z]
        radii: Radii in each axis [rx, ry, rz]
        rotation: Rotation as axis-angle [rx, ry, rz] in radians
    """

    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radii: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rotation: tuple[float, float, float] | None = None

    def __post_init__(self):
        if any(r <= 0 for r in self.radii):
            raise ValueError("radii must all be positive")


@dataclass
class FrustumFilter:
    """
    Camera frustum volume filter.

    Camera convention: -Z forward, +X right, +Y up.

    Attributes:
        position: Camera position [x, y, z]
        rotation: Camera rotation as axis-angle [rx, ry, rz] in radians
        fov: Vertical field of view in radians
        aspect: Aspect ratio (width/height)
        near: Near clipping plane distance
        far: Far clipping plane distance
    """

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] | None = None
    fov: float = 1.047  # 60 degrees
    aspect: float = 1.0
    near: float = 0.1
    far: float = 100.0

    def __post_init__(self):
        if self.fov <= 0 or self.fov >= 3.14159:
            raise ValueError("fov must be between 0 and pi radians")
        if self.aspect <= 0:
            raise ValueError("aspect must be positive")
        if self.near <= 0:
            raise ValueError("near must be positive")
        if self.far <= self.near:
            raise ValueError("far must be greater than near")


@dataclass
class QualityFilter:
    """
    Quality-based filtering (opacity and scale thresholds).

    Attributes:
        min_opacity: Minimum opacity to keep (0.0 to 1.0)
        max_scale: Maximum scale threshold
    """

    min_opacity: float = 0.0
    max_scale: float = 10.0

    def __post_init__(self):
        if not 0.0 <= self.min_opacity <= 1.0:
            raise ValueError("min_opacity must be between 0.0 and 1.0")
        if self.max_scale < 0.0:
            raise ValueError("max_scale must be non-negative")


# Union type for volume filters - this IS the polymorphism
VolumeFilter = SphereFilter | BoxFilter | EllipsoidFilter | FrustumFilter


# Default UI slider ranges
UI_RANGES = {
    "radius": {"min": 0.1, "max": 100.0, "step": 0.1, "default": 1.0},
    "size": {"min": 0.1, "max": 100.0, "step": 0.1, "default": 1.0},
    "radii": {"min": 0.1, "max": 100.0, "step": 0.1, "default": 1.0},
    "fov": {"min": 0.1, "max": 3.0, "step": 0.01, "default": 1.047},
    "aspect": {"min": 0.1, "max": 10.0, "step": 0.1, "default": 1.0},
    "near": {"min": 0.001, "max": 10.0, "step": 0.01, "default": 0.1},
    "far": {"min": 1.0, "max": 10000.0, "step": 1.0, "default": 100.0},
    "min_opacity": {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.0},
    "max_scale": {"min": 0.1, "max": 10.0, "step": 0.1, "default": 10.0},
}
