"""Configuration value dataclasses with merge support.

This module provides value-holding dataclasses that can be merged using
the + operator, with mathematically correct composition for each type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gsmod.torch.learn import LearnableColor, LearnableFilter, LearnableTransform


@dataclass
class ColorValues:
    """Color parameter values with merge support.

    Merge semantics:
    - Multiplicative fields (brightness, contrast, etc.): a * b
    - Additive fields (temperature, shadows, etc.): a + b
    - hue_shift: additive with wrap to [-180, 180]

    Example:
        >>> warm = ColorValues(temperature=0.3, saturation=1.2)
        >>> bright = ColorValues(brightness=1.3)
        >>> preset = warm + bright
        >>> # temperature=0.3, brightness=1.3, saturation=1.2
    """

    # Multiplicative composition (neutral=1.0)
    brightness: float = 1.0
    contrast: float = 1.0
    gamma: float = 1.0
    saturation: float = 1.0
    vibrance: float = 1.0

    # Additive composition (neutral=0.0)
    temperature: float = 0.0
    tint: float = 0.0  # -1.0 (green) to 1.0 (magenta)
    shadows: float = 0.0
    highlights: float = 0.0
    fade: float = 0.0  # 0.0 to 1.0 (black point lift)

    # Split toning (additive, neutral=0.0)
    shadow_tint_hue: float = 0.0  # -180 to 180 degrees
    shadow_tint_sat: float = 0.0  # 0.0 to 1.0 intensity
    highlight_tint_hue: float = 0.0  # -180 to 180 degrees
    highlight_tint_sat: float = 0.0  # 0.0 to 1.0 intensity

    # Additive with wrap (neutral=0.0)
    hue_shift: float = 0.0

    def __add__(self, other: ColorValues) -> ColorValues:
        """Merge using composition rules."""
        if not isinstance(other, ColorValues):
            return NotImplemented

        return ColorValues(
            # Multiplicative
            brightness=self.brightness * other.brightness,
            contrast=self.contrast * other.contrast,
            gamma=self.gamma * other.gamma,
            saturation=self.saturation * other.saturation,
            vibrance=self.vibrance * other.vibrance,
            # Additive
            temperature=self.temperature + other.temperature,
            tint=self.tint + other.tint,
            shadows=self.shadows + other.shadows,
            highlights=self.highlights + other.highlights,
            fade=self.fade + other.fade,
            # Split toning (additive)
            shadow_tint_hue=(self.shadow_tint_hue + other.shadow_tint_hue + 180) % 360 - 180,
            shadow_tint_sat=self.shadow_tint_sat + other.shadow_tint_sat,
            highlight_tint_hue=(self.highlight_tint_hue + other.highlight_tint_hue + 180) % 360
            - 180,
            highlight_tint_sat=self.highlight_tint_sat + other.highlight_tint_sat,
            # Additive with wrap to [-180, 180]
            hue_shift=(self.hue_shift + other.hue_shift + 180) % 360 - 180,
        )

    def __radd__(self, other):
        """Support sum() with initial value 0."""
        if other == 0:
            return self
        return self.__add__(other)

    def clamp(self) -> ColorValues:
        """Clamp all values to valid ranges.

        :returns: New ColorValues with clamped values
        """
        return ColorValues(
            brightness=max(0.0, min(5.0, self.brightness)),
            contrast=max(0.0, min(5.0, self.contrast)),
            gamma=max(0.1, min(5.0, self.gamma)),
            saturation=max(0.0, min(5.0, self.saturation)),
            vibrance=max(0.0, min(5.0, self.vibrance)),
            temperature=max(-1.0, min(1.0, self.temperature)),
            tint=max(-1.0, min(1.0, self.tint)),
            shadows=max(-1.0, min(1.0, self.shadows)),
            highlights=max(-1.0, min(1.0, self.highlights)),
            fade=max(0.0, min(1.0, self.fade)),
            shadow_tint_hue=max(-180.0, min(180.0, self.shadow_tint_hue)),
            shadow_tint_sat=max(0.0, min(1.0, self.shadow_tint_sat)),
            highlight_tint_hue=max(-180.0, min(180.0, self.highlight_tint_hue)),
            highlight_tint_sat=max(0.0, min(1.0, self.highlight_tint_sat)),
            hue_shift=max(-180.0, min(180.0, self.hue_shift)),
        )

    def is_neutral(self) -> bool:
        """Check if all values are neutral (no-op).

        :returns: True if applying these values would have no effect
        """
        return (
            self.brightness == 1.0
            and self.contrast == 1.0
            and self.gamma == 1.0
            and self.saturation == 1.0
            and self.vibrance == 1.0
            and self.temperature == 0.0
            and self.tint == 0.0
            and self.shadows == 0.0
            and self.highlights == 0.0
            and self.fade == 0.0
            and self.shadow_tint_sat == 0.0
            and self.highlight_tint_sat == 0.0
            and self.hue_shift == 0.0
        )

    def learn(self, *params: str) -> LearnableColor:
        """Create learnable nn.Module from these values.

        Args:
            *params: Parameter names to learn. If empty, learns all.

        Returns:
            LearnableColor nn.Module initialized with these values.

        Example:
            >>> model = ColorValues(brightness=1.0).learn('brightness', 'saturation').cuda()
            >>> # Train...
            >>> learned = model.to_values()
        """
        from gsmod.torch.learn import LearnableColor

        return LearnableColor.from_values(self, list(params) if params else None)

    # Unit conversion factory methods

    @classmethod
    def from_k(cls, kelvin: float) -> ColorValues:
        """Create ColorValues with temperature from Kelvin.

        Converts color temperature in Kelvin to the -1 to 1 range:
        - 2000K = +1.0 (very warm/orange)
        - 6500K = 0.0 (neutral/daylight)
        - 11000K = -1.0 (very cool/blue)

        :param kelvin: Color temperature in Kelvin (2000-11000)
        :returns: ColorValues with temperature set
        """
        # Linear mapping: 6500K = 0, range is 4500K per unit
        temp = (6500.0 - kelvin) / 4500.0
        temp = max(-1.0, min(1.0, temp))
        return cls(temperature=temp)

    @classmethod
    def from_rad(cls, radians: float) -> ColorValues:
        """Create ColorValues with hue shift from radians.

        :param radians: Hue shift in radians
        :returns: ColorValues with hue_shift set (converted to degrees)
        """
        degrees = np.degrees(radians)
        # Wrap to [-180, 180]
        degrees = (degrees + 180) % 360 - 180
        return cls(hue_shift=float(degrees))

    def to_k(self) -> float:
        """Convert current temperature to Kelvin.

        :returns: Color temperature in Kelvin
        """
        return 6500.0 - self.temperature * 4500.0

    def to_rad(self) -> float:
        """Convert current hue_shift to radians.

        :returns: Hue shift in radians
        """
        return np.radians(self.hue_shift)


@dataclass
class FilterValues:
    """Filter parameter values with AND-merge support.

    Merge semantics use 'stricter wins' (intersection) logic:
    - min_opacity: max(a, b) - higher threshold is stricter
    - max_scale: min(a, b) - lower cap is stricter
    - sphere_radius: min(a, b) - smaller region is stricter

    Filter Mode:
    - invert=False (default): Include mode - keep only what matches (filter out outside)
    - invert=True: Exclude mode - remove what matches (keep everything outside)

    Important - Composition vs Sequential Filtering:
    - When composing with '+', the invert parameter uses OR logic
    - For complex include/exclude logic, use sequential filtering instead:
      data.filter(f1).filter(f2) rather than data.filter(f1 + f2)

    Example:
        >>> f1 = FilterValues(min_opacity=0.3)
        >>> f2 = FilterValues(min_opacity=0.5, max_scale=2.0)
        >>> combined = f1 + f2
        >>> # min_opacity=0.5, max_scale=2.0

        >>> # Include mode (default): keep only inside sphere
        >>> f_include = FilterValues(sphere_radius=5.0, invert=False)

        >>> # Exclude mode: remove inside sphere, keep outside
        >>> f_exclude = FilterValues(sphere_radius=5.0, invert=True)

        >>> # Sequential filtering (recommended for complex logic)
        >>> data.filter(FilterValues(sphere_radius=3.0, invert=False))  # Include outer
        >>> data.filter(FilterValues(sphere_radius=1.5, invert=True))   # Exclude inner
        >>> # Result: Hollow shell
    """

    # Opacity filtering
    min_opacity: float = 0.0  # higher = stricter
    max_opacity: float = 1.0  # lower = stricter

    # Scale filtering
    min_scale: float = 0.0  # higher = stricter
    max_scale: float = 100.0  # lower = stricter

    # Spatial filtering - sphere
    sphere_radius: float = float("inf")  # smaller = stricter
    sphere_center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Spatial filtering - box
    box_min: tuple[float, float, float] | None = None
    box_max: tuple[float, float, float] | None = None
    box_rot: tuple[float, float, float] | None = None  # axis-angle (radians)

    # Spatial filtering - ellipsoid
    ellipsoid_center: tuple[float, float, float] | None = None
    ellipsoid_radii: tuple[float, float, float] | None = None
    ellipsoid_rot: tuple[float, float, float] | None = None  # axis-angle (radians)

    # Spatial filtering - frustum (camera view culling)
    frustum_pos: tuple[float, float, float] | None = None
    frustum_rot: tuple[float, float, float] | None = None  # axis-angle (radians)
    frustum_fov: float = 1.047  # vertical FOV in radians (default 60 deg)
    frustum_aspect: float = 1.0  # width/height
    frustum_near: float = 0.1
    frustum_far: float = 100.0

    # Filter mode: False=include (default), True=exclude
    invert: bool = False

    def __add__(self, other: FilterValues) -> FilterValues:
        """Merge using 'stricter wins' (intersection) logic."""
        if not isinstance(other, FilterValues):
            return NotImplemented

        # For sphere, use the smaller radius and its center
        if self.sphere_radius <= other.sphere_radius:
            sphere_radius = self.sphere_radius
            sphere_center = self.sphere_center
        else:
            sphere_radius = other.sphere_radius
            sphere_center = other.sphere_center

        # For box, take intersection (stricter bounds)
        box_min = None
        box_max = None
        box_rot = None
        if self.box_min is not None and other.box_min is not None:
            box_min = (
                max(self.box_min[0], other.box_min[0]),
                max(self.box_min[1], other.box_min[1]),
                max(self.box_min[2], other.box_min[2]),
            )
            box_max = (
                min(self.box_max[0], other.box_max[0]),
                min(self.box_max[1], other.box_max[1]),
                min(self.box_max[2], other.box_max[2]),
            )
            # Use rotation from smaller box (by volume)
            self_vol = (
                (self.box_max[0] - self.box_min[0])
                * (self.box_max[1] - self.box_min[1])
                * (self.box_max[2] - self.box_min[2])
            )
            other_vol = (
                (other.box_max[0] - other.box_min[0])
                * (other.box_max[1] - other.box_min[1])
                * (other.box_max[2] - other.box_min[2])
            )
            box_rot = self.box_rot if self_vol <= other_vol else other.box_rot
        elif self.box_min is not None:
            box_min = self.box_min
            box_max = self.box_max
            box_rot = self.box_rot
        elif other.box_min is not None:
            box_min = other.box_min
            box_max = other.box_max
            box_rot = other.box_rot

        # For ellipsoid, use smaller volume (product of radii)
        ellipsoid_center = None
        ellipsoid_radii = None
        ellipsoid_rot = None
        if self.ellipsoid_radii is not None and other.ellipsoid_radii is not None:
            self_vol = self.ellipsoid_radii[0] * self.ellipsoid_radii[1] * self.ellipsoid_radii[2]
            other_vol = (
                other.ellipsoid_radii[0] * other.ellipsoid_radii[1] * other.ellipsoid_radii[2]
            )
            if self_vol <= other_vol:
                ellipsoid_center = self.ellipsoid_center
                ellipsoid_radii = self.ellipsoid_radii
                ellipsoid_rot = self.ellipsoid_rot
            else:
                ellipsoid_center = other.ellipsoid_center
                ellipsoid_radii = other.ellipsoid_radii
                ellipsoid_rot = other.ellipsoid_rot
        elif self.ellipsoid_radii is not None:
            ellipsoid_center = self.ellipsoid_center
            ellipsoid_radii = self.ellipsoid_radii
            ellipsoid_rot = self.ellipsoid_rot
        elif other.ellipsoid_radii is not None:
            ellipsoid_center = other.ellipsoid_center
            ellipsoid_radii = other.ellipsoid_radii
            ellipsoid_rot = other.ellipsoid_rot

        # For frustum, use smaller far distance (stricter)
        frustum_pos = None
        frustum_rot = None
        frustum_fov = self.frustum_fov
        frustum_aspect = self.frustum_aspect
        frustum_near = self.frustum_near
        frustum_far = self.frustum_far
        if self.frustum_pos is not None and other.frustum_pos is not None:
            if self.frustum_far <= other.frustum_far:
                frustum_pos = self.frustum_pos
                frustum_rot = self.frustum_rot
                frustum_fov = self.frustum_fov
                frustum_aspect = self.frustum_aspect
                frustum_near = self.frustum_near
                frustum_far = self.frustum_far
            else:
                frustum_pos = other.frustum_pos
                frustum_rot = other.frustum_rot
                frustum_fov = other.frustum_fov
                frustum_aspect = other.frustum_aspect
                frustum_near = other.frustum_near
                frustum_far = other.frustum_far
        elif self.frustum_pos is not None:
            frustum_pos = self.frustum_pos
            frustum_rot = self.frustum_rot
        elif other.frustum_pos is not None:
            frustum_pos = other.frustum_pos
            frustum_rot = other.frustum_rot
            frustum_fov = other.frustum_fov
            frustum_aspect = other.frustum_aspect
            frustum_near = other.frustum_near
            frustum_far = other.frustum_far

        # For invert, use OR logic: if either inverts, result inverts
        invert = self.invert or other.invert

        return FilterValues(
            min_opacity=max(self.min_opacity, other.min_opacity),
            max_opacity=min(self.max_opacity, other.max_opacity),
            min_scale=max(self.min_scale, other.min_scale),
            max_scale=min(self.max_scale, other.max_scale),
            sphere_radius=sphere_radius,
            sphere_center=sphere_center,
            box_min=box_min,
            box_max=box_max,
            box_rot=box_rot,
            ellipsoid_center=ellipsoid_center,
            ellipsoid_radii=ellipsoid_radii,
            ellipsoid_rot=ellipsoid_rot,
            frustum_pos=frustum_pos,
            frustum_rot=frustum_rot,
            frustum_fov=frustum_fov,
            frustum_aspect=frustum_aspect,
            frustum_near=frustum_near,
            frustum_far=frustum_far,
            invert=invert,
        )

    def __radd__(self, other):
        """Support sum() with initial value 0."""
        if other == 0:
            return self
        return self.__add__(other)

    def clamp(self) -> FilterValues:
        """Clamp to valid ranges.

        :returns: New FilterValues with clamped values
        """
        return FilterValues(
            min_opacity=max(0.0, min(1.0, self.min_opacity)),
            max_opacity=max(0.0, min(1.0, self.max_opacity)),
            min_scale=max(0.0, min(100.0, self.min_scale)),
            max_scale=max(0.0, min(100.0, self.max_scale)),
            sphere_radius=max(0.0, self.sphere_radius),
            sphere_center=self.sphere_center,
            box_min=self.box_min,
            box_max=self.box_max,
            box_rot=self.box_rot,
            ellipsoid_center=self.ellipsoid_center,
            ellipsoid_radii=self.ellipsoid_radii,
            ellipsoid_rot=self.ellipsoid_rot,
            frustum_pos=self.frustum_pos,
            frustum_rot=self.frustum_rot,
            frustum_fov=max(0.01, min(3.14, self.frustum_fov)),
            frustum_aspect=max(0.1, min(10.0, self.frustum_aspect)),
            frustum_near=max(0.001, self.frustum_near),
            frustum_far=max(self.frustum_near + 0.001, self.frustum_far),
            invert=self.invert,
        )

    def is_neutral(self) -> bool:
        """Check if no filtering applied.

        :returns: True if these values would not filter anything
        """
        return (
            self.min_opacity == 0.0
            and self.max_opacity >= 1.0
            and self.min_scale == 0.0
            and self.max_scale >= 100.0
            and self.sphere_radius == float("inf")
            and self.box_min is None
            and self.ellipsoid_radii is None
            and self.frustum_pos is None
            and not self.invert  # invert=True is NOT neutral
        )

    def learn(self, *params: str) -> LearnableFilter:
        """Create learnable nn.Module from these values.

        Args:
            *params: Parameter names to learn. If empty, learns all.

        Returns:
            LearnableFilter nn.Module initialized with these values.

        Example:
            >>> model = FilterValues(min_opacity=0.1).learn('min_opacity').cuda()
        """
        from gsmod.torch.learn import LearnableFilter

        return LearnableFilter.from_values(self, list(params) if params else None)


@dataclass
class TransformValues:
    """Full 3D transform with matrix composition support.

    Convention: a + b applies a FIRST, then b.

    Example:
        >>> translate = TransformValues.from_translation(1, 0, 0)
        >>> rotate = TransformValues.from_rotation_euler(0, 0, 90)
        >>> composed = translate + rotate  # translate then rotate
    """

    scale: float = 1.0
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 homogeneous transformation matrix.

        :returns: 4x4 numpy array
        """
        from gsmod.transform.api import quaternion_to_rotation_matrix

        quat = np.array(self.rotation, dtype=np.float32)
        R = quaternion_to_rotation_matrix(quat)
        R = R * self.scale

        M = np.eye(4, dtype=np.float32)
        M[:3, :3] = R
        M[:3, 3] = self.translation
        return M

    @classmethod
    def from_matrix(cls, M: np.ndarray) -> TransformValues:
        """Extract parameters from 4x4 matrix.

        :param M: 4x4 homogeneous transformation matrix
        :returns: TransformValues instance
        """
        from gsmod.transform.api import rotation_matrix_to_quaternion

        scale = float(np.linalg.norm(M[:3, 0]))
        if scale > 1e-6:
            R = M[:3, :3] / scale
        else:
            R = np.eye(3, dtype=np.float32)
            scale = 0.0

        rotation = tuple(rotation_matrix_to_quaternion(R).tolist())
        translation = tuple(M[:3, 3].tolist())

        return cls(scale=scale, rotation=rotation, translation=translation)

    def __add__(self, other: TransformValues) -> TransformValues:
        """Compose transforms: self applied FIRST, then other.

        :param other: Transform to apply after self
        :returns: Composed transform
        """
        if not isinstance(other, TransformValues):
            return NotImplemented

        M_combined = other.to_matrix() @ self.to_matrix()
        return TransformValues.from_matrix(M_combined)

    def __radd__(self, other):
        """Support sum() with initial value 0."""
        if other == 0:
            return self
        return other.__add__(self)

    def is_neutral(self) -> bool:
        """Check if identity transform.

        :returns: True if this is the identity transform
        """
        import numpy as np

        scale_arr = np.asarray(self.scale, dtype=np.float32).reshape(-1)
        scale_val = float(scale_arr[0])
        rot = np.asarray(self.rotation, dtype=np.float32).reshape(-1)
        trans = np.asarray(self.translation, dtype=np.float32).reshape(-1)

        return (
            np.allclose(scale_val, 1.0)
            and np.allclose(rot, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
            and np.allclose(trans, np.array([0.0, 0.0, 0.0], dtype=np.float32))
        )

    def learn(self, *params: str) -> LearnableTransform:
        """Create learnable nn.Module from these values.

        Args:
            *params: Parameter names to learn. If empty, learns all.

        Returns:
            LearnableTransform nn.Module initialized with these values.

        Example:
            >>> model = TransformValues().learn('rotation', 'translation').cuda()
        """
        from gsmod.torch.learn import LearnableTransform

        return LearnableTransform.from_values(self, list(params) if params else None)

    # Factory methods
    @classmethod
    def from_scale(cls, factor: float) -> TransformValues:
        """Create uniform scale transform.

        :param factor: Scale factor
        :returns: TransformValues with only scale set
        """
        return cls(scale=factor)

    @classmethod
    def from_translation(cls, x: float, y: float, z: float) -> TransformValues:
        """Create translation transform.

        :param x: X translation
        :param y: Y translation
        :param z: Z translation
        :returns: TransformValues with only translation set
        """
        return cls(translation=(x, y, z))

    @classmethod
    def from_rotation_euler(cls, rx: float, ry: float, rz: float) -> TransformValues:
        """Create rotation transform from Euler angles.

        :param rx: X rotation in degrees
        :param ry: Y rotation in degrees
        :param rz: Z rotation in degrees
        :returns: TransformValues with only rotation set
        """
        from gsmod.transform.api import euler_to_quaternion

        # Convert degrees to radians for euler_to_quaternion
        angles = np.radians(np.array([rx, ry, rz], dtype=np.float32))
        quat = euler_to_quaternion(angles)
        return cls(rotation=tuple(quat.tolist()))

    @classmethod
    def from_rotation_axis_angle(
        cls, axis: tuple[float, float, float], angle: float
    ) -> TransformValues:
        """Create rotation transform from axis-angle.

        :param axis: Rotation axis [x, y, z]
        :param angle: Rotation angle in degrees
        :returns: TransformValues with only rotation set
        """
        from gsmod.transform.api import axis_angle_to_quaternion

        axis_arr = np.array(axis, dtype=np.float32)
        # Normalize axis and scale by angle (axis-angle representation)
        axis_norm = np.linalg.norm(axis_arr)
        if axis_norm > 1e-6:
            axis_normalized = axis_arr / axis_norm
        else:
            axis_normalized = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        angle_rad = np.radians(angle)
        axis_angle = axis_normalized * angle_rad
        quat = axis_angle_to_quaternion(axis_angle)
        return cls(rotation=tuple(quat.tolist()))

    # Radian unit aliases

    @classmethod
    def from_euler_rad(cls, rx: float, ry: float, rz: float) -> TransformValues:
        """Create rotation transform from Euler angles in radians.

        :param rx: X rotation in radians
        :param ry: Y rotation in radians
        :param rz: Z rotation in radians
        :returns: TransformValues with only rotation set
        """
        from gsmod.transform.api import euler_to_quaternion

        angles = np.array([rx, ry, rz], dtype=np.float32)
        quat = euler_to_quaternion(angles)
        return cls(rotation=tuple(quat.tolist()))

    @classmethod
    def from_axis_angle_rad(cls, axis: tuple[float, float, float], angle: float) -> TransformValues:
        """Create rotation transform from axis-angle with angle in radians.

        :param axis: Rotation axis [x, y, z]
        :param angle: Rotation angle in radians
        :returns: TransformValues with only rotation set
        """
        from gsmod.transform.api import axis_angle_to_quaternion

        axis_arr = np.array(axis, dtype=np.float32)
        # Normalize axis and scale by angle
        axis_norm = np.linalg.norm(axis_arr)
        if axis_norm > 1e-6:
            axis_normalized = axis_arr / axis_norm
        else:
            axis_normalized = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        axis_angle = axis_normalized * angle
        quat = axis_angle_to_quaternion(axis_angle)
        return cls(rotation=tuple(quat.tolist()))


@dataclass
class HistogramConfig:
    """Histogram configuration with merge support.

    Merge semantics:
    - n_bins: use larger value (more detail)
    - Range limits: expand to cover both
    - normalize: use True if either is True

    Example:
        >>> config = HistogramConfig(n_bins=128, normalize=True)
        >>> result = data.histogram_colors(config)
    """

    # Number of bins (default 256 for 8-bit-like resolution)
    n_bins: int = 256

    # Optional range limits (None = auto from data)
    min_value: float | None = None
    max_value: float | None = None

    # Normalization - return density instead of counts
    normalize: bool = False

    def __add__(self, other: HistogramConfig) -> HistogramConfig:
        """Merge configs using 'more detail' logic."""
        if not isinstance(other, HistogramConfig):
            return NotImplemented

        # Determine range - expand to cover both
        min_val = None
        if self.min_value is not None and other.min_value is not None:
            min_val = min(self.min_value, other.min_value)
        elif self.min_value is not None:
            min_val = self.min_value
        elif other.min_value is not None:
            min_val = other.min_value

        max_val = None
        if self.max_value is not None and other.max_value is not None:
            max_val = max(self.max_value, other.max_value)
        elif self.max_value is not None:
            max_val = self.max_value
        elif other.max_value is not None:
            max_val = other.max_value

        return HistogramConfig(
            n_bins=max(self.n_bins, other.n_bins),
            min_value=min_val,
            max_value=max_val,
            normalize=self.normalize or other.normalize,
        )

    def __radd__(self, other):
        """Support sum() with initial value 0."""
        if other == 0:
            return self
        return self.__add__(other)

    def is_neutral(self) -> bool:
        """Check if default configuration.

        :returns: True if this is the default configuration
        """
        return (
            self.n_bins == 256
            and self.min_value is None
            and self.max_value is None
            and not self.normalize
        )


@dataclass
class OpacityValues:
    """Opacity adjustment values with merge support.

    Handles both linear [0, 1] and PLY (logit) opacity formats correctly.
    The scale factor is applied in linear space regardless of storage format.

    Merge semantics: multiplicative composition.
        scale=0.5 + scale=0.5 -> scale=0.25 (50% of 50%)

    Example:
        >>> fade = OpacityValues(scale=0.5)  # 50% opacity
        >>> boost = OpacityValues(scale=1.5)  # 150% opacity (clamped)
        >>> combined = fade + boost  # 75% opacity
    """

    # Multiplicative scale factor (1.0 = no change)
    # Values < 1.0 make more transparent
    # Values > 1.0 make more opaque (with diminishing returns near 1.0)
    scale: float = 1.0

    def __add__(self, other: OpacityValues) -> OpacityValues:
        """Merge using multiplicative composition."""
        if not isinstance(other, OpacityValues):
            return NotImplemented
        return OpacityValues(scale=self.scale * other.scale)

    def __radd__(self, other):
        """Support sum() with initial value 0."""
        if other == 0:
            return self
        return self.__add__(other)

    def is_neutral(self) -> bool:
        """Check if this is a no-op (scale=1.0).

        :returns: True if scale is 1.0
        """
        return self.scale == 1.0

    def clamp(self) -> OpacityValues:
        """Return clamped copy with scale in valid range.

        :returns: OpacityValues with scale clamped to [0.0, 10.0]
        """
        return OpacityValues(scale=max(0.0, min(10.0, self.scale)))

    @classmethod
    def fade(cls, amount: float = 0.5) -> OpacityValues:
        """Create fade effect (reduce opacity).

        :param amount: Fade amount (0.0 = invisible, 1.0 = no change)
        :returns: OpacityValues with scale set to amount
        """
        return cls(scale=max(0.0, min(1.0, amount)))

    @classmethod
    def boost(cls, amount: float = 1.5) -> OpacityValues:
        """Create boost effect (increase opacity).

        :param amount: Boost factor (1.0 = no change, >1.0 = more opaque)
        :returns: OpacityValues with scale set to amount
        """
        return cls(scale=max(1.0, amount))
