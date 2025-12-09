"""Extended GSData with processing methods.

Provides a simplified API for color, filter, and transform operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from gsply import GSData
from gsply.gsdata import DataFormat

if TYPE_CHECKING:
    import numpy as np

    from gsmod.torch.gstensor_pro import GSTensorPro

from gsmod.config.values import (
    ColorValues,
    FilterValues,
    HistogramConfig,
    OpacityValues,
    TransformValues,
)
from gsmod.histogram.result import HistogramResult


class GSDataPro(GSData):
    """Extended GSData with processing methods.

    All methods modify data inplace by default and return self for chaining.
    Use inplace=False to get a copy.

    Example:
        >>> data = GSDataPro.from_ply("scene.ply")
        >>> data.color(ColorValues(brightness=1.2))
        >>> data.filter(FilterValues(min_opacity=0.3))
        >>> data.transform(TransformValues.from_scale(2.0))
        >>> data.to_ply("output.ply")
    """

    # ========================================================================
    # Core Processing Methods
    # ========================================================================

    def color(self, values: ColorValues, inplace: bool = True) -> Self:
        """Apply color transformation.

        :param values: Color parameters to apply
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications

        Example:
            >>> data.color(ColorValues(brightness=1.2, temperature=0.3))
            >>> data.color(ColorValues.from_k(3200) + ColorValues(contrast=1.1))
        """
        if not inplace:
            data = self.clone()
            return data.color(values, inplace=True)

        if values.is_neutral():
            return self

        # Apply color using SH-aware implementation
        from gsmod.color.apply import apply_color_values

        # Determine if sh0 is in RGB format
        is_sh0_rgb = self._format.get("sh0") == DataFormat.SH0_RGB

        # Apply to all SH bands
        self.sh0, self.shN = apply_color_values(self.sh0, values, self.shN, is_sh0_rgb=is_sh0_rgb)

        return self

    def filter(self, values: FilterValues, inplace: bool = True) -> Self:
        """Filter Gaussians based on criteria.

        Changes array sizes (N -> M where M <= N).

        :param values: Filter parameters
        :param inplace: If True, modify self; if False, return filtered copy
        :returns: Self (filtered) or filtered copy

        Example:
            >>> data.filter(FilterValues(min_opacity=0.3, sphere_radius=5.0))
            >>> data.filter(FilterValues(
            ...     ellipsoid_center=(0, 0, 0),
            ...     ellipsoid_radii=(2.0, 1.0, 1.5)
            ... ))
        """
        if not inplace:
            data = self.clone()
            return data.filter(values, inplace=True)

        if values.is_neutral():
            return self

        # Compute mask and apply using fused kernel
        from gsmod.filter.apply import apply_mask_fused, compute_filter_mask

        mask = compute_filter_mask(self, values)

        # Invert mask if exclude mode is requested
        if values.invert:
            mask = ~mask

        # Apply mask using fused parallel scatter (single pass)
        self.means, self.scales, self.quats, self.opacities, self.sh0, self.shN = apply_mask_fused(
            self, mask
        )

        return self

    def transform(self, values: TransformValues, inplace: bool = True) -> Self:
        """Apply geometric transformation.

        :param values: Transform parameters
        :param inplace: If True, modify self; if False, return transformed copy
        :returns: Self (transformed) or transformed copy

        Example:
            >>> data.transform(TransformValues.from_scale(2.0))
            >>> data.transform(
            ...     TransformValues.from_translation(1, 0, 0) +
            ...     TransformValues.from_rotation_euler(0, 45, 0)
            ... )
        """
        if not inplace:
            data = self.clone()
            return data.transform(values, inplace=True)

        if values.is_neutral():
            return self

        # Apply transform using fused kernel
        from gsmod.transform.apply import apply_transform_values

        self.means, self.quats, self.scales = apply_transform_values(
            self.means,
            self.quats,
            self.scales,
            values,
            is_scales_ply=self.is_scales_ply,
        )

        return self

    def opacity(self, values: OpacityValues, inplace: bool = True) -> Self:
        """Apply opacity adjustment.

        Handles both linear [0, 1] and PLY (logit) opacity formats correctly.
        The scale factor is always applied in linear space.

        :param values: Opacity parameters to apply
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications

        Example:
            >>> data.opacity(OpacityValues(scale=0.5))  # Fade to 50%
            >>> data.opacity(OpacityValues.fade(0.7))  # Fade to 70%
            >>> data.opacity(OpacityValues.boost(1.5))  # Boost opacity
        """
        if not inplace:
            data = self.clone()
            return data.opacity(values, inplace=True)

        if values.is_neutral():
            return self

        # Apply opacity using format-aware function
        from gsmod.opacity.apply import apply_opacity_values

        self.opacities = apply_opacity_values(
            self.opacities, values, is_ply_format=self.is_opacities_ply
        )

        return self

    # ========================================================================
    # Individual Filter Methods
    # ========================================================================

    def filter_min_opacity(self, threshold: float, inplace: bool = True) -> Self:
        """Filter by minimum opacity threshold.

        :param threshold: Minimum opacity value [0.0, 1.0]
        :param inplace: If True, modify self; if False, return filtered copy
        :returns: Self (filtered) or filtered copy
        """
        return self.filter(FilterValues(min_opacity=threshold), inplace=inplace)

    def filter_max_opacity(self, threshold: float, inplace: bool = True) -> Self:
        """Filter by maximum opacity threshold.

        :param threshold: Maximum opacity value [0.0, 1.0]
        :param inplace: If True, modify self; if False, return filtered copy
        :returns: Self (filtered) or filtered copy
        """
        return self.filter(FilterValues(max_opacity=threshold), inplace=inplace)

    def filter_min_scale(self, threshold: float, inplace: bool = True) -> Self:
        """Filter by minimum scale threshold.

        :param threshold: Minimum scale value
        :param inplace: If True, modify self; if False, return filtered copy
        :returns: Self (filtered) or filtered copy
        """
        return self.filter(FilterValues(min_scale=threshold), inplace=inplace)

    def filter_max_scale(self, threshold: float, inplace: bool = True) -> Self:
        """Filter by maximum scale threshold.

        :param threshold: Maximum scale value
        :param inplace: If True, modify self; if False, return filtered copy
        :returns: Self (filtered) or filtered copy
        """
        return self.filter(FilterValues(max_scale=threshold), inplace=inplace)

    def filter_within_sphere(
        self,
        radius: float,
        center: tuple[float, float, float] | None = None,
        inplace: bool = True,
    ) -> Self:
        """Filter to keep Gaussians inside sphere.

        :param radius: Sphere radius in world units
        :param center: Sphere center [x, y, z], defaults to origin
        :param inplace: If True, modify self; if False, return filtered copy
        :returns: Self (filtered) or filtered copy
        """
        values = FilterValues(
            sphere_radius=radius,
            sphere_center=center if center is not None else (0.0, 0.0, 0.0),
        )
        return self.filter(values, inplace=inplace)

    def filter_outside_sphere(
        self,
        radius: float,
        center: tuple[float, float, float] | None = None,
        inplace: bool = True,
    ) -> Self:
        """Filter to keep Gaussians outside sphere.

        :param radius: Sphere radius in world units
        :param center: Sphere center [x, y, z], defaults to origin
        :param inplace: If True, modify self; if False, return filtered copy
        :returns: Self (filtered) or filtered copy
        """
        values = FilterValues(
            sphere_radius=radius,
            sphere_center=center if center is not None else (0.0, 0.0, 0.0),
            invert=True,
        )
        return self.filter(values, inplace=inplace)

    def filter_within_box(
        self,
        min_corner: tuple[float, float, float],
        max_corner: tuple[float, float, float],
        rotation: tuple[float, float, float] | None = None,
        inplace: bool = True,
    ) -> Self:
        """Filter to keep Gaussians inside box.

        :param min_corner: Box minimum corner [x, y, z]
        :param max_corner: Box maximum corner [x, y, z]
        :param rotation: Optional axis-angle rotation (radians)
        :param inplace: If True, modify self; if False, return filtered copy
        :returns: Self (filtered) or filtered copy
        """
        values = FilterValues(box_min=min_corner, box_max=max_corner, box_rot=rotation)
        return self.filter(values, inplace=inplace)

    def filter_outside_box(
        self,
        min_corner: tuple[float, float, float],
        max_corner: tuple[float, float, float],
        rotation: tuple[float, float, float] | None = None,
        inplace: bool = True,
    ) -> Self:
        """Filter to keep Gaussians outside box.

        :param min_corner: Box minimum corner [x, y, z]
        :param max_corner: Box maximum corner [x, y, z]
        :param rotation: Optional axis-angle rotation (radians)
        :param inplace: If True, modify self; if False, return filtered copy
        :returns: Self (filtered) or filtered copy
        """
        values = FilterValues(box_min=min_corner, box_max=max_corner, box_rot=rotation, invert=True)
        return self.filter(values, inplace=inplace)

    def filter_within_ellipsoid(
        self,
        center: tuple[float, float, float],
        radii: tuple[float, float, float],
        rotation: tuple[float, float, float] | None = None,
        inplace: bool = True,
    ) -> Self:
        """Filter to keep Gaussians inside ellipsoid.

        :param center: Ellipsoid center [x, y, z]
        :param radii: Ellipsoid radii [rx, ry, rz]
        :param rotation: Optional axis-angle rotation (radians)
        :param inplace: If True, modify self; if False, return filtered copy
        :returns: Self (filtered) or filtered copy
        """
        values = FilterValues(
            ellipsoid_center=center, ellipsoid_radii=radii, ellipsoid_rot=rotation
        )
        return self.filter(values, inplace=inplace)

    def filter_outside_ellipsoid(
        self,
        center: tuple[float, float, float],
        radii: tuple[float, float, float],
        rotation: tuple[float, float, float] | None = None,
        inplace: bool = True,
    ) -> Self:
        """Filter to keep Gaussians outside ellipsoid.

        :param center: Ellipsoid center [x, y, z]
        :param radii: Ellipsoid radii [rx, ry, rz]
        :param rotation: Optional axis-angle rotation (radians)
        :param inplace: If True, modify self; if False, return filtered copy
        :returns: Self (filtered) or filtered copy
        """
        values = FilterValues(
            ellipsoid_center=center, ellipsoid_radii=radii, ellipsoid_rot=rotation, invert=True
        )
        return self.filter(values, inplace=inplace)

    def filter_within_frustum(
        self,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float] | None,
        fov: float,
        aspect: float,
        near: float,
        far: float,
        inplace: bool = True,
    ) -> Self:
        """Filter to keep Gaussians inside camera frustum.

        :param position: Camera position [x, y, z]
        :param rotation: Camera rotation as axis-angle (radians) or None
        :param fov: Vertical field of view in radians
        :param aspect: Aspect ratio (width/height)
        :param near: Near clipping plane distance
        :param far: Far clipping plane distance
        :param inplace: If True, modify self; if False, return filtered copy
        :returns: Self (filtered) or filtered copy
        """
        values = FilterValues(
            frustum_pos=position,
            frustum_rot=rotation,
            frustum_fov=fov,
            frustum_aspect=aspect,
            frustum_near=near,
            frustum_far=far,
        )
        return self.filter(values, inplace=inplace)

    def filter_outside_frustum(
        self,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float] | None,
        fov: float,
        aspect: float,
        near: float,
        far: float,
        inplace: bool = True,
    ) -> Self:
        """Filter to keep Gaussians outside camera frustum.

        :param position: Camera position [x, y, z]
        :param rotation: Camera rotation as axis-angle (radians) or None
        :param fov: Vertical field of view in radians
        :param aspect: Aspect ratio (width/height)
        :param near: Near clipping plane distance
        :param far: Far clipping plane distance
        :param inplace: If True, modify self; if False, return filtered copy
        :returns: Self (filtered) or filtered copy
        """
        values = FilterValues(
            frustum_pos=position,
            frustum_rot=rotation,
            frustum_fov=fov,
            frustum_aspect=aspect,
            frustum_near=near,
            frustum_far=far,
            invert=True,
        )
        return self.filter(values, inplace=inplace)

    # ========================================================================
    # Individual Transform Methods
    # ========================================================================

    def translate(self, translation: list[float] | tuple[float, ...], inplace: bool = True) -> Self:
        """Translate (move) the scene.

        :param translation: Translation vector [x, y, z]
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications
        """
        import numpy as np

        t = np.asarray(translation, dtype=np.float32)
        if np.allclose(t, 0):
            return self if inplace else self.clone()
        return self.transform(TransformValues.from_translation(t[0], t[1], t[2]), inplace=inplace)

    def scale_uniform(self, scale: float, inplace: bool = True) -> Self:
        """Apply uniform scale.

        :param scale: Scale factor (1.0 = no change)
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications
        """
        if scale == 1.0:
            return self if inplace else self.clone()
        return self.transform(TransformValues.from_scale(scale), inplace=inplace)

    def scale_nonuniform(
        self, scale: list[float] | tuple[float, ...], inplace: bool = True
    ) -> Self:
        """Apply non-uniform scale.

        :param scale: Scale factors [sx, sy, sz]
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications
        """
        import numpy as np

        s = np.asarray(scale, dtype=np.float32)
        if np.allclose(s, 1):
            return self if inplace else self.clone()

        if not inplace:
            data = self.clone()
            return data.scale_nonuniform(scale, inplace=True)

        # Apply non-uniform scale directly to positions and scales
        self.means = self.means * s
        self.scales = self.scales * s
        return self

    def rotate_quaternion(
        self, quat: list[float] | tuple[float, ...], inplace: bool = True
    ) -> Self:
        """Rotate by quaternion.

        :param quat: Quaternion [w, x, y, z] (wxyz format)
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications
        """
        import numpy as np

        q = np.asarray(quat, dtype=np.float32)
        # Identity quaternion is [1, 0, 0, 0]
        if np.allclose(q, [1, 0, 0, 0]):
            return self if inplace else self.clone()
        return self.transform(TransformValues(rotation=tuple(q.tolist())), inplace=inplace)

    def rotate_euler(
        self, angles: list[float] | tuple[float, ...], order: str = "XYZ", inplace: bool = True
    ) -> Self:
        """Rotate by Euler angles.

        :param angles: Rotation angles [rx, ry, rz] in degrees
        :param order: Rotation order (default 'XYZ')
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications
        """
        import numpy as np

        a = np.asarray(angles, dtype=np.float32)
        if np.allclose(a, 0):
            return self if inplace else self.clone()
        return self.transform(
            TransformValues.from_rotation_euler(a[0], a[1], a[2]), inplace=inplace
        )

    def rotate_axis_angle(
        self, axis: list[float] | tuple[float, ...], angle: float, inplace: bool = True
    ) -> Self:
        """Rotate by axis-angle.

        :param axis: Rotation axis [x, y, z]
        :param angle: Rotation angle in degrees
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications
        """
        if angle == 0.0:
            return self if inplace else self.clone()
        return self.transform(
            TransformValues.from_rotation_axis_angle(tuple(axis), angle), inplace=inplace
        )

    def transform_matrix(self, matrix: list | tuple | np.ndarray, inplace: bool = True) -> Self:
        """Apply 4x4 transformation matrix.

        :param matrix: 4x4 homogeneous transformation matrix (array-like)
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications
        """
        import numpy as np

        M = np.asarray(matrix, dtype=np.float32)
        if np.allclose(M, np.eye(4)):
            return self if inplace else self.clone()
        return self.transform(TransformValues.from_matrix(M), inplace=inplace)

    # ========================================================================
    # Individual Color Methods
    # ========================================================================

    def adjust_brightness(self, factor: float, inplace: bool = True) -> Self:
        """Adjust brightness by multiplicative factor.

        :param factor: Brightness multiplier (1.0 = no change)
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications
        """
        if factor == 1.0:
            return self if inplace else self.clone()
        return self.color(ColorValues(brightness=factor), inplace=inplace)

    def adjust_contrast(self, factor: float, inplace: bool = True) -> Self:
        """Adjust contrast by multiplicative factor.

        :param factor: Contrast multiplier (1.0 = no change)
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications
        """
        if factor == 1.0:
            return self if inplace else self.clone()
        return self.color(ColorValues(contrast=factor), inplace=inplace)

    def adjust_saturation(self, factor: float, inplace: bool = True) -> Self:
        """Adjust saturation by multiplicative factor.

        :param factor: Saturation multiplier (1.0 = no change)
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications
        """
        if factor == 1.0:
            return self if inplace else self.clone()
        return self.color(ColorValues(saturation=factor), inplace=inplace)

    def adjust_gamma(self, gamma: float, inplace: bool = True) -> Self:
        """Adjust gamma (power curve).

        :param gamma: Gamma value (1.0 = linear, no change)
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications
        """
        if gamma == 1.0:
            return self if inplace else self.clone()
        return self.color(ColorValues(gamma=gamma), inplace=inplace)

    def adjust_temperature(self, temp: float, inplace: bool = True) -> Self:
        """Adjust color temperature.

        :param temp: Temperature (-1.0 cool/blue to 1.0 warm/orange, 0.0 = neutral)
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications
        """
        if temp == 0.0:
            return self if inplace else self.clone()
        return self.color(ColorValues(temperature=temp), inplace=inplace)

    def adjust_vibrance(self, factor: float, inplace: bool = True) -> Self:
        """Adjust vibrance (smart saturation).

        :param factor: Vibrance multiplier (1.0 = no change)
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications
        """
        if factor == 1.0:
            return self if inplace else self.clone()
        return self.color(ColorValues(vibrance=factor), inplace=inplace)

    def adjust_hue_shift(self, degrees: float, inplace: bool = True) -> Self:
        """Shift hue by specified degrees.

        :param degrees: Hue shift in degrees (-180 to 180)
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications
        """
        if degrees == 0.0:
            return self if inplace else self.clone()
        return self.color(ColorValues(hue_shift=degrees), inplace=inplace)

    def adjust_shadows(self, factor: float, inplace: bool = True) -> Self:
        """Adjust shadow tones.

        :param factor: Shadow adjustment (-1.0 darker to 1.0 lighter, 0.0 = no change)
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications
        """
        if factor == 0.0:
            return self if inplace else self.clone()
        return self.color(ColorValues(shadows=factor), inplace=inplace)

    def adjust_highlights(self, factor: float, inplace: bool = True) -> Self:
        """Adjust highlight tones.

        :param factor: Highlight adjustment (-1.0 darker to 1.0 lighter, 0.0 = no change)
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications
        """
        if factor == 0.0:
            return self if inplace else self.clone()
        return self.color(ColorValues(highlights=factor), inplace=inplace)

    def adjust_tint(self, value: float, inplace: bool = True) -> Self:
        """Adjust tint (green-magenta balance).

        :param value: Tint value (-1.0 green to 1.0 magenta, 0.0 = neutral)
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications
        """
        if value == 0.0:
            return self if inplace else self.clone()
        return self.color(ColorValues(tint=value), inplace=inplace)

    def adjust_fade(self, value: float, inplace: bool = True) -> Self:
        """Apply fade effect (black point lift).

        :param value: Fade amount (0.0 = no fade, 1.0 = full lift to white)
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications
        """
        if value == 0.0:
            return self if inplace else self.clone()
        return self.color(ColorValues(fade=value), inplace=inplace)

    # ========================================================================
    # Scene Utilities
    # ========================================================================

    def compute_bounds(self) -> tuple:
        """Compute axis-aligned bounding box of the scene.

        :returns: Tuple of (min_bounds, max_bounds) as shape (3,) arrays

        Example:
            >>> min_b, max_b = data.compute_bounds()
            >>> scene_size = max_b - min_b
        """
        import numpy as np

        return np.min(self.means, axis=0), np.max(self.means, axis=0)

    def get_centroid(self):
        """Compute centroid (center of mass) of all Gaussian positions.

        :returns: Centroid as shape (3,) array

        Example:
            >>> center = data.get_centroid()
            >>> print(f"Scene center: {center}")
        """
        import numpy as np

        return np.mean(self.means, axis=0)

    def center_at_origin(self, inplace: bool = True) -> Self:
        """Translate scene so centroid is at origin.

        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications

        Example:
            >>> data.center_at_origin()  # Scene is now centered at (0, 0, 0)
        """
        import numpy as np

        if not inplace:
            data = self.clone()
            return data.center_at_origin(inplace=True)

        centroid = self.get_centroid()
        if np.allclose(centroid, 0):
            return self

        self.means = self.means - centroid
        return self

    def normalize_scale(self, target_size: float = 2.0, inplace: bool = True) -> Self:
        """Normalize scene to fit within target bounding box size.

        Uniformly scales the scene so the largest dimension equals target_size.

        :param target_size: Target size for largest dimension (default 2.0)
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications

        Example:
            >>> data.normalize_scale(target_size=2.0)  # Scene now fits in [-1, 1]^3
        """
        import numpy as np

        if not inplace:
            data = self.clone()
            return data.normalize_scale(target_size=target_size, inplace=True)

        min_b, max_b = self.compute_bounds()
        current_size = np.max(max_b - min_b)

        if current_size < 1e-8:
            return self  # Scene too small to scale

        scale_factor = target_size / current_size

        if np.isclose(scale_factor, 1.0):
            return self

        # Scale positions and scales
        self.means = self.means * scale_factor
        self.scales = self.scales * scale_factor
        return self

    # ========================================================================
    # Histogram Methods
    # ========================================================================

    def histogram_colors(self, config: HistogramConfig | None = None) -> HistogramResult:
        """Compute histogram of color values.

        :param config: Histogram configuration (default: 256 bins)
        :returns: HistogramResult with counts and statistics

        Example:
            >>> result = data.histogram_colors()
            >>> print(f"Mean RGB: {result.mean}")
            >>> adjustment = result.to_color_values("vibrant")
            >>> data.color(adjustment)
        """
        from gsmod.histogram.apply import compute_histogram_colors

        return compute_histogram_colors(self.sh0, config)

    def histogram_opacity(self, config: HistogramConfig | None = None) -> HistogramResult:
        """Compute histogram of opacity values.

        :param config: Histogram configuration (default: 256 bins)
        :returns: HistogramResult with counts and statistics

        Example:
            >>> result = data.histogram_opacity()
            >>> print(f"Mean opacity: {result.mean}")
        """
        from gsmod.histogram.apply import compute_histogram_opacity

        return compute_histogram_opacity(self.opacities, config)

    def histogram_scales(self, config: HistogramConfig | None = None) -> HistogramResult:
        """Compute histogram of scale values.

        Uses mean scale across all 3 dimensions for each Gaussian.

        :param config: Histogram configuration (default: 256 bins)
        :returns: HistogramResult with counts and statistics

        Example:
            >>> result = data.histogram_scales()
            >>> print(f"Mean scale: {result.mean}")
        """
        from gsmod.histogram.apply import compute_histogram_scales

        return compute_histogram_scales(self.scales, config)

    def histogram_positions(
        self, config: HistogramConfig | None = None, axis: int | None = None
    ) -> HistogramResult:
        """Compute histogram of position values.

        :param config: Histogram configuration (default: 256 bins)
        :param axis: Axis to histogram (0=X, 1=Y, 2=Z, None=distance from origin)
        :returns: HistogramResult with counts and statistics

        Example:
            >>> result = data.histogram_positions()  # Distance from origin
            >>> result = data.histogram_positions(axis=1)  # Y coordinates
        """
        from gsmod.histogram.apply import compute_histogram_positions

        return compute_histogram_positions(self.means, config, axis)

    # ========================================================================
    # Constructors
    # ========================================================================

    @classmethod
    def from_gsdata(cls, data: GSData) -> GSDataPro:
        """Create from existing GSData.

        :param data: Source GSData object
        :returns: GSDataPro instance
        """
        pro = cls.__new__(cls)
        # Copy all attributes
        pro.means = data.means
        pro.scales = data.scales
        pro.quats = data.quats
        pro.opacities = data.opacities
        pro.sh0 = data.sh0
        pro.shN = getattr(data, "shN", None)
        pro.masks = getattr(data, "masks", None)
        pro.mask_names = getattr(data, "mask_names", None)
        pro._base = getattr(data, "_base", None)
        # Copy format tracking using public API if available
        if hasattr(pro, "copy_format_from") and hasattr(data, "_format"):
            pro.copy_format_from(data)
        elif hasattr(data, "_format"):
            pro._format = data._format.copy()
        else:
            pro._format = {}
        return pro

    @classmethod
    def from_ply(cls, path: str) -> GSDataPro:
        """Load from PLY file.

        :param path: Path to PLY file
        :returns: GSDataPro instance
        """
        from gsply import plyread

        data = plyread(path)
        return cls.from_gsdata(data)

    # ========================================================================
    # Conversion
    # ========================================================================

    def to_gsdata(self) -> GSData:
        """Convert to base GSData.

        :returns: GSData instance
        """
        data = GSData(
            means=self.means,
            scales=self.scales,
            quats=self.quats,
            opacities=self.opacities,
            sh0=self.sh0,
            shN=self.shN,
            masks=getattr(self, "masks", None),
            mask_names=getattr(self, "mask_names", None),
            _base=getattr(self, "_base", None),
        )
        # Copy format tracking using public API
        if hasattr(data, "copy_format_from"):
            data.copy_format_from(self)
        elif hasattr(self, "_format"):
            data._format = self._format.copy()
        return data

    def to_gpu(self, device: str = "cuda") -> GSTensorPro:
        """Convert to GPU tensor.

        :param device: Target device (default 'cuda')
        :returns: GSTensorPro instance
        """
        from gsmod.torch.gstensor_pro import GSTensorPro

        return GSTensorPro.from_gsdata(self, device=device)

    def clone(self) -> GSDataPro:
        """Create deep copy.

        :returns: New GSDataPro with copied arrays
        """
        pro = GSDataPro.__new__(GSDataPro)
        pro.means = self.means.copy()
        pro.scales = self.scales.copy()
        pro.quats = self.quats.copy()
        pro.opacities = self.opacities.copy()
        pro.sh0 = self.sh0.copy()
        pro.shN = self.shN.copy() if self.shN is not None else None
        pro.masks = self.masks.copy() if self.masks is not None else None
        pro.mask_names = list(self.mask_names) if self.mask_names is not None else None
        pro._base = self._base.copy() if self._base is not None else None
        # Copy format tracking using public API
        if hasattr(pro, "copy_format_from"):
            pro.copy_format_from(self)
        elif hasattr(self, "_format"):
            pro._format = self._format.copy()
        return pro

    def to_ply(self, path: str) -> None:
        """Save to PLY file.

        :param path: Output path
        """
        from gsply import plywrite

        plywrite(path, self.to_gsdata())
