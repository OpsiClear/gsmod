"""Extended GSData with processing methods.

Provides a simplified API for color, filter, and transform operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from gsply import GSData

if TYPE_CHECKING:
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

        # Apply color using fused kernel
        from gsmod.color.apply import apply_color_values

        self.sh0 = apply_color_values(self.sh0, values)

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
            self.means, self.quats, self.scales, values
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
