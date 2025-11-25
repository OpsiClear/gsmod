"""
Unified processing interface for Gaussian Splatting data.

Provides auto-dispatch between CPU (NumPy/Numba) and GPU (PyTorch) backends
based on input data type. This is the recommended high-level API for processing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, overload

from gsmod.config.values import ColorValues, FilterValues, OpacityValues, TransformValues

if TYPE_CHECKING:
    from gsply import GSData
    from gsply.torch import GSTensor

    from gsmod.gsdata_pro import GSDataPro
    from gsmod.torch.gstensor_pro import GSTensorPro

logger = logging.getLogger(__name__)


class GaussianProcessor:
    """Unified processor supporting both CPU and GPU data.

    Automatically dispatches to the appropriate backend based on input type:
    - GSData/GSDataPro -> CPU processing (NumPy/Numba)
    - GSTensor/GSTensorPro -> GPU processing (PyTorch)

    Example:
        >>> from gsmod import GaussianProcessor
        >>> from gsmod.config.values import ColorValues, TransformValues
        >>>
        >>> processor = GaussianProcessor()
        >>>
        >>> # CPU processing
        >>> data = GSDataPro.from_ply("scene.ply")
        >>> data = processor.color(data, ColorValues(brightness=1.2))
        >>> data = processor.transform(data, TransformValues.from_scale(2.0))
        >>>
        >>> # GPU processing (auto-detected)
        >>> tensor = GSTensorPro.from_ply("scene.ply", device="cuda")
        >>> tensor = processor.color(tensor, ColorValues(brightness=1.2))
        >>> tensor = processor.transform(tensor, TransformValues.from_scale(2.0))
    """

    def __init__(self):
        """Initialize the processor."""
        self._color_gpu = None
        self._transform_gpu = None
        self._filter_gpu = None

    # ========================================================================
    # Color Processing
    # ========================================================================

    @overload
    def color(
        self, data: GSData | GSDataPro, values: ColorValues, inplace: bool = True
    ) -> GSData | GSDataPro: ...

    @overload
    def color(
        self, data: GSTensor | GSTensorPro, values: ColorValues, inplace: bool = True
    ) -> GSTensor | GSTensorPro: ...

    def color(self, data, values: ColorValues, inplace: bool = True):
        """Apply color adjustments - auto-dispatches to CPU or GPU.

        :param data: GSData/GSDataPro (CPU) or GSTensor/GSTensorPro (GPU)
        :param values: Color parameters to apply
        :param inplace: If True, modify data in-place; if False, create copy
        :return: Processed data (same type as input)

        Example:
            >>> processor = GaussianProcessor()
            >>> data = processor.color(data, ColorValues(brightness=1.2, saturation=1.1))
        """
        if values.is_neutral():
            return data if inplace else data.clone()

        if self._is_gpu_data(data):
            return self._color_gpu_impl(data, values, inplace)
        return self._color_cpu_impl(data, values, inplace)

    def _color_cpu_impl(self, data, values: ColorValues, inplace: bool):
        """CPU color implementation using Numba kernels."""
        from gsmod.color.apply import apply_color_values

        if not inplace:
            data = data.clone()

        data.sh0 = apply_color_values(data.sh0, values)
        return data

    def _color_gpu_impl(self, data, values: ColorValues, inplace: bool):
        """GPU color implementation using PyTorch."""
        from gsmod.torch.color import ColorGPU

        if self._color_gpu is None:
            self._color_gpu = ColorGPU()
        else:
            self._color_gpu.reset()

        # Build color pipeline from ColorValues
        if values.temperature != 0.0:
            self._color_gpu.temperature(values.temperature)
        if values.tint != 0.0:
            self._color_gpu.tint(values.tint)
        if values.brightness != 1.0:
            self._color_gpu.brightness(values.brightness)
        if values.contrast != 1.0:
            self._color_gpu.contrast(values.contrast)
        if values.gamma != 1.0:
            self._color_gpu.gamma(values.gamma)
        if values.saturation != 1.0:
            self._color_gpu.saturation(values.saturation)
        if values.vibrance != 1.0:
            self._color_gpu.vibrance(values.vibrance)
        if abs(values.hue_shift) >= 0.5:
            self._color_gpu.hue_shift(values.hue_shift)
        if values.shadows != 0.0:
            self._color_gpu.shadows(values.shadows)
        if values.highlights != 0.0:
            self._color_gpu.highlights(values.highlights)
        if values.fade != 0.0:
            self._color_gpu.fade(values.fade)
        if values.shadow_tint_sat != 0.0:
            self._color_gpu.shadow_tint(values.shadow_tint_hue, values.shadow_tint_sat)
        if values.highlight_tint_sat != 0.0:
            self._color_gpu.highlight_tint(values.highlight_tint_hue, values.highlight_tint_sat)

        return self._color_gpu(data, inplace=inplace)

    # ========================================================================
    # Transform Processing
    # ========================================================================

    @overload
    def transform(
        self, data: GSData | GSDataPro, values: TransformValues, inplace: bool = True
    ) -> GSData | GSDataPro: ...

    @overload
    def transform(
        self, data: GSTensor | GSTensorPro, values: TransformValues, inplace: bool = True
    ) -> GSTensor | GSTensorPro: ...

    def transform(self, data, values: TransformValues, inplace: bool = True):
        """Apply geometric transformation - auto-dispatches to CPU or GPU.

        :param data: GSData/GSDataPro (CPU) or GSTensor/GSTensorPro (GPU)
        :param values: Transform parameters to apply
        :param inplace: If True, modify data in-place; if False, create copy
        :return: Processed data (same type as input)

        Example:
            >>> processor = GaussianProcessor()
            >>> data = processor.transform(data, TransformValues.from_scale(2.0))
        """
        if values.is_neutral():
            return data if inplace else data.clone()

        if self._is_gpu_data(data):
            return self._transform_gpu_impl(data, values, inplace)
        return self._transform_cpu_impl(data, values, inplace)

    def _transform_cpu_impl(self, data, values: TransformValues, inplace: bool):
        """CPU transform implementation using Numba kernels."""
        from gsmod.transform.apply import apply_transform_values

        if not inplace:
            data = data.clone()

        data.means, data.quats, data.scales = apply_transform_values(
            data.means, data.quats, data.scales, values
        )
        return data

    def _transform_gpu_impl(self, data, values: TransformValues, inplace: bool):
        """GPU transform implementation using PyTorch."""
        from gsmod.torch.transform import TransformGPU

        if self._transform_gpu is None:
            self._transform_gpu = TransformGPU()
        else:
            self._transform_gpu.reset()

        # Build transform pipeline from TransformValues
        # Apply translation
        if values.translation != (0.0, 0.0, 0.0):
            self._transform_gpu.translate(list(values.translation))

        # Apply rotation (as quaternion)
        if values.rotation != (1.0, 0.0, 0.0, 0.0):
            self._transform_gpu.rotate_quaternion(list(values.rotation))

        # Apply scale
        if values.scale != 1.0:
            self._transform_gpu.scale(values.scale)

        return self._transform_gpu(data, inplace=inplace)

    # ========================================================================
    # Filter Processing
    # ========================================================================

    @overload
    def filter(
        self, data: GSData | GSDataPro, values: FilterValues, inplace: bool = True
    ) -> GSData | GSDataPro: ...

    @overload
    def filter(
        self, data: GSTensor | GSTensorPro, values: FilterValues, inplace: bool = True
    ) -> GSTensor | GSTensorPro: ...

    def filter(self, data, values: FilterValues, inplace: bool = True):
        """Apply filtering - auto-dispatches to CPU or GPU.

        :param data: GSData/GSDataPro (CPU) or GSTensor/GSTensorPro (GPU)
        :param values: Filter parameters to apply
        :param inplace: If True, modify data in-place; if False, create copy
        :return: Filtered data (same type as input)

        Example:
            >>> processor = GaussianProcessor()
            >>> data = processor.filter(data, FilterValues(min_opacity=0.3))
        """
        if values.is_neutral():
            return data if inplace else data.clone()

        if self._is_gpu_data(data):
            return self._filter_gpu_impl(data, values, inplace)
        return self._filter_cpu_impl(data, values, inplace)

    def _filter_cpu_impl(self, data, values: FilterValues, inplace: bool):
        """CPU filter implementation using Numba kernels."""
        from gsmod.filter.apply import apply_mask_fused, compute_filter_mask

        if not inplace:
            data = data.clone()

        mask = compute_filter_mask(data, values)

        # Invert mask if exclude mode is requested
        if values.invert:
            mask = ~mask

        (
            data.means,
            data.scales,
            data.quats,
            data.opacities,
            data.sh0,
            data.shN,
        ) = apply_mask_fused(data, mask)

        return data

    def _filter_gpu_impl(self, data, values: FilterValues, inplace: bool):
        """GPU filter implementation using PyTorch."""
        # Use GSTensorPro.filter() which already handles FilterValues correctly
        # including the invert parameter
        if not inplace:
            data = data.clone()

        return data.filter(values, inplace=True)

    # ========================================================================
    # Opacity Processing
    # ========================================================================

    @overload
    def opacity(
        self, data: GSData | GSDataPro, values: OpacityValues, inplace: bool = True
    ) -> GSData | GSDataPro: ...

    @overload
    def opacity(
        self, data: GSTensor | GSTensorPro, values: OpacityValues, inplace: bool = True
    ) -> GSTensor | GSTensorPro: ...

    def opacity(self, data, values: OpacityValues, inplace: bool = True):
        """Apply opacity adjustment - auto-dispatches to CPU or GPU.

        Handles both linear [0, 1] and PLY (logit) opacity formats correctly.
        The scale factor is always applied in linear space.

        :param data: GSData/GSDataPro (CPU) or GSTensor/GSTensorPro (GPU)
        :param values: Opacity parameters to apply
        :param inplace: If True, modify data in-place; if False, create copy
        :return: Processed data (same type as input)

        Example:
            >>> processor = GaussianProcessor()
            >>> data = processor.opacity(data, OpacityValues(scale=0.5))  # Fade
            >>> data = processor.opacity(data, OpacityValues.fade(0.7))  # 70%
        """
        if values.is_neutral():
            return data if inplace else data.clone()

        if self._is_gpu_data(data):
            return self._opacity_gpu_impl(data, values, inplace)
        return self._opacity_cpu_impl(data, values, inplace)

    def _opacity_cpu_impl(self, data, values: OpacityValues, inplace: bool):
        """CPU opacity implementation."""
        from gsmod.opacity.apply import apply_opacity_values

        if not inplace:
            data = data.clone()

        # Use format-aware opacity adjustment
        is_ply = getattr(data, "is_opacities_ply", False)
        data.opacities = apply_opacity_values(data.opacities, values, is_ply_format=is_ply)

        return data

    def _opacity_gpu_impl(self, data, values: OpacityValues, inplace: bool):
        """GPU opacity implementation using PyTorch."""
        import torch

        if not inplace:
            data = data.clone()

        scale = values.scale
        is_ply = getattr(data, "is_opacities_ply", False)

        if is_ply:
            # PLY format: stored as logit(opacity)
            linear = torch.sigmoid(data.opacities)

            if scale <= 1.0:
                scaled = linear * scale
            else:
                boost_factor = (scale - 1.0) / 2.0
                scaled = linear + (1.0 - linear) * boost_factor

            scaled = torch.clamp(scaled, 1e-7, 1.0 - 1e-7)
            data.opacities = torch.logit(scaled)
        else:
            # Linear format
            if scale <= 1.0:
                data.opacities = torch.clamp(data.opacities * scale, 0.0, 1.0)
            else:
                boost_factor = (scale - 1.0) / 2.0
                data.opacities = torch.clamp(
                    data.opacities + (1.0 - data.opacities) * boost_factor,
                    0.0,
                    1.0,
                )

        return data

    # ========================================================================
    # Utility Methods
    # ========================================================================

    @staticmethod
    def _is_gpu_data(data) -> bool:
        """Check if data is GPU-based (GSTensor or GSTensorPro)."""
        # Check by attribute presence to avoid import overhead
        return hasattr(data, "device") and hasattr(data, "means") and hasattr(data.means, "device")

    def process(
        self,
        data,
        color: ColorValues | None = None,
        transform: TransformValues | None = None,
        filter_values: FilterValues | None = None,
        opacity_values: OpacityValues | None = None,
        inplace: bool = True,
    ):
        """Apply multiple processing steps in optimal order.

        Processing order: filter -> transform -> color -> opacity
        (Filter first to reduce data, transform before color for correct positions)

        :param data: GSData/GSDataPro (CPU) or GSTensor/GSTensorPro (GPU)
        :param color: Optional color parameters
        :param transform: Optional transform parameters
        :param filter_values: Optional filter parameters
        :param opacity_values: Optional opacity parameters
        :param inplace: If True, modify data in-place; if False, create copy
        :return: Processed data (same type as input)

        Example:
            >>> processor = GaussianProcessor()
            >>> result = processor.process(
            ...     data,
            ...     color=ColorValues(brightness=1.2),
            ...     transform=TransformValues.from_scale(2.0),
            ...     filter_values=FilterValues(min_opacity=0.3),
            ...     opacity_values=OpacityValues(scale=0.8),
            ... )
        """
        if not inplace:
            data = data.clone()

        # Apply in optimal order: filter -> transform -> color -> opacity
        if filter_values is not None:
            data = self.filter(data, filter_values, inplace=True)

        if transform is not None:
            data = self.transform(data, transform, inplace=True)

        if color is not None:
            data = self.color(data, color, inplace=True)

        if opacity_values is not None:
            data = self.opacity(data, opacity_values, inplace=True)

        return data


# Singleton instance for convenience
_default_processor = None


def get_processor() -> GaussianProcessor:
    """Get the default GaussianProcessor instance.

    :return: Singleton GaussianProcessor instance

    Example:
        >>> from gsmod.processing import get_processor
        >>> processor = get_processor()
        >>> data = processor.color(data, ColorValues(brightness=1.2))
    """
    global _default_processor
    if _default_processor is None:
        _default_processor = GaussianProcessor()
    return _default_processor
