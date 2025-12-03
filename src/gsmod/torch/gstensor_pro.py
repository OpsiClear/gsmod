"""GSTensorPro - GPU-accelerated Gaussian Splatting processing with PyTorch.

This module provides a GPU-optimized wrapper for Gaussian Splatting data, building
on top of gsply's GSTensor but adding gsmod's processing capabilities on GPU.

Takes full advantage of gsply's latest features:
- GPU I/O functions (plyread_gpu, plywrite_gpu)
- Mask layer management
- GPU compression
- Format conversion methods
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import numpy as np
import torch

# Import GSTensor from gsply
from gsply.torch import GSTensor

from gsmod.config.values import (
    ColorValues,
    FilterValues,
    HistogramConfig,
    OpacityValues,
    TransformValues,
)
from gsmod.histogram.result import HistogramResult

# Try to import GPU I/O functions if available
try:
    from gsply.torch import plyread_gpu, plywrite_gpu

    GPU_IO_AVAILABLE = True
except ImportError:
    GPU_IO_AVAILABLE = False

# Import GSData for conversions

# Import DataFormat enum for format tracking
from gsply.gsdata import DataFormat

# Import Triton-accelerated kernels
from gsmod.torch.triton_kernels import (
    triton_adjust_brightness,
    triton_adjust_contrast,
    triton_adjust_gamma,
    triton_adjust_saturation,
    triton_adjust_temperature,
)


class GSTensorPro(GSTensor):
    """GPU-accelerated Gaussian Splatting processing with PyTorch tensors.

    Extends gsply's GSTensor with gsmod's processing capabilities, providing
    GPU-accelerated color adjustments, transforms, and filtering operations.

    Attributes:
        Inherits all attributes from GSTensor:
        - means, scales, quats, opacities, sh0, shN, masks, etc.

    Performance:
        - Color adjustments: 10-100x faster than CPU
        - Transforms: 20-50x faster than CPU
        - Filtering: 50-100x faster than CPU
        - All operations leverage GPU parallelism

    Example:
        >>> import gsmod.torch as gspt
        >>> data = gsply.plyread("scene.ply")  # GSData on CPU
        >>> gstensor = gspt.GSTensorPro.from_gsdata(data, device='cuda')
        >>> # GPU color adjustments
        >>> gstensor.adjust_brightness(1.2, inplace=True)
        >>> gstensor.adjust_saturation(1.3, inplace=True)
        >>> # GPU transforms
        >>> gstensor.rotate_quaternion(quat, inplace=True)
        >>> gstensor.translate([1, 0, 0], inplace=True)
        >>> # GPU filtering
        >>> mask = gstensor.filter_within_sphere(radius=1.0)
        >>> filtered = gstensor[mask]
    """

    def __init__(self, *args, **kwargs):
        """Initialize GSTensorPro with format tracking."""
        super().__init__(*args, **kwargs)
        # Initialize format tracking - check if parent already has it
        if not hasattr(self, "_format"):
            self._format = {}

    @classmethod
    def from_gsdata(cls, data, device="cuda", dtype=None, requires_grad=False):
        """Convert GSData to GSTensorPro.

        Overrides GSTensor.from_gsdata to return GSTensorPro instead of GSTensor.
        """
        # Call parent class method to get GSTensor
        base_tensor = GSTensor.from_gsdata(data, device, dtype, requires_grad)

        # Convert GSTensor to GSTensorPro by creating new instance with same data
        result = cls(
            means=base_tensor.means,
            scales=base_tensor.scales,
            quats=base_tensor.quats,
            opacities=base_tensor.opacities,
            sh0=base_tensor.sh0,
            shN=base_tensor.shN,
            masks=base_tensor.masks if hasattr(base_tensor, "masks") else None,
            _base=base_tensor._base if hasattr(base_tensor, "_base") else None,
        )

        # Copy format tracking using public API
        if hasattr(result, "copy_format_from") and hasattr(base_tensor, "_format"):
            result.copy_format_from(base_tensor)
        elif hasattr(base_tensor, "_format"):
            result._format = base_tensor._format.copy()

        return result

    @classmethod
    def from_gstensor(
        cls,
        tensor: GSTensor,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> GSTensorPro:
        """Convert GSTensor to GSTensorPro while preserving format state.

        This is the recommended way to wrap a GSTensor as GSTensorPro for
        gsmod processing operations (color, transform, filter, opacity).

        :param tensor: Source GSTensor instance
        :param device: Optional target device (if None, keeps current device)
        :param dtype: Optional target dtype (if None, keeps current dtype)
        :returns: GSTensorPro with format state preserved

        Example:
            >>> gstensor = gsply.plyread_gpu("scene.ply", device="cuda")
            >>> gstensor = gstensor.to_rgb(inplace=True)  # is_sh0_rgb=True
            >>> pro = GSTensorPro.from_gstensor(gstensor)  # Preserves is_sh0_rgb=True
            >>> edited = pro.color(ColorValues(brightness=1.1), inplace=False)
        """
        # Handle device/dtype conversion if requested
        source = tensor
        if device is not None or dtype is not None:
            target_device = device if device is not None else tensor.means.device
            target_dtype = dtype if dtype is not None else tensor.means.dtype
            source = tensor.to(target_device, dtype=target_dtype)

        # Create GSTensorPro with same tensor data
        result = cls(
            means=source.means,
            scales=source.scales,
            quats=source.quats,
            opacities=source.opacities,
            sh0=source.sh0,
            shN=source.shN,
            masks=getattr(source, "masks", None),
            _base=getattr(source, "_base", None),
        )

        # Preserve format state (is_sh0_rgb, is_scales_ply, etc.)
        if hasattr(tensor, "_format"):
            result.copy_format_from(tensor)

        return result

    @classmethod
    def load(cls, file_path: str | Path, device: str = "cuda"):
        """Load PLY file directly to GPU using GPU-accelerated I/O.

        Uses gsply's GPU decompression for maximum performance.

        :param file_path: Path to PLY file (compressed or uncompressed)
        :param device: Target device (default: 'cuda')
        :returns: GSTensorPro on GPU

        Performance:
            - 4-5x faster than CPU load + GPU transfer for compressed files
            - Direct GPU memory allocation

        Example:
            >>> gstensor = GSTensorPro.load("scene.ply", device='cuda')
        """
        # Use GPU-accelerated loading if available
        if GPU_IO_AVAILABLE:
            base_tensor = plyread_gpu(file_path, device)
        else:
            # Fallback to CPU loading + GPU transfer
            from gsply import plyread

            data = plyread(file_path)
            base_tensor = GSTensor.from_gsdata(data, device=device)

        # Convert to GSTensorPro
        result = cls(
            means=base_tensor.means,
            scales=base_tensor.scales,
            quats=base_tensor.quats,
            opacities=base_tensor.opacities,
            sh0=base_tensor.sh0,
            shN=base_tensor.shN,
            masks=base_tensor.masks if hasattr(base_tensor, "masks") else None,
            _base=base_tensor._base if hasattr(base_tensor, "_base") else None,
        )

        # Copy format tracking using public API
        if hasattr(result, "copy_format_from") and hasattr(base_tensor, "_format"):
            result.copy_format_from(base_tensor)
        elif hasattr(base_tensor, "_format"):
            result._format = base_tensor._format.copy()

        return result

    def clone(self):
        """Create a deep copy of the GSTensorPro.

        Overrides GSTensor.clone to return GSTensorPro instead of GSTensor.
        """
        # Clone using parent method
        base_clone = super().clone()

        # Convert to GSTensorPro
        result = GSTensorPro(
            means=base_clone.means,
            scales=base_clone.scales,
            quats=base_clone.quats,
            opacities=base_clone.opacities,
            sh0=base_clone.sh0,
            shN=base_clone.shN,
            masks=base_clone.masks if hasattr(base_clone, "masks") else None,
            _base=base_clone._base if hasattr(base_clone, "_base") else None,
        )

        # Copy format tracking using public API
        if hasattr(result, "copy_format_from"):
            result.copy_format_from(self)
        elif hasattr(self, "_format"):
            result._format = self._format.copy()

        return result

    def save(self, file_path: str | Path, compressed: bool = True):
        """Save to PLY file using GPU compression.

        Uses gsply's GPU compression for maximum performance.

        :param file_path: Output file path
        :param compressed: If True, use GPU compression (default)

        Performance:
            - 4-5x faster than CPU compression
            - ~20 M Gaussians/sec throughput

        Example:
            >>> gstensor.save("output.ply", compressed=True)
        """
        if compressed and GPU_IO_AVAILABLE:
            # Use GPU-accelerated compression
            plywrite_gpu(file_path, self, compressed=True)
        else:
            # Convert to GSData and use CPU path for uncompressed
            data = self.to_gsdata()
            from gsply import plywrite

            plywrite(file_path, data)

    # ==========================================================================
    # Unified Processing Methods (New API)
    # ==========================================================================

    def color(self, values: ColorValues, inplace: bool = True) -> Self:
        """Apply color transformation.

        :param values: Color parameters
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications

        Example:
            >>> tensor.color(ColorValues(brightness=1.2, saturation=1.3))
            >>> tensor.color(ColorValues.from_k(3200) + ColorValues(contrast=1.1))
        """

        if not inplace:
            result = self.clone()
            return result.color(values, inplace=True)

        if values.is_neutral():
            return self

        # Convert to RGB if needed for certain operations
        needs_rgb = (
            values.saturation != 1.0
            or values.vibrance != 1.0
            or values.hue_shift != 0.0
            or values.temperature != 0.0
            or values.tint != 0.0
            or values.shadows != 0.0
            or values.highlights != 0.0
            or values.shadow_tint_sat != 0.0
            or values.highlight_tint_sat != 0.0
        )

        if needs_rgb and not self.is_sh0_rgb:
            self.to_rgb(inplace=True)

        # Apply operations in order
        if values.temperature != 0.0:
            self.adjust_temperature(values.temperature, inplace=True)
        if values.tint != 0.0:
            self.adjust_tint(values.tint, inplace=True)
        if values.brightness != 1.0:
            self.adjust_brightness(values.brightness, inplace=True)
        if values.contrast != 1.0:
            self.adjust_contrast(values.contrast, inplace=True)
        if values.gamma != 1.0:
            self.adjust_gamma(values.gamma, inplace=True)
        if values.saturation != 1.0:
            self.adjust_saturation(values.saturation, inplace=True)
        if values.vibrance != 1.0:
            self.adjust_vibrance(values.vibrance, inplace=True)
        if values.hue_shift != 0.0:
            self.adjust_hue_shift(values.hue_shift, inplace=True)

        # Shadows/highlights (apply using luminance-based adjustment)
        if values.shadows != 0.0 or values.highlights != 0.0:
            self._apply_shadows_highlights(values.shadows, values.highlights)

        # Fade (black point lift)
        if values.fade != 0.0:
            self.adjust_fade(values.fade, inplace=True)

        # Shadow/highlight tinting (split toning)
        if values.shadow_tint_sat != 0.0:
            self._apply_shadow_tint(values.shadow_tint_hue, values.shadow_tint_sat)
        if values.highlight_tint_sat != 0.0:
            self._apply_highlight_tint(values.highlight_tint_hue, values.highlight_tint_sat)

        return self

    def _apply_shadows_highlights(self, shadows: float, highlights: float):
        """Apply shadows and highlights adjustment (GPU-optimized).

        Uses smoothstep curves matching supersplat shader for smooth transitions.

        :param shadows: Shadow adjustment (-1 to 1)
        :param highlights: Highlight adjustment (-1 to 1)
        """
        # Calculate luminance
        luminance = torch.sum(
            self.sh0 * torch.tensor([0.299, 0.587, 0.114], device=self.device), dim=-1, keepdim=True
        )

        # Smoothstep curves matching supersplat shader
        # shadowCurve = 1.0 - smoothstep(0.0, 0.5, lum)
        # highlightCurve = smoothstep(0.5, 1.0, lum)
        t_shadow = torch.clamp(luminance / 0.5, 0.0, 1.0)
        shadow_curve = 1.0 - t_shadow * t_shadow * (3.0 - 2.0 * t_shadow)

        t_highlight = torch.clamp((luminance - 0.5) / 0.5, 0.0, 1.0)
        highlight_curve = t_highlight * t_highlight * (3.0 - 2.0 * t_highlight)

        # Multiplicative adjustment: color * (1 + shadows * shadowCurve + highlights * highlightCurve)
        adjustment = 1.0 + shadows * shadow_curve + highlights * highlight_curve
        self.sh0 = self.sh0 * adjustment

    def _apply_shadow_tint(self, hue: float, sat: float):
        """Apply shadow tinting (split toning for shadows).

        :param hue: Hue angle in degrees (-180 to 180)
        :param sat: Saturation intensity (0 to 1)
        """
        import math

        # Calculate luminance
        luminance = torch.sum(
            self.sh0 * torch.tensor([0.299, 0.587, 0.114], device=self.device), dim=-1, keepdim=True
        )

        # Shadow mask: strong in dark areas, fades in mid-tones
        shadow_mask = torch.clamp(1.0 - luminance * 2.0, 0.0, 1.0)

        # Convert hue to RGB offset
        hue_rad = hue * math.pi / 180.0
        tint_r = math.cos(hue_rad) * 0.5
        tint_g = math.cos(hue_rad - 2.0 * math.pi / 3.0) * 0.5
        tint_b = math.cos(hue_rad - 4.0 * math.pi / 3.0) * 0.5

        # Apply tint weighted by mask and saturation
        tint_strength = shadow_mask * sat
        self.sh0[:, 0] = self.sh0[:, 0] + tint_r * tint_strength.squeeze(-1)
        self.sh0[:, 1] = self.sh0[:, 1] + tint_g * tint_strength.squeeze(-1)
        self.sh0[:, 2] = self.sh0[:, 2] + tint_b * tint_strength.squeeze(-1)
        self.sh0.clamp_(0, 1)

    def _apply_highlight_tint(self, hue: float, sat: float):
        """Apply highlight tinting (split toning for highlights).

        :param hue: Hue angle in degrees (-180 to 180)
        :param sat: Saturation intensity (0 to 1)
        """
        import math

        # Calculate luminance
        luminance = torch.sum(
            self.sh0 * torch.tensor([0.299, 0.587, 0.114], device=self.device), dim=-1, keepdim=True
        )

        # Highlight mask: strong in bright areas, fades in mid-tones
        highlight_mask = torch.clamp(luminance * 2.0 - 1.0, 0.0, 1.0)

        # Convert hue to RGB offset
        hue_rad = hue * math.pi / 180.0
        tint_r = math.cos(hue_rad) * 0.5
        tint_g = math.cos(hue_rad - 2.0 * math.pi / 3.0) * 0.5
        tint_b = math.cos(hue_rad - 4.0 * math.pi / 3.0) * 0.5

        # Apply tint weighted by mask and saturation
        tint_strength = highlight_mask * sat
        self.sh0[:, 0] = self.sh0[:, 0] + tint_r * tint_strength.squeeze(-1)
        self.sh0[:, 1] = self.sh0[:, 1] + tint_g * tint_strength.squeeze(-1)
        self.sh0[:, 2] = self.sh0[:, 2] + tint_b * tint_strength.squeeze(-1)
        self.sh0.clamp_(0, 1)

    def filter(self, values: FilterValues, inplace: bool = True) -> Self:
        """Filter Gaussians based on criteria.

        Changes array sizes (N -> M where M <= N).

        :param values: Filter parameters
        :param inplace: If True, modify self; if False, return filtered copy
        :returns: Self (filtered) or filtered copy

        Example:
            >>> tensor.filter(FilterValues(min_opacity=0.3, sphere_radius=5.0))
            >>> tensor.filter(FilterValues(
            ...     ellipsoid_center=(0, 0, 0),
            ...     ellipsoid_radii=(2.0, 1.0, 1.5)
            ... ))
        """

        if not inplace:
            result = self.clone()
            return result.filter(values, inplace=True)

        if values.is_neutral():
            return self

        # Compute mask
        N = len(self.means)
        mask = torch.ones(N, dtype=torch.bool, device=self.device)

        # Opacity filtering
        if values.min_opacity > 0:
            if self.is_opacities_ply:
                threshold_logit = torch.logit(torch.tensor(values.min_opacity, device=self.device))
                mask &= self.opacities.flatten() >= threshold_logit
            else:
                mask &= self.opacities.flatten() >= values.min_opacity

        if values.max_opacity < 1.0:
            if self.is_opacities_ply:
                threshold_logit = torch.logit(torch.tensor(values.max_opacity, device=self.device))
                mask &= self.opacities.flatten() <= threshold_logit
            else:
                mask &= self.opacities.flatten() <= values.max_opacity

        # Scale filtering
        if values.min_scale > 0:
            if self.is_scales_ply:
                threshold_log = torch.log(torch.tensor(values.min_scale, device=self.device))
                max_scales = torch.max(self.scales, dim=1)[0]
                mask &= max_scales >= threshold_log
            else:
                max_scales = torch.max(self.scales, dim=1)[0]
                mask &= max_scales >= values.min_scale

        if values.max_scale < 100.0:
            if self.is_scales_ply:
                threshold_log = torch.log(torch.tensor(values.max_scale, device=self.device))
                max_scales = torch.max(self.scales, dim=1)[0]
                mask &= max_scales <= threshold_log
            else:
                max_scales = torch.max(self.scales, dim=1)[0]
                mask &= max_scales <= values.max_scale

        # Sphere filtering
        if values.sphere_radius < float("inf"):
            center = torch.tensor(values.sphere_center, dtype=self.dtype, device=self.device)
            distances = torch.norm(self.means - center, dim=1)
            mask &= distances <= values.sphere_radius

        # Box filtering (with optional rotation support)
        if values.box_min is not None and values.box_max is not None:
            box_min = torch.tensor(values.box_min, dtype=self.dtype, device=self.device)
            box_max = torch.tensor(values.box_max, dtype=self.dtype, device=self.device)

            if values.box_rot is not None:
                # Rotated box (OBB) - compute center and half extents
                center = (box_min + box_max) / 2
                half_extents = (box_max - box_min) / 2

                # Convert axis-angle to rotation matrix
                axis_angle = torch.tensor(values.box_rot, dtype=self.dtype, device=self.device)
                angle = torch.norm(axis_angle)
                if angle > 1e-8:
                    axis = axis_angle / angle
                    K = torch.tensor(
                        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
                        dtype=self.dtype,
                        device=self.device,
                    )
                    rot_matrix = (
                        torch.eye(3, dtype=self.dtype, device=self.device)
                        + torch.sin(angle) * K
                        + (1 - torch.cos(angle)) * (K @ K)
                    )
                    rot_matrix = rot_matrix.T  # Transpose for world-to-local
                else:
                    rot_matrix = torch.eye(3, dtype=self.dtype, device=self.device)

                # Transform to local coordinates and check box bounds
                delta = self.means - center
                local = delta @ rot_matrix.T
                mask &= torch.all(torch.abs(local) <= half_extents, dim=1)
            else:
                # Axis-aligned box (AABB) - simple bounds check
                mask &= torch.all(self.means >= box_min, dim=1)
                mask &= torch.all(self.means <= box_max, dim=1)

        # Ellipsoid filtering
        if values.ellipsoid_radii is not None:
            center = torch.tensor(values.ellipsoid_center, dtype=self.dtype, device=self.device)
            radii = torch.tensor(values.ellipsoid_radii, dtype=self.dtype, device=self.device)

            # Get rotation matrix
            if values.ellipsoid_rot is not None:
                # Convert axis-angle to rotation matrix
                axis_angle = torch.tensor(
                    values.ellipsoid_rot, dtype=self.dtype, device=self.device
                )
                angle = torch.norm(axis_angle)
                if angle > 1e-8:
                    axis = axis_angle / angle
                    K = torch.tensor(
                        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
                        dtype=self.dtype,
                        device=self.device,
                    )
                    rot_matrix = (
                        torch.eye(3, dtype=self.dtype, device=self.device)
                        + torch.sin(angle) * K
                        + (1 - torch.cos(angle)) * (K @ K)
                    )
                    rot_matrix = rot_matrix.T  # Transpose for world-to-local
                else:
                    rot_matrix = torch.eye(3, dtype=self.dtype, device=self.device)
            else:
                rot_matrix = torch.eye(3, dtype=self.dtype, device=self.device)

            # Transform to local coordinates and check ellipsoid distance
            delta = self.means - center
            local = delta @ rot_matrix.T
            normalized = local / radii
            dist_sq = torch.sum(normalized**2, dim=1)
            mask &= dist_sq <= 1.0

        # Frustum filtering
        if values.frustum_pos is not None:
            camera_pos = torch.tensor(values.frustum_pos, dtype=self.dtype, device=self.device)

            # Get rotation matrix
            if values.frustum_rot is not None:
                axis_angle = torch.tensor(values.frustum_rot, dtype=self.dtype, device=self.device)
                angle = torch.norm(axis_angle)
                if angle > 1e-8:
                    axis = axis_angle / angle
                    K = torch.tensor(
                        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
                        dtype=self.dtype,
                        device=self.device,
                    )
                    rot_matrix = (
                        torch.eye(3, dtype=self.dtype, device=self.device)
                        + torch.sin(angle) * K
                        + (1 - torch.cos(angle)) * (K @ K)
                    )
                    rot_matrix = rot_matrix.T  # Transpose for world-to-camera
                else:
                    rot_matrix = torch.eye(3, dtype=self.dtype, device=self.device)
            else:
                rot_matrix = torch.eye(3, dtype=self.dtype, device=self.device)

            # Transform to camera coordinates
            delta = self.means - camera_pos
            local = delta @ rot_matrix.T

            # Calculate FOV tangents
            tan_half_fov_y = torch.tan(torch.tensor(values.frustum_fov / 2, device=self.device))
            tan_half_fov_x = tan_half_fov_y * values.frustum_aspect

            # Check frustum bounds (camera looks down -Z)
            lz = local[:, 2]
            in_depth = (lz < -values.frustum_near) & (lz > -values.frustum_far)
            abs_z = -lz
            in_fov_x = torch.abs(local[:, 0]) <= abs_z * tan_half_fov_x
            in_fov_y = torch.abs(local[:, 1]) <= abs_z * tan_half_fov_y
            mask &= in_depth & in_fov_x & in_fov_y

        # Invert mask if exclude mode is requested
        if values.invert:
            mask = ~mask

        # Apply mask to all arrays
        self.means = self.means[mask]
        self.scales = self.scales[mask]
        self.quats = self.quats[mask]
        self.opacities = self.opacities[mask]
        self.sh0 = self.sh0[mask]
        if self.shN is not None:
            self.shN = self.shN[mask]
        if hasattr(self, "_base") and self._base is not None:
            self._base = self._base[mask]

        return self

    def transform(self, values: TransformValues, inplace: bool = True) -> Self:
        """Apply geometric transformation.

        :param values: Transform parameters
        :param inplace: If True, modify self; if False, return transformed copy
        :returns: Self (transformed) or transformed copy

        Example:
            >>> tensor.transform(TransformValues.from_scale(2.0))
            >>> tensor.transform(
            ...     TransformValues.from_translation(1, 0, 0) +
            ...     TransformValues.from_rotation_euler(0, 45, 0)
            ... )
        """

        if not inplace:
            result = self.clone()
            return result.transform(values, inplace=True)

        if values.is_neutral():
            return self

        # Get transformation matrix
        matrix = values.to_matrix()
        matrix_tensor = torch.tensor(matrix, dtype=self.dtype, device=self.device)

        # Apply to positions (homogeneous coordinates)
        ones = torch.ones((len(self), 1), dtype=self.dtype, device=self.device)
        homogeneous = torch.cat([self.means, ones], dim=1)
        self.means = torch.matmul(homogeneous, matrix_tensor.T)[:, :3]

        # Apply rotation to quaternions
        if not np.allclose(values.rotation, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)):
            rot_quat = torch.tensor(values.rotation, dtype=self.dtype, device=self.device)
            self.quats = self._quaternion_multiply(rot_quat.unsqueeze(0), self.quats)

        # Apply scale
        if not np.allclose(values.scale, 1.0):
            if self.is_scales_ply:
                self.scales += np.log(values.scale)
            else:
                self.scales *= values.scale

        self._base = None
        return self

    def opacity(self, values: OpacityValues, inplace: bool = True) -> Self:
        """Apply opacity adjustment.

        Handles both linear [0, 1] and PLY (logit) opacity formats correctly.
        The scale factor is always applied in linear space.

        :param values: Opacity parameters to apply
        :param inplace: If True, modify self; if False, return modified copy
        :returns: Self (modified) or copy with modifications

        Example:
            >>> tensor.opacity(OpacityValues(scale=0.5))  # Fade to 50%
            >>> tensor.opacity(OpacityValues.fade(0.7))  # Fade to 70%
            >>> tensor.opacity(OpacityValues.boost(1.5))  # Boost opacity
        """
        if not inplace:
            result = self.clone()
            return result.opacity(values, inplace=True)

        if values.is_neutral():
            return self

        scale = values.scale

        if self.is_opacities_ply:
            # PLY format: stored as logit(opacity)
            # Convert to linear, scale, convert back
            linear = torch.sigmoid(self.opacities)

            if scale <= 1.0:
                # Simple scaling for fade
                scaled = linear * scale
            else:
                # Boost: move toward 1.0 with diminishing returns
                boost_factor = (scale - 1.0) / 2.0
                scaled = linear + (1.0 - linear) * boost_factor

            # Clamp to valid range for logit (avoid inf)
            scaled = torch.clamp(scaled, 1e-7, 1.0 - 1e-7)

            # Convert back to logit
            self.opacities = torch.logit(scaled)
        else:
            # Linear format: direct scaling
            if scale <= 1.0:
                self.opacities = torch.clamp(self.opacities * scale, 0.0, 1.0)
            else:
                # Boost: move toward 1.0 with diminishing returns
                boost_factor = (scale - 1.0) / 2.0
                self.opacities = torch.clamp(
                    self.opacities + (1.0 - self.opacities) * boost_factor,
                    0.0,
                    1.0,
                )

        return self

    def to_cpu(self):
        """Convert to CPU GSDataPro.

        :returns: GSDataPro instance on CPU
        """
        from gsmod.gsdata_pro import GSDataPro

        return GSDataPro.from_gsdata(self.to_gsdata())

    # ==========================================================================
    # Format Conversion Methods (Leveraging gsply's methods)
    # ==========================================================================

    def to_rgb(self, inplace=True):
        """Convert sh0 from spherical harmonics to RGB format.

        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)
        """
        # Already in RGB format - nothing to do
        if self.is_sh0_rgb:
            return self if inplace else self.clone()

        if inplace:
            # Convert SH to RGB: rgb = sh0 * SH_C0 + 0.5
            # SH_C0 = 1 / (2 * sqrt(pi)) = 0.28209479177387814
            SH_C0 = 0.28209479177387814
            self.sh0 = self.sh0 * SH_C0 + 0.5
            self.sh0.clamp_(0, 1)
            self._format["sh0"] = DataFormat.SH0_RGB
            if hasattr(self, "_base"):
                self._base = None
            return self

        result = self.clone()
        return result.to_rgb(inplace=True)

    def to_sh(self, inplace=True):
        """Convert sh0 from RGB format back to spherical harmonics.

        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)
        """
        # Already in SH format - nothing to do
        if self.is_sh0_sh:
            return self if inplace else self.clone()

        if inplace:
            # Convert RGB to SH: sh = (rgb - 0.5) / SH_C0
            # SH_C0 = 1 / (2 * sqrt(pi)) = 0.28209479177387814
            SH_C0 = 0.28209479177387814
            self.sh0 = (self.sh0 - 0.5) / SH_C0
            self._format["sh0"] = DataFormat.SH0_SH
            if hasattr(self, "_base"):
                self._base = None
            return self

        result = self.clone()
        return result.to_sh(inplace=True)

    def _wrap_result(self, result):
        """Wrap a GSTensor result as GSTensorPro."""
        if isinstance(result, GSTensorPro):
            return result

        # Convert GSTensor to GSTensorPro
        wrapped = GSTensorPro(
            means=result.means,
            scales=result.scales,
            quats=result.quats,
            opacities=result.opacities,
            sh0=result.sh0,
            shN=result.shN,
            masks=result.masks if hasattr(result, "masks") else None,
            _base=result._base if hasattr(result, "_base") else None,
        )

        # Copy format tracking using public API
        if hasattr(wrapped, "copy_format_from") and hasattr(result, "_format"):
            wrapped.copy_format_from(result)
        elif hasattr(result, "_format"):
            wrapped._format = result._format.copy()

        return wrapped

    # ==========================================================================
    # Color Adjustment Operations (GPU-Optimized)
    # ==========================================================================

    def adjust_brightness(self, factor: float, inplace: bool = False) -> GSTensorPro:
        """Adjust brightness of colors (GPU-optimized with Triton, format-aware).

        Uses Triton-accelerated kernel when available for maximum GPU efficiency.
        Falls back to PyTorch operations if Triton is not available.

        :param factor: Brightness factor (1.0 = no change, >1.0 = brighter, <1.0 = darker)
        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)

        Example:
            >>> gstensor.adjust_brightness(1.2, inplace=True)  # 20% brighter
        """
        # Skip if neutral
        if factor == 1.0:
            return self if inplace else self.clone()

        if inplace:
            # Use Triton-accelerated brightness adjustment
            sh0_out, shN_out = triton_adjust_brightness(self.sh0, self.shN, factor)
            self.sh0 = sh0_out.clamp_(0, 1)
            if shN_out is not None:
                self.shN = shN_out
            self._base = None  # Invalidate base
            return self

        # Create copy for non-inplace
        result = self.clone()
        return result.adjust_brightness(factor, inplace=True)

    def adjust_contrast(self, factor: float, inplace: bool = False) -> GSTensorPro:
        """Adjust contrast of colors (GPU-optimized with Triton, format-aware).

        Uses Triton-accelerated kernel when available for maximum GPU efficiency.
        Adjusts contrast around midpoint. Works correctly on both SH and RGB formats.

        :param factor: Contrast factor (1.0 = no change, >1.0 = more contrast, <1.0 = less)
        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)

        Example:
            >>> gstensor.adjust_contrast(1.1, inplace=True)  # 10% more contrast
        """
        # Skip if neutral
        if factor == 1.0:
            return self if inplace else self.clone()

        if inplace:
            # Use Triton-accelerated contrast adjustment
            self.sh0 = triton_adjust_contrast(self.sh0, factor).clamp_(0, 1)
            self._base = None
            return self

        result = self.clone()
        return result.adjust_contrast(factor, inplace=True)

    def adjust_saturation(self, factor: float, inplace: bool = False) -> GSTensorPro:
        """Adjust saturation of colors (GPU-optimized with Triton, format-aware).

        Uses Triton-accelerated kernel when available for maximum GPU efficiency.
        Falls back to PyTorch operations if Triton is not available.

        Note: Saturation adjustment requires RGB format for accurate color space operations.
        Will convert to RGB if in SH format, but preserves format if already RGB.

        :param factor: Saturation factor (1.0 = no change, 0 = grayscale, >1.0 = more saturated)
        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)

        Example:
            >>> gstensor.adjust_saturation(1.3, inplace=True)  # 30% more saturated
        """
        # Skip if neutral
        if factor == 1.0:
            return self if inplace else self.clone()

        if inplace:
            # Convert to RGB if needed (saturation requires RGB color space)
            if not self.is_sh0_rgb:
                self.to_rgb(inplace=True)

            # Use Triton-accelerated saturation adjustment
            sh0_out, shN_out = triton_adjust_saturation(self.sh0, self.shN, factor)
            self.sh0 = sh0_out.clamp_(0, 1)
            if shN_out is not None:
                self.shN = shN_out
            self._base = None

            # Note: Keep in RGB format since conversion happened
            # Caller can convert back if needed using to_sh()
            return self

        result = self.clone()
        return result.adjust_saturation(factor, inplace=True)

    def adjust_gamma(self, gamma: float, inplace: bool = False) -> GSTensorPro:
        """Apply gamma correction (GPU-optimized with Triton, format-aware).

        Uses Triton-accelerated kernel when available for maximum GPU efficiency.
        Works correctly on both SH and RGB formats.

        :param gamma: Gamma value (1.0 = no change, <1.0 = brighter, >1.0 = darker)
        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)

        Example:
            >>> gstensor.adjust_gamma(0.8, inplace=True)  # Brighter mid-tones
        """
        # Skip if neutral
        if gamma == 1.0:
            return self if inplace else self.clone()

        if inplace:
            # Use Triton-accelerated gamma correction
            self.sh0 = triton_adjust_gamma(self.sh0, gamma).clamp_(0, 1)
            self._base = None
            return self

        result = self.clone()
        return result.adjust_gamma(gamma, inplace=True)

    def adjust_temperature(self, temp: float, inplace: bool = False) -> GSTensorPro:
        """Adjust color temperature (GPU-optimized with Triton, format-aware).

        Uses Triton-accelerated kernel when available for maximum GPU efficiency.
        Note: Temperature adjustment requires RGB format for accurate color temperature shifts.
        Will convert to RGB if in SH format.

        :param temp: Temperature adjustment (-1.0 = cold/blue, 0 = neutral, 1.0 = warm/orange)
        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)

        Example:
            >>> gstensor.adjust_temperature(0.2, inplace=True)  # Slightly warm
        """
        # Skip if neutral
        if temp == 0.0:
            return self if inplace else self.clone()

        if inplace:
            # Convert to RGB if needed (temperature requires RGB color space)
            if not self.is_sh0_rgb:
                self.to_rgb(inplace=True)

            # Use Triton-accelerated temperature adjustment
            self.sh0 = triton_adjust_temperature(self.sh0, temp).clamp_(0, 1)
            self._base = None
            return self

        result = self.clone()
        return result.adjust_temperature(temp, inplace=True)

    def adjust_vibrance(self, factor: float, inplace: bool = False) -> GSTensorPro:
        """Adjust vibrance (smart saturation) of colors (GPU-optimized, format-aware).

        Increases saturation more for less-saturated colors, protecting skin tones.
        Note: Vibrance requires RGB format for accurate saturation analysis.
        Will convert to RGB if in SH format, but preserves format if already RGB.

        :param factor: Vibrance factor (1.0 = no change, >1.0 = more vibrant)
        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)

        Example:
            >>> gstensor.adjust_vibrance(1.2, inplace=True)  # 20% more vibrant
        """
        # Skip if neutral
        if factor == 1.0:
            return self if inplace else self.clone()

        if inplace:
            # Convert to RGB if needed (vibrance requires RGB color space)
            if not self.is_sh0_rgb:
                self.to_rgb(inplace=True)

            # Calculate current saturation
            max_rgb = torch.max(self.sh0, dim=-1, keepdim=True)[0]
            min_rgb = torch.min(self.sh0, dim=-1, keepdim=True)[0]
            saturation = max_rgb - min_rgb

            # Calculate luminance
            luminance = torch.sum(
                self.sh0 * torch.tensor([0.299, 0.587, 0.114], device=self.device),
                dim=-1,
                keepdim=True,
            )

            # Vibrance: boost less-saturated colors more
            boost = (1.0 - saturation) * (factor - 1.0) + 1.0

            # Apply vibrance
            self.sh0 = luminance + (self.sh0 - luminance) * boost
            self.sh0.clamp_(0, 1)
            self._base = None
            return self

        result = self.clone()
        return result.adjust_vibrance(factor, inplace=True)

    def adjust_hue_shift(self, degrees: float, inplace: bool = False) -> GSTensorPro:
        """Shift hue of colors (GPU-optimized, format-aware).

        Note: Hue shift requires RGB format for proper color space rotation.
        Will convert to RGB if in SH format, but preserves format if already RGB.

        :param degrees: Hue shift in degrees (-180 to 180)
        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)

        Example:
            >>> gstensor.adjust_hue_shift(30, inplace=True)  # Shift hue by 30 degrees
        """
        # Skip if neutral
        if degrees == 0.0:
            return self if inplace else self.clone()

        if inplace:
            # Convert to RGB if needed (hue shift requires RGB color space)
            if not self.is_sh0_rgb:
                self.to_rgb(inplace=True)

            # Convert degrees to radians
            angle = torch.tensor(degrees * np.pi / 180.0, device=self.device)
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)

            # Hue rotation matrix (approximation in RGB space)
            # This is a simplified version - full HSV conversion would be more accurate
            mat = torch.tensor(
                [
                    [
                        0.299 + 0.701 * cos_a + 0.168 * sin_a,
                        0.587 - 0.587 * cos_a + 0.330 * sin_a,
                        0.114 - 0.114 * cos_a - 0.497 * sin_a,
                    ],
                    [
                        0.299 - 0.299 * cos_a - 0.328 * sin_a,
                        0.587 + 0.413 * cos_a + 0.035 * sin_a,
                        0.114 - 0.114 * cos_a + 0.292 * sin_a,
                    ],
                    [
                        0.299 - 0.300 * cos_a + 1.250 * sin_a,
                        0.587 - 0.588 * cos_a - 1.050 * sin_a,
                        0.114 + 0.886 * cos_a - 0.203 * sin_a,
                    ],
                ],
                device=self.device,
            )

            # Apply transformation
            self.sh0 = torch.matmul(self.sh0, mat.T)
            self.sh0.clamp_(0, 1)
            self._base = None
            return self

        result = self.clone()
        return result.adjust_hue_shift(degrees, inplace=True)

    def adjust_tint(self, value: float, inplace: bool = False) -> GSTensorPro:
        """Adjust green/magenta tint (GPU-optimized, format-aware).

        Complements temperature adjustment for white balance control.

        Note: Tint adjustment requires RGB format for accurate color shifts.
        Will convert to RGB if in SH format.

        :param value: Tint value (-1.0 = green, 0.0 = neutral, 1.0 = magenta)
        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)

        Example:
            >>> gstensor.adjust_tint(-0.1, inplace=True)  # Slight green tint
        """
        # Skip if neutral
        if value == 0.0:
            return self if inplace else self.clone()

        if inplace:
            # Convert to RGB if needed
            if not self.is_sh0_rgb:
                self.to_rgb(inplace=True)

            # Tint: negative = green boost, positive = magenta (reduce green)
            tint_offset_g = -value * 0.1
            tint_offset_rb = value * 0.05

            self.sh0[..., 0] += tint_offset_rb  # R
            self.sh0[..., 1] += tint_offset_g  # G
            self.sh0[..., 2] += tint_offset_rb  # B
            self.sh0.clamp_(0, 1)
            self._base = None
            return self

        result = self.clone()
        return result.adjust_tint(value, inplace=True)

    def adjust_fade(self, value: float, inplace: bool = False) -> GSTensorPro:
        """Apply fade/black point lift for film look (GPU-optimized, format-aware).

        Lifts the black point, creating a faded/matte appearance common in
        film photography.

        Note: Fade adjustment works on both SH and RGB formats.

        :param value: Fade amount (0.0 = no change, 1.0 = full lift)
        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)

        Example:
            >>> gstensor.adjust_fade(0.1, inplace=True)  # Subtle film look
        """
        # Skip if neutral
        if value == 0.0:
            return self if inplace else self.clone()

        if inplace:
            # Fade: output = fade + input * (1 - fade)
            self.sh0.mul_(1.0 - value).add_(value)
            self.sh0.clamp_(0, 1)
            self._base = None
            return self

        result = self.clone()
        return result.adjust_fade(value, inplace=True)

    # ==========================================================================
    # Transform Operations (GPU-Optimized)
    # ==========================================================================

    def translate(
        self, translation: list[float] | np.ndarray | torch.Tensor, inplace: bool = False
    ) -> GSTensorPro:
        """Translate Gaussian positions (GPU-optimized).

        :param translation: Translation vector [x, y, z]
        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)

        Example:
            >>> gstensor.translate([1.0, 0.0, 0.5], inplace=True)
        """
        # Convert to tensor
        if not isinstance(translation, torch.Tensor):
            translation = torch.tensor(translation, dtype=self.dtype, device=self.device)
        else:
            translation = translation.to(dtype=self.dtype, device=self.device)

        if translation.shape != (3,):
            raise ValueError(f"Translation must be shape (3,), got {translation.shape}")

        if inplace:
            self.means += translation
            self._base = None
            return self

        result = self.clone()
        return result.translate(translation, inplace=True)

    def scale_uniform(self, scale: float, inplace: bool = False) -> GSTensorPro:
        """Scale Gaussians uniformly (GPU-optimized).

        :param scale: Uniform scale factor
        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)

        Example:
            >>> gstensor.scale_uniform(2.0, inplace=True)  # Double size
        """
        if inplace:
            self.means *= scale
            # For log-scales, add log(scale); for linear scales, multiply by scale
            if self.is_scales_ply:
                self.scales += np.log(scale)
            else:
                self.scales *= scale
            self._base = None
            return self

        result = self.clone()
        return result.scale_uniform(scale, inplace=True)

    def scale_nonuniform(
        self, scale: list[float] | np.ndarray | torch.Tensor, inplace: bool = False
    ) -> GSTensorPro:
        """Scale Gaussians non-uniformly (GPU-optimized).

        :param scale: Scale factors [sx, sy, sz]
        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)

        Example:
            >>> gstensor.scale_nonuniform([1.0, 2.0, 0.5], inplace=True)
        """
        # Convert to tensor
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=self.dtype, device=self.device)
        else:
            scale = scale.to(dtype=self.dtype, device=self.device)

        if scale.shape != (3,):
            raise ValueError(f"Scale must be shape (3,), got {scale.shape}")

        if inplace:
            self.means *= scale
            # For log-scales, add log(scale); for linear scales, multiply by scale
            if self.is_scales_ply:
                self.scales += torch.log(scale)
            else:
                self.scales *= scale
            self._base = None
            return self

        result = self.clone()
        return result.scale_nonuniform(scale, inplace=True)

    def rotate_quaternion(
        self, quaternion: np.ndarray | torch.Tensor, inplace: bool = False
    ) -> GSTensorPro:
        """Rotate Gaussians using quaternion (GPU-optimized).

        :param quaternion: Rotation quaternion [w, x, y, z]
        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)

        Example:
            >>> import gsmod.transform.api as tf
            >>> quat = tf.axis_angle_to_quaternion([0, 1, 0], np.pi/4)
            >>> gstensor.rotate_quaternion(quat, inplace=True)
        """
        # Convert to tensor and normalize
        if not isinstance(quaternion, torch.Tensor):
            quaternion = torch.tensor(quaternion, dtype=self.dtype, device=self.device)
        else:
            quaternion = quaternion.to(dtype=self.dtype, device=self.device)

        if quaternion.shape != (4,):
            raise ValueError(f"Quaternion must be shape (4,), got {quaternion.shape}")

        # Normalize quaternion
        quaternion = quaternion / torch.norm(quaternion)

        if inplace:
            # Rotate positions
            self.means = self._rotate_points_by_quaternion(self.means, quaternion)
            # Rotate Gaussian orientations
            self.quats = self._quaternion_multiply(quaternion.unsqueeze(0), self.quats)
            self._base = None
            return self

        result = self.clone()
        return result.rotate_quaternion(quaternion, inplace=True)

    def rotate_euler(
        self,
        angles: list[float] | np.ndarray | torch.Tensor,
        order: str = "XYZ",
        inplace: bool = False,
    ) -> GSTensorPro:
        """Rotate using Euler angles (GPU-optimized).

        :param angles: Euler angles [x, y, z] in radians
        :param order: Rotation order (e.g., "XYZ", "ZYX")
        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)

        Example:
            >>> gstensor.rotate_euler([0, np.pi/4, 0], order="XYZ", inplace=True)
        """
        # Convert to quaternion
        from gsmod.transform.api import euler_to_quaternion

        if isinstance(angles, torch.Tensor):
            angles = angles.cpu().numpy()
        elif not isinstance(angles, np.ndarray):
            angles = np.array(angles)

        # Note: order parameter is currently ignored - euler_to_quaternion uses XYZ order
        quaternion = euler_to_quaternion(angles)
        return self.rotate_quaternion(quaternion, inplace=inplace)

    def rotate_axis_angle(
        self, axis: list[float] | np.ndarray | torch.Tensor, angle: float, inplace: bool = False
    ) -> GSTensorPro:
        """Rotate around axis by angle (GPU-optimized).

        :param axis: Rotation axis [x, y, z]
        :param angle: Rotation angle in radians
        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)

        Example:
            >>> gstensor.rotate_axis_angle([0, 1, 0], np.pi/4, inplace=True)
        """
        # Convert to quaternion manually - matching CPU implementation
        if isinstance(axis, torch.Tensor):
            axis = axis.cpu().numpy()
        elif not isinstance(axis, np.ndarray):
            axis = np.array(axis, dtype=np.float32)

        # Normalize axis
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            # No rotation
            return self if inplace else self.clone()

        axis = axis / axis_norm

        # Convert to quaternion using the same formula as CPU
        half_angle = angle / 2
        sin_half = np.sin(half_angle)

        w = np.cos(half_angle)
        x = axis[0] * sin_half
        y = axis[1] * sin_half
        z = axis[2] * sin_half

        quaternion = np.array([w, x, y, z], dtype=np.float32)
        return self.rotate_quaternion(quaternion, inplace=inplace)

    def transform_matrix(
        self, matrix: np.ndarray | torch.Tensor, inplace: bool = False
    ) -> GSTensorPro:
        """Apply 4x4 transformation matrix (GPU-optimized).

        :param matrix: 4x4 homogeneous transformation matrix
        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)

        Example:
            >>> transform = np.eye(4)
            >>> transform[:3, 3] = [1, 0, 0]  # Translation
            >>> gstensor.transform_matrix(transform, inplace=True)
        """
        # Convert to tensor
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.tensor(matrix, dtype=self.dtype, device=self.device)
        else:
            matrix = matrix.to(dtype=self.dtype, device=self.device)

        if matrix.shape != (4, 4):
            raise ValueError(f"Matrix must be shape (4, 4), got {matrix.shape}")

        if inplace:
            # Extract rotation, translation, scale
            rotation = matrix[:3, :3]
            matrix[:3, 3]

            # Apply to positions (homogeneous coordinates)
            ones = torch.ones((len(self), 1), dtype=self.dtype, device=self.device)
            homogeneous = torch.cat([self.means, ones], dim=1)
            self.means = torch.matmul(homogeneous, matrix.T)[:, :3]

            # Apply rotation to quaternions
            from gsmod.transform.api import rotation_matrix_to_quaternion

            rot_quat = rotation_matrix_to_quaternion(rotation.cpu().numpy())
            self.quats = self._quaternion_multiply(
                torch.tensor(rot_quat, dtype=self.dtype, device=self.device).unsqueeze(0),
                self.quats,
            )

            self._base = None
            return self

        result = self.clone()
        return result.transform_matrix(matrix, inplace=True)

    # ==========================================================================
    # Filtering Operations (GPU-Optimized with Mask Layer Support)
    # ==========================================================================

    def filter_within_sphere(
        self,
        center: list[float] | np.ndarray | torch.Tensor = None,
        radius: float = 1.0,
        save_mask: str = None,
    ) -> torch.Tensor:
        """Filter Gaussians within sphere (GPU-optimized).

        :param center: Sphere center [x, y, z] (default: origin [0, 0, 0])
        :param radius: Absolute radius in world units
        :param save_mask: Optional name to save mask as layer
        :returns: Boolean mask tensor

        Example:
            >>> mask = gstensor.filter_within_sphere(radius=5.0, save_mask="sphere")
            >>> filtered = gstensor[mask]
            >>> # Or use saved mask layer
            >>> filtered = gstensor.apply_masks(layers=["sphere"])
        """
        # Use origin by default (matching CPU)
        if center is None:
            center = torch.zeros(3, dtype=self.dtype, device=self.device)
        elif not isinstance(center, torch.Tensor):
            center = torch.tensor(center, dtype=self.dtype, device=self.device)
        else:
            center = center.to(dtype=self.dtype, device=self.device)

        # Compute distances (GPU-parallelized)
        distances = torch.norm(self.means - center, dim=1)
        mask = distances <= radius

        # Save mask as layer if requested
        if save_mask and hasattr(self, "add_mask_layer"):
            self.add_mask_layer(save_mask, mask)

        return mask

    def filter_within_box(
        self,
        min_bounds: list[float] | np.ndarray | torch.Tensor,
        max_bounds: list[float] | np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """Filter Gaussians within axis-aligned box (GPU-optimized).

        :param min_bounds: Minimum bounds [x, y, z]
        :param max_bounds: Maximum bounds [x, y, z]
        :returns: Boolean mask tensor

        Example:
            >>> mask = gstensor.filter_within_box([-1, -1, -1], [1, 1, 1])
            >>> filtered = gstensor[mask]
        """
        if not isinstance(min_bounds, torch.Tensor):
            min_bounds = torch.tensor(min_bounds, dtype=self.dtype, device=self.device)
        else:
            min_bounds = min_bounds.to(dtype=self.dtype, device=self.device)

        if not isinstance(max_bounds, torch.Tensor):
            max_bounds = torch.tensor(max_bounds, dtype=self.dtype, device=self.device)
        else:
            max_bounds = max_bounds.to(dtype=self.dtype, device=self.device)

        # Check bounds (GPU-parallelized)
        within_min = torch.all(self.means >= min_bounds, dim=1)
        within_max = torch.all(self.means <= max_bounds, dim=1)
        return within_min & within_max

    def filter_min_opacity(self, threshold: float = 0.1) -> torch.Tensor:
        """Filter Gaussians with minimum opacity (GPU-optimized).

        :param threshold: Minimum opacity threshold
        :returns: Boolean mask tensor

        Example:
            >>> mask = gstensor.filter_min_opacity(0.1)
            >>> filtered = gstensor[mask]
        """
        # Handle different opacity formats
        if self.is_opacities_ply:
            # Logit opacities - convert threshold to logit
            threshold_logit = torch.logit(torch.tensor(threshold, device=self.device))
            return self.opacities >= threshold_logit
        else:
            # Linear opacities
            return self.opacities >= threshold

    def filter_max_scale(self, threshold: float = 0.1) -> torch.Tensor:
        """Filter Gaussians with maximum scale (GPU-optimized).

        :param threshold: Maximum scale threshold
        :returns: Boolean mask tensor

        Example:
            >>> mask = gstensor.filter_max_scale(0.1)
            >>> filtered = gstensor[mask]
        """
        # Handle different scale formats
        if self.is_scales_ply:
            # Log scales - convert threshold to log
            threshold_log = torch.log(torch.tensor(threshold, device=self.device))
            max_scales = torch.max(self.scales, dim=1)[0]
            return max_scales <= threshold_log
        else:
            # Linear scales
            max_scales = torch.max(self.scales, dim=1)[0]
            return max_scales <= threshold

    # ==========================================================================
    # Helper Methods (GPU-Optimized)
    # ==========================================================================

    def _rotate_points_by_quaternion(
        self, points: torch.Tensor, quaternion: torch.Tensor
    ) -> torch.Tensor:
        """Rotate points using quaternion (GPU-optimized).

        :param points: Points tensor (N, 3)
        :param quaternion: Quaternion tensor (4,) [w, x, y, z]
        :returns: Rotated points (N, 3)
        """
        # Extract quaternion components
        w, x, y, z = quaternion

        # Compute rotation matrix from quaternion
        rotation_matrix = torch.tensor(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ],
            dtype=self.dtype,
            device=self.device,
        )

        # Apply rotation
        return torch.matmul(points, rotation_matrix.T)

    def _quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply quaternions (GPU-optimized).

        :param q1: First quaternion(s) - shape (..., 4)
        :param q2: Second quaternion(s) - shape (..., 4)
        :returns: Product quaternion(s) - shape (..., 4)
        """
        # q1 = [w1, x1, y1, z1], q2 = [w2, x2, y2, z2]
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        # Quaternion multiplication formula
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w, x, y, z], dim=-1)

    # ==========================================================================
    # Pipeline Support
    # ==========================================================================

    def apply_pipeline(self, pipeline) -> GSTensorPro:
        """Apply a gsmod pipeline to this GSTensorPro.

        :param pipeline: Pipeline object (Color, Transform, Filter, or Pipeline)
        :returns: New GSTensorPro with pipeline applied

        Example:
            >>> from gsmod import Color, Transform, Pipeline
            >>> color_pipeline = Color().brightness(1.2).saturation(1.3)
            >>> result = gstensor.apply_pipeline(color_pipeline)
        """
        # Convert to GSData, apply pipeline, convert back
        # This is a compatibility layer - future versions will have native GPU pipelines
        gsdata = self.to_gsdata()
        result_data = pipeline(gsdata)
        return GSTensorPro.from_gsdata(result_data, device=self.device, dtype=self.dtype)

    # ==========================================================================
    # Convenience Methods
    # ==========================================================================

    def apply_color_preset(
        self, preset: str, strength: float = 1.0, inplace: bool = False
    ) -> GSTensorPro:
        """Apply a color preset (GPU-optimized).

        :param preset: Preset name ("cinematic", "warm", "cool", "vibrant", "muted", "dramatic")
        :param strength: Preset strength (0.0 to 1.0)
        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)

        Example:
            >>> gstensor.apply_color_preset("cinematic", strength=0.8, inplace=True)
        """
        # Preset configurations
        presets = {
            "cinematic": {
                "brightness": 1.05,
                "contrast": 1.15,
                "saturation": 0.95,
                "gamma": 0.92,
                "temperature": 0.05,
            },
            "warm": {"brightness": 1.1, "temperature": 0.3, "saturation": 1.1, "vibrance": 1.2},
            "cool": {"brightness": 0.95, "temperature": -0.2, "saturation": 0.9, "contrast": 1.05},
            "vibrant": {"saturation": 1.3, "vibrance": 1.4, "contrast": 1.1, "brightness": 1.05},
            "muted": {"saturation": 0.7, "contrast": 0.9, "brightness": 0.95, "gamma": 1.1},
            "dramatic": {"contrast": 1.3, "saturation": 1.15, "gamma": 0.85, "vibrance": 1.2},
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")

        config = presets[preset]

        if not inplace:
            result = self.clone()
        else:
            result = self

        # Apply adjustments with strength interpolation
        for key, value in config.items():
            # Interpolate between 1.0 (no change) and target value based on strength
            if key in ["temperature"]:
                adjusted_value = value * strength
            else:
                adjusted_value = 1.0 + (value - 1.0) * strength

            if key == "brightness":
                result.adjust_brightness(adjusted_value, inplace=True)
            elif key == "contrast":
                result.adjust_contrast(adjusted_value, inplace=True)
            elif key == "saturation":
                result.adjust_saturation(adjusted_value, inplace=True)
            elif key == "gamma":
                result.adjust_gamma(adjusted_value, inplace=True)
            elif key == "temperature":
                result.adjust_temperature(adjusted_value, inplace=True)
            elif key == "vibrance":
                result.adjust_vibrance(adjusted_value, inplace=True)

        return result

    def compute_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute scene bounds (GPU-optimized).

        :returns: (min_bounds, max_bounds) tensors of shape (3,)

        Example:
            >>> min_bounds, max_bounds = gstensor.compute_bounds()
            >>> print(f"Scene bounds: {min_bounds} to {max_bounds}")
        """
        min_bounds = torch.min(self.means, dim=0)[0]
        max_bounds = torch.max(self.means, dim=0)[0]
        return min_bounds, max_bounds

    def center_at_origin(self, inplace: bool = False) -> GSTensorPro:
        """Center scene at origin (GPU-optimized).

        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)

        Example:
            >>> gstensor.center_at_origin(inplace=True)
        """
        center = torch.mean(self.means, dim=0)
        return self.translate(-center, inplace=inplace)

    def normalize_scale(self, target_size: float = 2.0, inplace: bool = False) -> GSTensorPro:
        """Normalize scene to fit in target size (GPU-optimized).

        :param target_size: Target bounding box size
        :param inplace: If True, modify in-place; if False, return new object
        :returns: GSTensorPro object (self if inplace, new otherwise)

        Example:
            >>> gstensor.normalize_scale(target_size=2.0, inplace=True)
        """
        min_bounds, max_bounds = self.compute_bounds()
        scene_size = torch.max(max_bounds - min_bounds)
        scale_factor = target_size / scene_size
        return self.scale_uniform(scale_factor.item(), inplace=inplace)

    # ==========================================================================
    # Histogram Methods (GPU-Optimized)
    # ==========================================================================

    def histogram_colors(self, config: HistogramConfig | None = None) -> HistogramResult:
        """Compute histogram of color values (GPU-accelerated).

        :param config: Histogram configuration (default: 256 bins)
        :returns: HistogramResult with counts and statistics

        Example:
            >>> result = tensor.histogram_colors()
            >>> print(f"Mean RGB: {result.mean}")
            >>> adjustment = result.to_color_values("vibrant")
            >>> tensor.color(adjustment)
        """
        if config is None:
            config = HistogramConfig()

        N = len(self)
        n_bins = config.n_bins

        # Handle empty data
        if N == 0:
            return HistogramResult.empty(n_bins, n_channels=3)

        # Ensure RGB format for histogram
        sh0 = self.sh0
        if not self.is_sh0_rgb:
            # Convert to RGB temporarily
            SH_C0 = 0.28209479177387814
            sh0 = sh0 * SH_C0 + 0.5
            sh0 = torch.clamp(sh0, 0, 1)

        # Determine range
        if config.min_value is not None:
            min_val = config.min_value
        else:
            min_val = float(sh0.min().item())

        if config.max_value is not None:
            max_val = config.max_value
        else:
            max_val = float(sh0.max().item())

        # Compute histograms per channel using torch.histc
        counts = []
        for c in range(3):
            hist = torch.histc(sh0[:, c], bins=n_bins, min=min_val, max=max_val)
            counts.append(hist)

        counts = torch.stack(counts)

        # Compute statistics
        mean = torch.mean(sh0, dim=0)
        std = torch.std(sh0, dim=0)
        min_arr = torch.min(sh0, dim=0)[0]
        max_arr = torch.max(sh0, dim=0)[0]

        # Normalize if requested
        if config.normalize:
            bin_width = (max_val - min_val) / n_bins
            counts = counts.float() / (N * bin_width)

        return HistogramResult(
            counts=counts.cpu().numpy().astype(np.int64 if not config.normalize else np.float64),
            bin_edges=np.linspace(min_val, max_val, n_bins + 1),
            mean=mean.cpu().numpy().astype(np.float64),
            std=std.cpu().numpy().astype(np.float64),
            min_val=min_arr.cpu().numpy().astype(np.float64),
            max_val=max_arr.cpu().numpy().astype(np.float64),
            n_samples=N,
        )

    def histogram_opacity(self, config: HistogramConfig | None = None) -> HistogramResult:
        """Compute histogram of opacity values (GPU-accelerated).

        :param config: Histogram configuration (default: 256 bins)
        :returns: HistogramResult with counts and statistics

        Example:
            >>> result = tensor.histogram_opacity()
            >>> print(f"Mean opacity: {result.mean}")
        """
        if config is None:
            config = HistogramConfig()

        N = len(self)
        n_bins = config.n_bins

        # Handle empty data
        if N == 0:
            return HistogramResult.empty(n_bins, n_channels=1)

        # Get opacities in linear format
        opacities = self.opacities.flatten()
        if self.is_opacities_ply:
            # Convert from logit to linear
            opacities = torch.sigmoid(opacities)

        # Determine range
        if config.min_value is not None:
            min_val = config.min_value
        else:
            min_val = float(opacities.min().item())

        if config.max_value is not None:
            max_val = config.max_value
        else:
            max_val = float(opacities.max().item())

        # Compute histogram
        counts = torch.histc(opacities, bins=n_bins, min=min_val, max=max_val)

        # Compute statistics
        mean = torch.mean(opacities)
        std = torch.std(opacities)
        min_stat = torch.min(opacities)
        max_stat = torch.max(opacities)

        # Normalize if requested
        if config.normalize:
            bin_width = (max_val - min_val) / n_bins
            counts = counts.float() / (N * bin_width)

        return HistogramResult(
            counts=counts.cpu().numpy().astype(np.int64 if not config.normalize else np.float64),
            bin_edges=np.linspace(min_val, max_val, n_bins + 1),
            mean=np.array(mean.cpu().item(), dtype=np.float64),
            std=np.array(std.cpu().item(), dtype=np.float64),
            min_val=np.array(min_stat.cpu().item(), dtype=np.float64),
            max_val=np.array(max_stat.cpu().item(), dtype=np.float64),
            n_samples=N,
        )

    def histogram_scales(self, config: HistogramConfig | None = None) -> HistogramResult:
        """Compute histogram of scale values (GPU-accelerated).

        Uses mean scale across all 3 dimensions for each Gaussian.

        :param config: Histogram configuration (default: 256 bins)
        :returns: HistogramResult with counts and statistics

        Example:
            >>> result = tensor.histogram_scales()
            >>> print(f"Mean scale: {result.mean}")
        """
        if config is None:
            config = HistogramConfig()

        N = len(self)
        n_bins = config.n_bins

        # Handle empty data
        if N == 0:
            return HistogramResult.empty(n_bins, n_channels=1)

        # Get scales in linear format and compute mean
        scales = self.scales
        if self.is_scales_ply:
            # Convert from log to linear
            scales = torch.exp(scales)

        mean_scales = torch.mean(scales, dim=1)

        # Determine range
        if config.min_value is not None:
            min_val = config.min_value
        else:
            min_val = float(mean_scales.min().item())

        if config.max_value is not None:
            max_val = config.max_value
        else:
            max_val = float(mean_scales.max().item())

        # Compute histogram
        counts = torch.histc(mean_scales, bins=n_bins, min=min_val, max=max_val)

        # Compute statistics
        mean = torch.mean(mean_scales)
        std = torch.std(mean_scales)
        min_stat = torch.min(mean_scales)
        max_stat = torch.max(mean_scales)

        # Normalize if requested
        if config.normalize:
            bin_width = (max_val - min_val) / n_bins
            counts = counts.float() / (N * bin_width)

        return HistogramResult(
            counts=counts.cpu().numpy().astype(np.int64 if not config.normalize else np.float64),
            bin_edges=np.linspace(min_val, max_val, n_bins + 1),
            mean=np.array(mean.cpu().item(), dtype=np.float64),
            std=np.array(std.cpu().item(), dtype=np.float64),
            min_val=np.array(min_stat.cpu().item(), dtype=np.float64),
            max_val=np.array(max_stat.cpu().item(), dtype=np.float64),
            n_samples=N,
        )

    def histogram_positions(
        self, config: HistogramConfig | None = None, axis: int | None = None
    ) -> HistogramResult:
        """Compute histogram of position values (GPU-accelerated).

        :param config: Histogram configuration (default: 256 bins)
        :param axis: Axis to histogram (0=X, 1=Y, 2=Z, None=distance from origin)
        :returns: HistogramResult with counts and statistics

        Example:
            >>> result = tensor.histogram_positions()  # Distance from origin
            >>> result = tensor.histogram_positions(axis=1)  # Y coordinates
        """
        if config is None:
            config = HistogramConfig()

        N = len(self)
        n_bins = config.n_bins

        # Handle empty data
        if N == 0:
            return HistogramResult.empty(n_bins, n_channels=1)

        # Extract values based on axis
        if axis is not None:
            values = self.means[:, axis]
        else:
            # Distance from origin
            values = torch.norm(self.means, dim=1)

        # Determine range
        if config.min_value is not None:
            min_val = config.min_value
        else:
            min_val = float(values.min().item())

        if config.max_value is not None:
            max_val = config.max_value
        else:
            max_val = float(values.max().item())

        # Compute histogram
        counts = torch.histc(values, bins=n_bins, min=min_val, max=max_val)

        # Compute statistics
        mean = torch.mean(values)
        std = torch.std(values)
        min_stat = torch.min(values)
        max_stat = torch.max(values)

        # Normalize if requested
        if config.normalize:
            bin_width = (max_val - min_val) / n_bins
            counts = counts.float() / (N * bin_width)

        return HistogramResult(
            counts=counts.cpu().numpy().astype(np.int64 if not config.normalize else np.float64),
            bin_edges=np.linspace(min_val, max_val, n_bins + 1),
            mean=np.array(mean.cpu().item(), dtype=np.float64),
            std=np.array(std.cpu().item(), dtype=np.float64),
            min_val=np.array(min_stat.cpu().item(), dtype=np.float64),
            max_val=np.array(max_stat.cpu().item(), dtype=np.float64),
            n_samples=N,
        )
