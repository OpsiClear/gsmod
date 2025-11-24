"""Training-compatible learnable modules for gsmod.

This module provides nn.Module implementations with proper gradient support
for training pipelines. Use these instead of the inference-optimized
GSTensorPro methods when you need to learn parameters.

Example:
    >>> from gsmod.torch.learn import (
    ...     LearnableGSTensor, LearnableColor, LearnableTransform,
    ...     ColorGradingConfig, TransformConfig
    ... )
    >>>
    >>> # Configure what to learn
    >>> config = ColorGradingConfig(learnable=['brightness', 'saturation'])
    >>> color_model = LearnableColor(config).cuda()
    >>>
    >>> # Load data with gradient support
    >>> data = LearnableGSTensor.from_ply("scene.ply", device="cuda")
    >>>
    >>> # Apply learnable operations
    >>> result = data.apply_color(color_model)
    >>>
    >>> # Compute loss and backprop
    >>> loss = criterion(result.sh0, target)
    >>> loss.backward()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# Configuration Dataclasses
# ============================================================================


@dataclass
class ColorGradingConfig:
    """Configuration for learnable color grading.

    :param brightness: Initial brightness multiplier (1.0 = no change)
    :param contrast: Initial contrast multiplier (1.0 = no change)
    :param saturation: Initial saturation multiplier (1.0 = no change)
    :param gamma: Initial gamma value (1.0 = linear)
    :param temperature: Initial temperature (-1 to 1, 0 = neutral)
    :param vibrance: Initial vibrance multiplier (1.0 = no change)
    :param shadows: Initial shadow adjustment (-1 to 1)
    :param highlights: Initial highlight adjustment (-1 to 1)
    :param hue_shift: Initial hue shift in degrees
    :param tint: Initial green/magenta tint (-1 to 1)
    :param fade: Initial black point lift (0 to 1)
    :param shadow_tint_hue: Shadow tint hue in degrees
    :param shadow_tint_sat: Shadow tint saturation (0 to 1)
    :param highlight_tint_hue: Highlight tint hue in degrees
    :param highlight_tint_sat: Highlight tint saturation (0 to 1)
    :param learnable: List of parameter names to learn (None = all)
    :param device: Device to create tensors on ('cpu', 'cuda', etc.)

    Example:
        >>> config = ColorGradingConfig(
        ...     brightness=1.2,
        ...     learnable=['brightness', 'saturation'],
        ...     device='cuda'
        ... )
    """

    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    gamma: float = 1.0
    temperature: float = 0.0
    tint: float = 0.0
    vibrance: float = 1.0
    shadows: float = 0.0
    highlights: float = 0.0
    fade: float = 0.0
    hue_shift: float = 0.0
    shadow_tint_hue: float = 0.0
    shadow_tint_sat: float = 0.0
    highlight_tint_hue: float = 0.0
    highlight_tint_sat: float = 0.0

    learnable: list[str] | None = None
    device: str = "cuda"  # 'cuda:0', 'cuda:1', 'cpu', etc.

    def __post_init__(self):
        if self.learnable is None:
            self.learnable = [
                "brightness",
                "contrast",
                "saturation",
                "gamma",
                "temperature",
                "tint",
                "vibrance",
                "shadows",
                "highlights",
                "fade",
                "hue_shift",
                "shadow_tint_hue",
                "shadow_tint_sat",
                "highlight_tint_hue",
                "highlight_tint_sat",
            ]


@dataclass
class TransformConfig:
    """Configuration for learnable 3D transforms.

    :param translation: Initial translation (x, y, z)
    :param scale: Initial uniform scale
    :param rotation: Initial rotation as axis-angle (rx, ry, rz)
        The magnitude is the angle in radians, direction is the axis.
        Internally uses 6D continuous representation (Zhou et al. 2019) for speed.
    :param learnable: List of parameter names to learn (None = all)
        Use 'rotation' to learn rotation parameters.

    Example:
        >>> config = TransformConfig(
        ...     scale=1.0,
        ...     rotation=(0.0, 0.0, 0.785),  # 45 degrees around Z
        ...     learnable=['translation', 'scale', 'rotation']
        ... )
    """

    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    learnable: list[str] | None = None
    device: str = "cuda"  # 'cuda:0', 'cuda:1', 'cpu', etc.

    def __post_init__(self):
        if self.learnable is None:
            self.learnable = ["translation", "scale", "rotation"]


@dataclass
class LearnableFilterConfig:
    """Configuration for differentiable soft filtering.

    Uses sigmoid functions to create soft masks instead of hard boolean masks,
    allowing gradients to flow through filtering operations.

    :param geometry_type: Which geometry filter to use ('sphere', 'ellipsoid', 'box', 'none')
    :param opacity_threshold: Minimum opacity (0-1)
    :param opacity_sharpness: Sigmoid sharpness for opacity (higher = harder)
    :param scale_threshold: Maximum scale threshold
    :param scale_sharpness: Sigmoid sharpness for scale
    :param sphere_radius: Sphere filter radius (inf = no filter)
    :param sphere_center: Sphere center (x, y, z)
    :param sphere_sharpness: Sigmoid sharpness for sphere boundary
    :param ellipsoid_radii: Ellipsoid radii (rx, ry, rz)
    :param ellipsoid_center: Ellipsoid center (x, y, z)
    :param ellipsoid_rotation: Ellipsoid rotation as axis-angle
    :param ellipsoid_rotation_repr: Rotation representation ('axis_angle' or '6d')
    :param ellipsoid_sharpness: Sigmoid sharpness for ellipsoid boundary
    :param box_extents: Box half-extents (hx, hy, hz)
    :param box_center: Box center (x, y, z)
    :param box_rotation: Box rotation as axis-angle
    :param box_rotation_repr: Rotation representation ('axis_angle' or '6d')
    :param box_sharpness: Sigmoid sharpness for box boundary
    :param learnable: List of parameter names to learn (None = all)

    Example:
        >>> config = LearnableFilterConfig(
        ...     geometry_type='ellipsoid',
        ...     ellipsoid_radii=(2.0, 1.0, 1.0),
        ...     learnable=['ellipsoid_radii', 'ellipsoid_rotation']
        ... )
    """

    # Geometry filter selection
    geometry_type: str = "sphere"  # 'sphere', 'ellipsoid', 'box', 'none'

    # Attribute filters
    opacity_threshold: float = 0.0
    opacity_sharpness: float = 10.0
    scale_threshold: float = 100.0
    scale_sharpness: float = 10.0

    # Sphere filter
    sphere_radius: float = float("inf")
    sphere_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sphere_sharpness: float = 10.0

    # Ellipsoid filter (with rotation)
    ellipsoid_radii: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ellipsoid_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ellipsoid_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ellipsoid_rotation_repr: str = "axis_angle"  # 'axis_angle' or '6d'
    ellipsoid_sharpness: float = 10.0

    # Box filter (with rotation)
    box_extents: tuple[float, float, float] = (1.0, 1.0, 1.0)
    box_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    box_rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    box_rotation_repr: str = "axis_angle"  # 'axis_angle' or '6d'
    box_sharpness: float = 10.0

    learnable: list[str] | None = None
    device: str = "cuda"  # 'cuda:0', 'cuda:1', 'cpu', etc.

    def __post_init__(self):
        if self.geometry_type not in ("sphere", "ellipsoid", "box", "none"):
            raise ValueError(
                f"geometry_type must be 'sphere', 'ellipsoid', 'box', or 'none', got {self.geometry_type}"
            )
        if self.ellipsoid_rotation_repr not in ("axis_angle", "6d"):
            raise ValueError(
                f"ellipsoid_rotation_repr must be 'axis_angle' or '6d', got {self.ellipsoid_rotation_repr}"
            )
        if self.box_rotation_repr not in ("axis_angle", "6d"):
            raise ValueError(
                f"box_rotation_repr must be 'axis_angle' or '6d', got {self.box_rotation_repr}"
            )
        if self.learnable is None:
            self.learnable = [
                "opacity_threshold",
                "scale_threshold",
                "sphere_radius",
                "sphere_center",
                "ellipsoid_radii",
                "ellipsoid_center",
                "ellipsoid_rotation",
                "box_extents",
                "box_center",
                "box_rotation",
            ]


# ============================================================================
# Learnable Modules
# ============================================================================


class LearnableColor(nn.Module):
    """Learnable color grading with full gradient support.

    All operations are differentiable and avoid in-place modifications
    to maintain the computation graph for backpropagation.

    Example:
        >>> config = ColorGradingConfig(learnable=['brightness', 'saturation'])
        >>> model = LearnableColor(config).cuda()
        >>>
        >>> # Forward pass
        >>> output = model(sh0_tensor)
        >>>
        >>> # Backprop
        >>> loss = F.mse_loss(output, target)
        >>> loss.backward()
    """

    def __init__(self, config: ColorGradingConfig | None = None):
        super().__init__()
        config = config or ColorGradingConfig()
        self.config = config

        # Parameter names and their neutral values
        param_defaults = {
            "brightness": 1.0,
            "contrast": 1.0,
            "saturation": 1.0,
            "gamma": 1.0,
            "temperature": 0.0,
            "tint": 0.0,
            "vibrance": 1.0,
            "shadows": 0.0,
            "highlights": 0.0,
            "fade": 0.0,
            "hue_shift": 0.0,
            "shadow_tint_hue": 0.0,
            "shadow_tint_sat": 0.0,
            "highlight_tint_hue": 0.0,
            "highlight_tint_sat": 0.0,
        }

        # Register each parameter as learnable or fixed buffer
        for name, default in param_defaults.items():
            value = getattr(config, name, default)
            tensor = torch.tensor(float(value))

            if name in config.learnable:
                setattr(self, name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)

        # Register luminance weights as buffer (avoids recreating tensor each forward)
        self.register_buffer("lum_weights", torch.tensor([0.299, 0.587, 0.114]))

        # Move to device
        self.to(config.device)

    def forward(self, sh0: torch.Tensor) -> torch.Tensor:
        """Apply color grading (fully differentiable).

        :param sh0: Input colors [N, 3] in RGB format [0, 1]
        :return: Adjusted colors [N, 3]

        Note: Operations are skipped only for non-learnable parameters at neutral values.
        Learnable parameters are always applied to maintain gradient flow.
        """
        x = sh0
        learnable = self.config.learnable

        # Brightness (multiplicative) - skip only if not learnable and neutral
        if "brightness" in learnable or self.brightness.item() != 1.0:
            x = x * self.brightness

        # Contrast (around 0.5 midpoint) - skip only if not learnable and neutral
        if "contrast" in learnable or self.contrast.item() != 1.0:
            x = (x - 0.5) * self.contrast + 0.5

        # Gamma correction - skip only if not learnable and neutral
        if "gamma" in learnable or self.gamma.item() != 1.0:
            x = torch.pow(torch.clamp(x, min=1e-8), self.gamma)

        # Temperature (R/B balance) - skip only if not learnable and neutral
        if "temperature" in learnable or self.temperature.item() != 0.0:
            r_factor = 1.0 + self.temperature * 0.3
            b_factor = 1.0 - self.temperature * 0.3
            x = torch.stack([x[..., 0] * r_factor, x[..., 1], x[..., 2] * b_factor], dim=-1)

        # Tint (G/M balance) - skip only if not learnable and neutral
        if "tint" in learnable or self.tint.item() != 0.0:
            tint_offset_g = -self.tint * 0.1
            tint_offset_rb = self.tint * 0.05
            x = torch.stack(
                [x[..., 0] + tint_offset_rb, x[..., 1] + tint_offset_g, x[..., 2] + tint_offset_rb],
                dim=-1,
            )

        # Saturation - skip only if not learnable and neutral
        if "saturation" in learnable or self.saturation.item() != 1.0:
            gray = (x * self.lum_weights).sum(-1, keepdim=True)
            x = gray + (x - gray) * self.saturation

        # Vibrance (selective saturation) - skip only if not learnable and neutral
        if "vibrance" in learnable or self.vibrance.item() != 1.0:
            max_rgb = x.max(dim=-1, keepdim=True)[0]
            min_rgb = x.min(dim=-1, keepdim=True)[0]
            current_sat = max_rgb - min_rgb
            boost = (1.0 - current_sat) * (self.vibrance - 1.0) + 1.0
            gray = (x * self.lum_weights).sum(-1, keepdim=True)
            x = gray + (x - gray) * boost

        # Shadows/Highlights (luminance-based) - skip only if not learnable and both neutral
        shadows_active = "shadows" in learnable or self.shadows.item() != 0.0
        highlights_active = "highlights" in learnable or self.highlights.item() != 0.0
        if shadows_active or highlights_active:
            lum = (x * self.lum_weights).sum(-1, keepdim=True)
            # Soft threshold at 0.5 luminance
            shadow_mask = torch.sigmoid((0.5 - lum) * 10.0)
            highlight_mask = 1.0 - shadow_mask

            shadow_factor = shadow_mask * self.shadows
            highlight_factor = highlight_mask * self.highlights
            x = x + x * (shadow_factor + highlight_factor)

        # Fade (black point lift) - skip only if not learnable and neutral
        if "fade" in learnable or self.fade.item() != 0.0:
            x = self.fade + x * (1.0 - self.fade)

        # Hue shift (rotation in RGB space) - skip only if not learnable and neutral
        if "hue_shift" in learnable or self.hue_shift.item() != 0.0:
            angle = self.hue_shift * np.pi / 180.0
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)

            # Hue rotation matrix (simplified RGB rotation)
            # Use torch.stack to avoid UserWarning about requires_grad tensor conversion
            row0 = torch.stack(
                [
                    0.299 + 0.701 * cos_a + 0.168 * sin_a,
                    0.587 - 0.587 * cos_a + 0.330 * sin_a,
                    0.114 - 0.114 * cos_a - 0.497 * sin_a,
                ]
            )
            row1 = torch.stack(
                [
                    0.299 - 0.299 * cos_a - 0.328 * sin_a,
                    0.587 + 0.413 * cos_a + 0.035 * sin_a,
                    0.114 - 0.114 * cos_a + 0.292 * sin_a,
                ]
            )
            row2 = torch.stack(
                [
                    0.299 - 0.300 * cos_a + 1.250 * sin_a,
                    0.587 - 0.588 * cos_a - 1.050 * sin_a,
                    0.114 + 0.886 * cos_a - 0.203 * sin_a,
                ]
            )
            rot_matrix = torch.stack([row0, row1, row2])

            x = torch.matmul(x, rot_matrix.T)

        # Shadow tinting (split toning) - skip only if not learnable and saturation is zero
        shadow_tint_active = (
            "shadow_tint_hue" in learnable
            or "shadow_tint_sat" in learnable
            or self.shadow_tint_sat.item() > 0.0
        )
        if shadow_tint_active:
            lum = (x * self.lum_weights).sum(-1, keepdim=True)
            shadow_mask = torch.clamp(1.0 - lum * 2.0, 0.0, 1.0)

            # Convert hue to RGB offset
            hue_rad = self.shadow_tint_hue * np.pi / 180.0
            tint_r = torch.cos(hue_rad) * 0.5
            tint_g = torch.cos(hue_rad - 2.0 * np.pi / 3.0) * 0.5
            tint_b = torch.cos(hue_rad - 4.0 * np.pi / 3.0) * 0.5

            tint_strength = shadow_mask * self.shadow_tint_sat
            x = torch.stack(
                [
                    x[..., 0] + tint_r * tint_strength.squeeze(-1),
                    x[..., 1] + tint_g * tint_strength.squeeze(-1),
                    x[..., 2] + tint_b * tint_strength.squeeze(-1),
                ],
                dim=-1,
            )

        # Highlight tinting (split toning) - skip only if not learnable and saturation is zero
        highlight_tint_active = (
            "highlight_tint_hue" in learnable
            or "highlight_tint_sat" in learnable
            or self.highlight_tint_sat.item() > 0.0
        )
        if highlight_tint_active:
            lum = (x * self.lum_weights).sum(-1, keepdim=True)
            highlight_mask = torch.clamp(lum * 2.0 - 1.0, 0.0, 1.0)

            # Convert hue to RGB offset
            hue_rad = self.highlight_tint_hue * np.pi / 180.0
            tint_r = torch.cos(hue_rad) * 0.5
            tint_g = torch.cos(hue_rad - 2.0 * np.pi / 3.0) * 0.5
            tint_b = torch.cos(hue_rad - 4.0 * np.pi / 3.0) * 0.5

            tint_strength = highlight_mask * self.highlight_tint_sat
            x = torch.stack(
                [
                    x[..., 0] + tint_r * tint_strength.squeeze(-1),
                    x[..., 1] + tint_g * tint_strength.squeeze(-1),
                    x[..., 2] + tint_b * tint_strength.squeeze(-1),
                ],
                dim=-1,
            )

        return torch.clamp(x, 0, 1)

    @classmethod
    def from_values(cls, values, learnable: list[str] | None = None):
        """Create from ColorValues with specified learnable parameters.

        :param values: ColorValues instance
        :param learnable: Parameter names to make learnable (None = all)
        :return: LearnableColor module

        Example:
            >>> from gsmod import CINEMATIC
            >>> model = LearnableColor.from_color_values(
            ...     CINEMATIC,
            ...     learnable=['brightness', 'contrast']
            ... )
        """
        config = ColorGradingConfig(
            brightness=values.brightness,
            contrast=values.contrast,
            saturation=values.saturation,
            gamma=values.gamma,
            temperature=values.temperature,
            tint=values.tint,
            vibrance=values.vibrance,
            shadows=values.shadows,
            highlights=values.highlights,
            fade=values.fade,
            hue_shift=values.hue_shift,
            shadow_tint_hue=values.shadow_tint_hue,
            shadow_tint_sat=values.shadow_tint_sat,
            highlight_tint_hue=values.highlight_tint_hue,
            highlight_tint_sat=values.highlight_tint_sat,
            learnable=learnable,
        )
        return cls(config)

    def to_values(self):
        """Export current parameter values to ColorValues.

        :return: ColorValues with learned parameters

        Example:
            >>> learned = model.to_color_values()
            >>> save_color_json(learned, "learned_style.json")
        """
        from gsmod.config.values import ColorValues

        return ColorValues(
            brightness=self.brightness.detach().item(),
            contrast=self.contrast.detach().item(),
            saturation=self.saturation.detach().item(),
            gamma=self.gamma.detach().item(),
            temperature=self.temperature.detach().item(),
            tint=self.tint.detach().item(),
            vibrance=self.vibrance.detach().item(),
            shadows=self.shadows.detach().item(),
            highlights=self.highlights.detach().item(),
            fade=self.fade.detach().item(),
            hue_shift=self.hue_shift.detach().item(),
            shadow_tint_hue=self.shadow_tint_hue.detach().item(),
            shadow_tint_sat=self.shadow_tint_sat.detach().item(),
            highlight_tint_hue=self.highlight_tint_hue.detach().item(),
            highlight_tint_sat=self.highlight_tint_sat.detach().item(),
        )


class LearnableTransform(nn.Module):
    """Learnable 3D transform with full gradient support.

    Uses 6D continuous rotation representation (Zhou et al. 2019) internally
    for optimal performance and gradient flow. Input/output uses axis-angle
    format for user convenience.

    Example:
        >>> config = TransformConfig(
        ...     rotation=(0, 0, 0.785),  # 45 degrees around Z
        ...     learnable=['translation', 'scale', 'rotation']
        ... )
        >>> model = LearnableTransform(config).cuda()
        >>>
        >>> new_means, new_scales, new_quats = model(means, scales, quats)
    """

    def __init__(self, config: TransformConfig | None = None):
        super().__init__()
        config = config or TransformConfig()
        self.config = config

        # Translation
        translation = torch.tensor(config.translation, dtype=torch.float32)
        if "translation" in config.learnable:
            self.translation = nn.Parameter(translation)
        else:
            self.register_buffer("translation", translation)

        # Scale
        scale = torch.tensor(config.scale, dtype=torch.float32)
        if "scale" in config.learnable:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)

        # Rotation - always use 6D internally, convert from axis-angle input
        rot_6d = self._axis_angle_to_6d(torch.tensor(config.rotation, dtype=torch.float32))
        if "rotation" in config.learnable:
            self.rotation_6d = nn.Parameter(rot_6d)
        else:
            self.register_buffer("rotation_6d", rot_6d)

        # Move to device
        self.to(config.device)

    def forward(
        self, means: torch.Tensor, scales: torch.Tensor, quats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply transform (fully differentiable).

        :param means: Gaussian positions [N, 3]
        :param scales: Gaussian scales [N, 3]
        :param quats: Gaussian quaternions [N, 4] (w, x, y, z)
        :return: (new_means, new_scales, new_quats)
        """
        # Scale positions and Gaussian sizes
        new_means = means * self.scale
        new_scales = scales * self.scale

        # Rotation using 6D continuous representation
        rot_matrix = self._6d_to_rotation_matrix(self.rotation_6d)
        new_means = torch.matmul(new_means, rot_matrix.T)
        rot_quat = self._rotation_matrix_to_quat(rot_matrix)
        new_quats = self._quat_multiply(rot_quat.unsqueeze(0), quats)

        # Translation
        new_means = new_means + self.translation

        return new_means, new_scales, new_quats

    def _axis_angle_to_rotation_matrix(self, axis_angle: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle to rotation matrix directly (faster than via quaternion).

        Uses Rodrigues formula: R = I + sin(theta)*K + (1-cos(theta))*K^2
        """
        angle_sq = torch.sum(axis_angle * axis_angle)
        angle = torch.sqrt(angle_sq + 1e-12)

        # Normalized axis
        k = axis_angle / (angle + 1e-12)
        kx, ky, kz = k[0], k[1], k[2]

        # Build skew-symmetric matrix K (gradient-safe, single tensor op)
        zero = torch.zeros_like(kx)
        K = torch.stack(
            [
                torch.stack([zero, -kz, ky]),
                torch.stack([kz, zero, -kx]),
                torch.stack([-ky, kx, zero]),
            ]
        )

        sin_angle = torch.sin(angle)
        cos_angle = torch.cos(angle)
        I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)  # noqa

        return I + sin_angle * K + (1 - cos_angle) * (K @ K)

    def _axis_angle_vec_to_quat(self, axis_angle: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle vector to quaternion (differentiable at zero).

        Uses numerically stable computation that maintains gradient flow at all angles.
        For axis-angle r, quaternion is [cos(||r||/2), r * sin(||r||/2) / ||r||].
        """
        angle_sq = torch.sum(axis_angle * axis_angle)
        angle = torch.sqrt(angle_sq + 1e-12)
        half_angle = angle / 2

        # Compute sin(half_angle) / angle in a numerically stable way
        # torch.sinc(x) = sin(pi*x)/(pi*x), so sinc(half_angle/pi) = sin(half_angle)/half_angle
        # Then sin(half_angle)/angle = (sin(half_angle)/half_angle) * (half_angle/angle) = sinc * 0.5
        sinc_half = torch.sinc(half_angle / torch.pi) / 2

        w = torch.cos(half_angle)
        xyz = axis_angle * sinc_half
        return torch.cat([w.unsqueeze(0), xyz])

    def _axis_angle_to_quat(self, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle to quaternion (differentiable)."""
        half_angle = angle / 2
        w = torch.cos(half_angle)
        xyz = axis * torch.sin(half_angle)
        return torch.cat([w.unsqueeze(0), xyz])

    def _rotate_points(self, points: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """Rotate points by quaternion (differentiable)."""
        w, x, y, z = quat

        # Rotation matrix from quaternion
        rot = torch.stack(
            [
                torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
                torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
                torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
            ]
        )

        return torch.matmul(points, rot.T)

    def _quat_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply quaternions (differentiable)."""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w, x, y, z], dim=-1)

    def _axis_angle_to_6d(self, axis_angle: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle to 6D representation for initialization."""
        angle_sq = torch.sum(axis_angle * axis_angle)
        angle = torch.sqrt(angle_sq + 1e-12)

        if angle < 1e-8:
            # Identity rotation: first two columns of I
            return torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)

        # Build rotation matrix from axis-angle
        axis = axis_angle / angle
        K = torch.tensor(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
            dtype=torch.float32,
        )
        R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

        # Extract first two columns as 6D
        return torch.cat([R[:, 0], R[:, 1]])

    def _6d_to_rotation_matrix(self, rot_6d: torch.Tensor) -> torch.Tensor:
        """Convert 6D representation to rotation matrix via Gram-Schmidt.

        This is the continuous representation from Zhou et al. 2019.
        """
        # Split into two 3D vectors
        v1 = rot_6d[:3]
        v2 = rot_6d[3:6]

        # Gram-Schmidt orthogonalization
        col1 = v1 / (torch.norm(v1) + 1e-8)
        col2 = v2 - torch.dot(col1, v2) * col1
        col2 = col2 / (torch.norm(col2) + 1e-8)
        col3 = torch.linalg.cross(col1, col2)

        return torch.stack([col1, col2, col3], dim=1)

    def _rotation_matrix_to_quat(self, R: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to quaternion (differentiable)."""
        # Shepperd's method for numerical stability
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = 0.5 / torch.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return torch.stack([w, x, y, z])

    @classmethod
    def from_values(cls, values, learnable: list[str] | None = None):
        """Create from TransformValues.

        :param values: TransformValues with initial parameters
        :param learnable: List of parameters to learn
        """
        from gsmod.transform.api import quaternion_to_euler

        # Convert quaternion to axis-angle
        euler = quaternion_to_euler(np.array(values.rotation))

        config = TransformConfig(
            translation=values.translation,
            scale=values.scale,
            rotation=tuple(euler),
            learnable=learnable,
        )
        return cls(config)

    def get_rotation_axis_angle(self) -> np.ndarray:
        """Get current rotation as axis-angle vector.

        :return: Axis-angle (rx, ry, rz) where magnitude is angle in radians
        """
        # Convert 6D to rotation matrix
        rot_matrix = self._6d_to_rotation_matrix(self.rotation_6d)
        R = rot_matrix.detach().cpu().numpy()

        # Convert rotation matrix to axis-angle
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        if angle < 1e-8:
            return np.zeros(3)

        # Extract axis from skew-symmetric part
        axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        axis = axis / (2 * np.sin(angle) + 1e-12)
        axis = axis / (np.linalg.norm(axis) + 1e-12)

        return axis * angle

    def to_values(self):
        """Export current parameter values to TransformValues."""
        from gsmod.config.values import TransformValues

        # Convert 6D to rotation matrix, then to quaternion
        rot_matrix = self._6d_to_rotation_matrix(self.rotation_6d)
        quat_tensor = self._rotation_matrix_to_quat(rot_matrix)
        quat = quat_tensor.detach().cpu().numpy()

        return TransformValues(
            scale=self.scale.detach().item(),
            rotation=tuple(quat),
            translation=tuple(self.translation.detach().cpu().numpy()),
        )


class LearnableFilter(nn.Module):
    """Differentiable soft filtering using sigmoid masks.

    Instead of hard boolean masks (which break gradients), this uses
    smooth sigmoid functions to create soft [0, 1] weights.

    Example:
        >>> config = LearnableFilterConfig(opacity_threshold=0.1, opacity_sharpness=20)
        >>> model = LearnableFilter(config).cuda()
        >>>
        >>> # Get soft weights
        >>> weights = model(means, opacities, scales)
        >>>
        >>> # Use weights to modulate opacity
        >>> weighted_opacities = opacities * weights.unsqueeze(-1)
    """

    def __init__(self, config: LearnableFilterConfig | None = None):
        super().__init__()
        config = config or LearnableFilterConfig()
        self.config = config
        self.geometry_type = config.geometry_type

        # Opacity threshold
        opacity_threshold = torch.tensor(config.opacity_threshold, dtype=torch.float32)
        if "opacity_threshold" in config.learnable:
            self.opacity_threshold = nn.Parameter(opacity_threshold)
        else:
            self.register_buffer("opacity_threshold", opacity_threshold)

        # Opacity sharpness (typically fixed)
        self.register_buffer(
            "opacity_sharpness", torch.tensor(config.opacity_sharpness, dtype=torch.float32)
        )

        # Scale threshold
        scale_threshold = torch.tensor(config.scale_threshold, dtype=torch.float32)
        if "scale_threshold" in config.learnable:
            self.scale_threshold = nn.Parameter(scale_threshold)
        else:
            self.register_buffer("scale_threshold", scale_threshold)

        # Scale sharpness
        self.register_buffer(
            "scale_sharpness", torch.tensor(config.scale_sharpness, dtype=torch.float32)
        )

        # Sphere parameters
        sphere_radius = torch.tensor(config.sphere_radius, dtype=torch.float32)
        if "sphere_radius" in config.learnable:
            self.sphere_radius = nn.Parameter(sphere_radius)
        else:
            self.register_buffer("sphere_radius", sphere_radius)

        sphere_center = torch.tensor(config.sphere_center, dtype=torch.float32)
        if "sphere_center" in config.learnable:
            self.sphere_center = nn.Parameter(sphere_center)
        else:
            self.register_buffer("sphere_center", sphere_center)

        self.register_buffer(
            "sphere_sharpness", torch.tensor(config.sphere_sharpness, dtype=torch.float32)
        )

        # Ellipsoid parameters
        ellipsoid_radii = torch.tensor(config.ellipsoid_radii, dtype=torch.float32)
        if "ellipsoid_radii" in config.learnable:
            self.ellipsoid_radii = nn.Parameter(ellipsoid_radii)
        else:
            self.register_buffer("ellipsoid_radii", ellipsoid_radii)

        ellipsoid_center = torch.tensor(config.ellipsoid_center, dtype=torch.float32)
        if "ellipsoid_center" in config.learnable:
            self.ellipsoid_center = nn.Parameter(ellipsoid_center)
        else:
            self.register_buffer("ellipsoid_center", ellipsoid_center)

        # Ellipsoid rotation
        self.ellipsoid_rotation_repr = config.ellipsoid_rotation_repr
        if self.ellipsoid_rotation_repr == "axis_angle":
            ellipsoid_rotation = torch.tensor(config.ellipsoid_rotation, dtype=torch.float32)
            if "ellipsoid_rotation" in config.learnable:
                self.ellipsoid_rotation = nn.Parameter(ellipsoid_rotation)
            else:
                self.register_buffer("ellipsoid_rotation", ellipsoid_rotation)
        else:  # '6d'
            ellipsoid_rot_6d = self._axis_angle_to_6d(
                torch.tensor(config.ellipsoid_rotation, dtype=torch.float32)
            )
            if "ellipsoid_rotation" in config.learnable:
                self.ellipsoid_rotation_6d = nn.Parameter(ellipsoid_rot_6d)
            else:
                self.register_buffer("ellipsoid_rotation_6d", ellipsoid_rot_6d)

        self.register_buffer(
            "ellipsoid_sharpness", torch.tensor(config.ellipsoid_sharpness, dtype=torch.float32)
        )

        # Box parameters
        box_extents = torch.tensor(config.box_extents, dtype=torch.float32)
        if "box_extents" in config.learnable:
            self.box_extents = nn.Parameter(box_extents)
        else:
            self.register_buffer("box_extents", box_extents)

        box_center = torch.tensor(config.box_center, dtype=torch.float32)
        if "box_center" in config.learnable:
            self.box_center = nn.Parameter(box_center)
        else:
            self.register_buffer("box_center", box_center)

        # Box rotation
        self.box_rotation_repr = config.box_rotation_repr
        if self.box_rotation_repr == "axis_angle":
            box_rotation = torch.tensor(config.box_rotation, dtype=torch.float32)
            if "box_rotation" in config.learnable:
                self.box_rotation = nn.Parameter(box_rotation)
            else:
                self.register_buffer("box_rotation", box_rotation)
        else:  # '6d'
            box_rot_6d = self._axis_angle_to_6d(
                torch.tensor(config.box_rotation, dtype=torch.float32)
            )
            if "box_rotation" in config.learnable:
                self.box_rotation_6d = nn.Parameter(box_rot_6d)
            else:
                self.register_buffer("box_rotation_6d", box_rot_6d)

        self.register_buffer(
            "box_sharpness", torch.tensor(config.box_sharpness, dtype=torch.float32)
        )

        # Move to device
        self.to(config.device)

    def forward(
        self, means: torch.Tensor, opacities: torch.Tensor, scales: torch.Tensor
    ) -> torch.Tensor:
        """Compute soft filter weights (differentiable).

        :param means: Gaussian positions [N, 3]
        :param opacities: Gaussian opacities [N] or [N, 1]
        :param scales: Gaussian scales [N, 3]
        :return: Soft weights [N] in [0, 1]

        Example:
            >>> weights = model(means, opacities, scales)
            >>> # Use as opacity multiplier
            >>> filtered_opacities = opacities * weights.unsqueeze(-1)
            >>> # Or as loss weights
            >>> loss = (weights * per_point_loss).mean()
        """
        N = len(means)
        weights = torch.ones(N, device=means.device, dtype=means.dtype)

        # Flatten opacities if needed
        opacities_flat = opacities.flatten()

        # Opacity filter (soft threshold)
        if self.opacity_threshold > 0:
            opacity_weight = torch.sigmoid(
                (opacities_flat - self.opacity_threshold) * self.opacity_sharpness
            )
            weights = weights * opacity_weight

        # Scale filter (soft threshold on max scale)
        if self.scale_threshold < 100.0:
            max_scales = scales.max(dim=-1)[0]
            scale_weight = torch.sigmoid((self.scale_threshold - max_scales) * self.scale_sharpness)
            weights = weights * scale_weight

        # Geometry filter (only one active at a time)
        if self.geometry_type == "sphere":
            if self.sphere_radius < float("inf"):
                distances = torch.norm(means - self.sphere_center, dim=1)
                geo_weight = torch.sigmoid((self.sphere_radius - distances) * self.sphere_sharpness)
                weights = weights * geo_weight

        elif self.geometry_type == "ellipsoid":
            # Get rotation matrix
            rot_matrix = self._get_ellipsoid_rotation_matrix()
            # Transform to local space (inverse rotation)
            local_means = (means - self.ellipsoid_center) @ rot_matrix
            # Normalized distance in ellipsoid space
            normalized = local_means / self.ellipsoid_radii
            distances = torch.norm(normalized, dim=1)
            geo_weight = torch.sigmoid((1.0 - distances) * self.ellipsoid_sharpness)
            weights = weights * geo_weight

        elif self.geometry_type == "box":
            # Get rotation matrix
            rot_matrix = self._get_box_rotation_matrix()
            # Transform to local space (inverse rotation)
            local_means = (means - self.box_center) @ rot_matrix
            # Max normalized distance (Chebyshev)
            normalized = torch.abs(local_means) / self.box_extents
            distances = normalized.max(dim=1)[0]
            geo_weight = torch.sigmoid((1.0 - distances) * self.box_sharpness)
            weights = weights * geo_weight

        return weights

    def _axis_angle_to_6d(self, axis_angle: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle to 6D representation."""
        angle_sq = torch.sum(axis_angle * axis_angle)
        angle = torch.sqrt(angle_sq + 1e-12)

        if angle < 1e-8:
            return torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32)

        axis = axis_angle / angle
        K = torch.tensor(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
            dtype=torch.float32,
        )
        R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        return torch.cat([R[:, 0], R[:, 1]])

    def _6d_to_rotation_matrix(self, rot_6d: torch.Tensor) -> torch.Tensor:
        """Convert 6D to rotation matrix via Gram-Schmidt."""
        v1 = rot_6d[:3]
        v2 = rot_6d[3:6]
        col1 = v1 / (torch.norm(v1) + 1e-8)
        col2 = v2 - torch.dot(col1, v2) * col1
        col2 = col2 / (torch.norm(col2) + 1e-8)
        col3 = torch.linalg.cross(col1, col2)
        return torch.stack([col1, col2, col3], dim=1)

    def _axis_angle_to_rotation_matrix(self, axis_angle: torch.Tensor) -> torch.Tensor:
        """Convert axis-angle to rotation matrix."""
        angle_sq = torch.sum(axis_angle * axis_angle)
        angle = torch.sqrt(angle_sq + 1e-12)

        if angle < 1e-8:
            return torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)

        # Use sinc for numerical stability
        torch.sinc(angle / torch.pi)
        K = torch.stack(
            [
                torch.stack([torch.zeros_like(angle), -axis_angle[2], axis_angle[1]]),
                torch.stack([axis_angle[2], torch.zeros_like(angle), -axis_angle[0]]),
                torch.stack([-axis_angle[1], axis_angle[0], torch.zeros_like(angle)]),
            ]
        )

        # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
        sin_angle = torch.sin(angle)
        cos_angle = torch.cos(angle)
        I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)  # noqa
        K_normalized = K / angle
        return I + sin_angle * K_normalized + (1 - cos_angle) * (K_normalized @ K_normalized)

    def _get_ellipsoid_rotation_matrix(self) -> torch.Tensor:
        """Get rotation matrix for ellipsoid (returns transpose for inverse)."""
        if self.ellipsoid_rotation_repr == "axis_angle":
            return self._axis_angle_to_rotation_matrix(self.ellipsoid_rotation)
        else:
            return self._6d_to_rotation_matrix(self.ellipsoid_rotation_6d)

    def _get_box_rotation_matrix(self) -> torch.Tensor:
        """Get rotation matrix for box (returns transpose for inverse)."""
        if self.box_rotation_repr == "axis_angle":
            return self._axis_angle_to_rotation_matrix(self.box_rotation)
        else:
            return self._6d_to_rotation_matrix(self.box_rotation_6d)

    @classmethod
    def from_values(cls, values, learnable: list[str] | None = None):
        """Create from FilterValues.

        :param values: FilterValues instance
        :param learnable: Parameter names to make learnable (None = all)
        :return: LearnableFilter module
        """
        config = LearnableFilterConfig(
            opacity_threshold=values.min_opacity,
            scale_threshold=values.max_scale,
            sphere_radius=values.sphere_radius,
            sphere_center=values.sphere_center,
            learnable=learnable,
        )
        return cls(config)

    def to_values(self):
        """Export current parameter values to FilterValues.

        :return: FilterValues with learned parameters
        """
        from gsmod.config.values import FilterValues

        return FilterValues(
            min_opacity=self.opacity_threshold.detach().item(),
            max_scale=self.scale_threshold.detach().item(),
            sphere_radius=self.sphere_radius.detach().item(),
            sphere_center=tuple(self.sphere_center.detach().cpu().numpy()),
        )


# ============================================================================
# LearnableGSTensor - Gradient-aware tensor container
# ============================================================================


class LearnableGSTensor:
    """Gaussian Splatting tensor container with native gradient support.

    Unlike GSTensorPro which uses in-place operations for inference speed,
    this class maintains the computation graph for backpropagation by:
    - Never using in-place operations
    - Returning new instances from all operations
    - Properly tracking requires_grad

    Example:
        >>> # Load with gradient tracking
        >>> data = LearnableGSTensor.from_ply("scene.ply", device="cuda")
        >>>
        >>> # Apply learnable operations
        >>> color_model = LearnableColor().cuda()
        >>> result = data.apply_color(color_model)
        >>>
        >>> # Backprop through the operations
        >>> loss = criterion(result.sh0, target)
        >>> loss.backward()
    """

    def __init__(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        opacities: torch.Tensor,
        sh0: torch.Tensor,
        shN: torch.Tensor | None = None,
    ):
        self.means = means
        self.scales = scales
        self.quats = quats
        self.opacities = opacities
        self.sh0 = sh0
        self.shN = shN

    @classmethod
    def from_gstensor_pro(
        cls, gstensor, requires_grad: bool = True, device: str | None = None
    ) -> LearnableGSTensor:
        """Create from GSTensorPro with gradient tracking.

        :param gstensor: GSTensorPro instance
        :param requires_grad: Whether to track gradients
        :param device: Device to move tensors to ('cuda:0', 'cuda:1', etc.)
        :return: LearnableGSTensor with gradient support
        """
        result = cls(
            means=gstensor.means.clone().requires_grad_(requires_grad),
            scales=gstensor.scales.clone().requires_grad_(requires_grad),
            quats=gstensor.quats.clone().requires_grad_(requires_grad),
            opacities=gstensor.opacities.clone().requires_grad_(requires_grad),
            sh0=gstensor.sh0.clone().requires_grad_(requires_grad),
            shN=(
                gstensor.shN.clone().requires_grad_(requires_grad)
                if gstensor.shN is not None
                else None
            ),
        )
        if device is not None:
            result = result.to(device)
        return result

    @classmethod
    def from_ply(
        cls, path: str | Path, device: str = "cuda", requires_grad: bool = True
    ) -> LearnableGSTensor:
        """Load from PLY file with gradient tracking.

        :param path: Path to PLY file
        :param device: Target device
        :param requires_grad: Whether to track gradients
        :return: LearnableGSTensor on device
        """
        from gsmod.torch import GSTensorPro

        gstensor = GSTensorPro.load(path, device=device)
        return cls.from_gstensor_pro(gstensor, requires_grad=requires_grad)

    def to_gstensor_pro(self):
        """Convert to GSTensorPro for inference.

        :return: GSTensorPro instance (detached from computation graph)
        """
        from gsmod.torch import GSTensorPro

        return GSTensorPro(
            means=self.means.detach(),
            scales=self.scales.detach(),
            quats=self.quats.detach(),
            opacities=self.opacities.detach(),
            sh0=self.sh0.detach(),
            shN=self.shN.detach() if self.shN is not None else None,
        )

    def apply_color(self, color_module: LearnableColor) -> LearnableGSTensor:
        """Apply learnable color grading (creates new instance).

        :param color_module: LearnableColor module
        :return: New LearnableGSTensor with modified colors
        """
        new_sh0 = color_module(self.sh0)
        return LearnableGSTensor(
            means=self.means,
            scales=self.scales,
            quats=self.quats,
            opacities=self.opacities,
            sh0=new_sh0,
            shN=self.shN,
        )

    def apply_transform(self, transform_module: LearnableTransform) -> LearnableGSTensor:
        """Apply learnable transform (creates new instance).

        :param transform_module: LearnableTransform module
        :return: New LearnableGSTensor with transformed geometry
        """
        new_means, new_scales, new_quats = transform_module(self.means, self.scales, self.quats)
        return LearnableGSTensor(
            means=new_means,
            scales=new_scales,
            quats=new_quats,
            opacities=self.opacities,
            sh0=self.sh0,
            shN=self.shN,
        )

    def apply_soft_filter(
        self, filter_module: LearnableFilter
    ) -> tuple[LearnableGSTensor, torch.Tensor]:
        """Apply soft filter, returning weights.

        :param filter_module: LearnableFilter module
        :return: (self, weights) tuple

        Use the weights to modulate opacities or as loss weights:
            >>> data, weights = data.apply_soft_filter(filter_model)
            >>> weighted_opacities = data.opacities * weights.unsqueeze(-1)
        """
        weights = filter_module(self.means, self.opacities, self.scales)
        return self, weights

    def clone(self) -> LearnableGSTensor:
        """Create a deep copy with gradient tracking preserved."""
        return LearnableGSTensor(
            means=self.means.clone(),
            scales=self.scales.clone(),
            quats=self.quats.clone(),
            opacities=self.opacities.clone(),
            sh0=self.sh0.clone(),
            shN=self.shN.clone() if self.shN is not None else None,
        )

    def detach(self) -> LearnableGSTensor:
        """Detach from computation graph."""
        return LearnableGSTensor(
            means=self.means.detach(),
            scales=self.scales.detach(),
            quats=self.quats.detach(),
            opacities=self.opacities.detach(),
            sh0=self.sh0.detach(),
            shN=self.shN.detach() if self.shN is not None else None,
        )

    @property
    def device(self) -> torch.device:
        """Get device of tensors."""
        return self.means.device

    @property
    def dtype(self) -> torch.dtype:
        """Get dtype of tensors."""
        return self.means.dtype

    def __len__(self) -> int:
        """Number of Gaussians."""
        return len(self.means)

    def to(self, device: str | torch.device) -> LearnableGSTensor:
        """Move to device."""
        return LearnableGSTensor(
            means=self.means.to(device),
            scales=self.scales.to(device),
            quats=self.quats.to(device),
            opacities=self.opacities.to(device),
            sh0=self.sh0.to(device),
            shN=self.shN.to(device) if self.shN is not None else None,
        )


# ============================================================================
# Backwards Compatibility Aliases (deprecated)
# ============================================================================

# Old names - use new names instead
LearnableColorGrading = LearnableColor
SoftFilter = LearnableFilter
GSTensorProLearn = LearnableGSTensor
SoftFilterConfig = LearnableFilterConfig
