"""Test individual optimizations for learnable modules."""

import logging
import time

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Check CUDA availability
if not torch.cuda.is_available():
    logger.error("CUDA not available. Benchmarks require GPU.")
    exit(1)

from gsmod.torch.learn import (
    ColorGradingConfig,
    LearnableColor,
    LearnableTransform,
    TransformConfig,
)


def create_test_tensors(n: int, device: str = 'cuda'):
    """Create test tensors for benchmarking."""
    means = torch.randn(n, 3, device=device, dtype=torch.float32)
    scales = torch.rand(n, 3, device=device, dtype=torch.float32) * 0.1
    quats = torch.randn(n, 4, device=device, dtype=torch.float32)
    quats = quats / quats.norm(dim=1, keepdim=True)
    opacities = torch.rand(n, device=device, dtype=torch.float32)
    sh0 = torch.rand(n, 3, device=device, dtype=torch.float32)
    return means, scales, quats, opacities, sh0


def benchmark(func, warmup=20, iterations=200):
    """Benchmark a function, return avg time in ms."""
    # Warmup
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return (elapsed / iterations) * 1000


# =============================================================================
# Optimized LearnableColor Variants
# =============================================================================


class LearnableColorOpt1(nn.Module):
    """Optimization 1: Register luminance weights as buffer."""

    def __init__(self, config: ColorGradingConfig | None = None):
        super().__init__()
        config = config or ColorGradingConfig()
        self.config = config

        param_defaults = {
            'brightness': 1.0, 'contrast': 1.0, 'saturation': 1.0,
            'gamma': 1.0, 'temperature': 0.0, 'vibrance': 1.0,
            'shadows': 0.0, 'highlights': 0.0, 'hue_shift': 0.0,
        }

        for name, default in param_defaults.items():
            value = getattr(config, name, default)
            tensor = torch.tensor(float(value))
            if name in config.learnable:
                setattr(self, name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)

        # OPTIMIZATION: Register luminance weights once
        self.register_buffer('lum_weights', torch.tensor([0.299, 0.587, 0.114]))

        self.to(config.device)

    def forward(self, sh0: torch.Tensor) -> torch.Tensor:
        x = sh0
        x = x * self.brightness
        x = (x - 0.5) * self.contrast + 0.5
        x = torch.pow(torch.clamp(x, min=1e-8), self.gamma)

        r_factor = 1.0 + self.temperature * 0.3
        b_factor = 1.0 - self.temperature * 0.3
        x = torch.stack([
            x[..., 0] * r_factor,
            x[..., 1],
            x[..., 2] * b_factor
        ], dim=-1)

        # Use cached luminance weights
        gray = (x * self.lum_weights).sum(-1, keepdim=True)
        x = gray + (x - gray) * self.saturation

        if not torch.equal(self.vibrance, torch.tensor(1.0, device=x.device)):
            max_rgb = x.max(dim=-1, keepdim=True)[0]
            min_rgb = x.min(dim=-1, keepdim=True)[0]
            current_sat = max_rgb - min_rgb
            boost = (1.0 - current_sat) * (self.vibrance - 1.0) + 1.0
            gray = (x * self.lum_weights).sum(-1, keepdim=True)
            x = gray + (x - gray) * boost

        if not (torch.equal(self.shadows, torch.tensor(0.0, device=x.device)) and
                torch.equal(self.highlights, torch.tensor(0.0, device=x.device))):
            lum = (x * self.lum_weights).sum(-1, keepdim=True)
            shadow_mask = torch.sigmoid((0.5 - lum) * 10.0)
            highlight_mask = 1.0 - shadow_mask
            shadow_factor = shadow_mask * self.shadows
            highlight_factor = highlight_mask * self.highlights
            x = x + x * (shadow_factor + highlight_factor)

        if not torch.equal(self.hue_shift, torch.tensor(0.0, device=x.device)):
            angle = self.hue_shift * np.pi / 180.0
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            rot_matrix = torch.tensor([
                [0.299 + 0.701*cos_a + 0.168*sin_a, 0.587 - 0.587*cos_a + 0.330*sin_a, 0.114 - 0.114*cos_a - 0.497*sin_a],
                [0.299 - 0.299*cos_a - 0.328*sin_a, 0.587 + 0.413*cos_a + 0.035*sin_a, 0.114 - 0.114*cos_a + 0.292*sin_a],
                [0.299 - 0.300*cos_a + 1.250*sin_a, 0.587 - 0.588*cos_a - 1.050*sin_a, 0.114 + 0.886*cos_a - 0.203*sin_a]
            ], device=x.device, dtype=x.dtype)
            x = torch.matmul(x, rot_matrix.T)

        return torch.clamp(x, 0, 1)


class LearnableColorOpt2(nn.Module):
    """Optimization 2: Remove all conditionals (always execute)."""

    def __init__(self, config: ColorGradingConfig | None = None):
        super().__init__()
        config = config or ColorGradingConfig()
        self.config = config

        param_defaults = {
            'brightness': 1.0, 'contrast': 1.0, 'saturation': 1.0,
            'gamma': 1.0, 'temperature': 0.0, 'vibrance': 1.0,
            'shadows': 0.0, 'highlights': 0.0, 'hue_shift': 0.0,
        }

        for name, default in param_defaults.items():
            value = getattr(config, name, default)
            tensor = torch.tensor(float(value))
            if name in config.learnable:
                setattr(self, name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)

        self.register_buffer('lum_weights', torch.tensor([0.299, 0.587, 0.114]))
        self.to(config.device)

    def forward(self, sh0: torch.Tensor) -> torch.Tensor:
        x = sh0
        x = x * self.brightness
        x = (x - 0.5) * self.contrast + 0.5
        x = torch.pow(torch.clamp(x, min=1e-8), self.gamma)

        r_factor = 1.0 + self.temperature * 0.3
        b_factor = 1.0 - self.temperature * 0.3
        x = torch.stack([
            x[..., 0] * r_factor,
            x[..., 1],
            x[..., 2] * b_factor
        ], dim=-1)

        gray = (x * self.lum_weights).sum(-1, keepdim=True)
        x = gray + (x - gray) * self.saturation

        # OPTIMIZATION: Always execute vibrance (no conditional)
        max_rgb = x.max(dim=-1, keepdim=True)[0]
        min_rgb = x.min(dim=-1, keepdim=True)[0]
        current_sat = max_rgb - min_rgb
        boost = (1.0 - current_sat) * (self.vibrance - 1.0) + 1.0
        gray = (x * self.lum_weights).sum(-1, keepdim=True)
        x = gray + (x - gray) * boost

        # OPTIMIZATION: Always execute shadows/highlights (no conditional)
        lum = (x * self.lum_weights).sum(-1, keepdim=True)
        shadow_mask = torch.sigmoid((0.5 - lum) * 10.0)
        highlight_mask = 1.0 - shadow_mask
        shadow_factor = shadow_mask * self.shadows
        highlight_factor = highlight_mask * self.highlights
        x = x + x * (shadow_factor + highlight_factor)

        # OPTIMIZATION: Always execute hue_shift (no conditional)
        angle = self.hue_shift * np.pi / 180.0
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        rot_matrix = torch.tensor([
            [0.299 + 0.701*cos_a + 0.168*sin_a, 0.587 - 0.587*cos_a + 0.330*sin_a, 0.114 - 0.114*cos_a - 0.497*sin_a],
            [0.299 - 0.299*cos_a - 0.328*sin_a, 0.587 + 0.413*cos_a + 0.035*sin_a, 0.114 - 0.114*cos_a + 0.292*sin_a],
            [0.299 - 0.300*cos_a + 1.250*sin_a, 0.587 - 0.588*cos_a - 1.050*sin_a, 0.114 + 0.886*cos_a - 0.203*sin_a]
        ], device=x.device, dtype=x.dtype)
        x = torch.matmul(x, rot_matrix.T)

        return torch.clamp(x, 0, 1)


class LearnableColorOpt3(nn.Module):
    """Optimization 3: Optimize temperature with broadcasting."""

    def __init__(self, config: ColorGradingConfig | None = None):
        super().__init__()
        config = config or ColorGradingConfig()
        self.config = config

        param_defaults = {
            'brightness': 1.0, 'contrast': 1.0, 'saturation': 1.0,
            'gamma': 1.0, 'temperature': 0.0, 'vibrance': 1.0,
            'shadows': 0.0, 'highlights': 0.0, 'hue_shift': 0.0,
        }

        for name, default in param_defaults.items():
            value = getattr(config, name, default)
            tensor = torch.tensor(float(value))
            if name in config.learnable:
                setattr(self, name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)

        self.register_buffer('lum_weights', torch.tensor([0.299, 0.587, 0.114]))
        self.to(config.device)

    def forward(self, sh0: torch.Tensor) -> torch.Tensor:
        x = sh0
        x = x * self.brightness
        x = (x - 0.5) * self.contrast + 0.5
        x = torch.pow(torch.clamp(x, min=1e-8), self.gamma)

        # OPTIMIZATION: Use broadcasting instead of stack
        temp_factors = torch.stack([
            1.0 + self.temperature * 0.3,
            torch.ones_like(self.temperature),
            1.0 - self.temperature * 0.3
        ])
        x = x * temp_factors

        gray = (x * self.lum_weights).sum(-1, keepdim=True)
        x = gray + (x - gray) * self.saturation

        max_rgb = x.max(dim=-1, keepdim=True)[0]
        min_rgb = x.min(dim=-1, keepdim=True)[0]
        current_sat = max_rgb - min_rgb
        boost = (1.0 - current_sat) * (self.vibrance - 1.0) + 1.0
        gray = (x * self.lum_weights).sum(-1, keepdim=True)
        x = gray + (x - gray) * boost

        lum = (x * self.lum_weights).sum(-1, keepdim=True)
        shadow_mask = torch.sigmoid((0.5 - lum) * 10.0)
        highlight_mask = 1.0 - shadow_mask
        shadow_factor = shadow_mask * self.shadows
        highlight_factor = highlight_mask * self.highlights
        x = x + x * (shadow_factor + highlight_factor)

        angle = self.hue_shift * np.pi / 180.0
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        rot_matrix = torch.tensor([
            [0.299 + 0.701*cos_a + 0.168*sin_a, 0.587 - 0.587*cos_a + 0.330*sin_a, 0.114 - 0.114*cos_a - 0.497*sin_a],
            [0.299 - 0.299*cos_a - 0.328*sin_a, 0.587 + 0.413*cos_a + 0.035*sin_a, 0.114 - 0.114*cos_a + 0.292*sin_a],
            [0.299 - 0.300*cos_a + 1.250*sin_a, 0.587 - 0.588*cos_a - 1.050*sin_a, 0.114 + 0.886*cos_a - 0.203*sin_a]
        ], device=x.device, dtype=x.dtype)
        x = torch.matmul(x, rot_matrix.T)

        return torch.clamp(x, 0, 1)


class LearnableColorOpt4(nn.Module):
    """Optimization 4: torch.compile on forward."""

    def __init__(self, config: ColorGradingConfig | None = None):
        super().__init__()
        config = config or ColorGradingConfig()
        self.config = config

        param_defaults = {
            'brightness': 1.0, 'contrast': 1.0, 'saturation': 1.0,
            'gamma': 1.0, 'temperature': 0.0, 'vibrance': 1.0,
            'shadows': 0.0, 'highlights': 0.0, 'hue_shift': 0.0,
        }

        for name, default in param_defaults.items():
            value = getattr(config, name, default)
            tensor = torch.tensor(float(value))
            if name in config.learnable:
                setattr(self, name, nn.Parameter(tensor))
            else:
                self.register_buffer(name, tensor)

        self.register_buffer('lum_weights', torch.tensor([0.299, 0.587, 0.114]))
        self.to(config.device)

    def forward(self, sh0: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(sh0)

    @torch.compile(mode="reduce-overhead")
    def _forward_impl(self, sh0: torch.Tensor) -> torch.Tensor:
        x = sh0
        x = x * self.brightness
        x = (x - 0.5) * self.contrast + 0.5
        x = torch.pow(torch.clamp(x, min=1e-8), self.gamma)

        temp_factors = torch.stack([
            1.0 + self.temperature * 0.3,
            torch.ones_like(self.temperature),
            1.0 - self.temperature * 0.3
        ])
        x = x * temp_factors

        gray = (x * self.lum_weights).sum(-1, keepdim=True)
        x = gray + (x - gray) * self.saturation

        max_rgb = x.max(dim=-1, keepdim=True)[0]
        min_rgb = x.min(dim=-1, keepdim=True)[0]
        current_sat = max_rgb - min_rgb
        boost = (1.0 - current_sat) * (self.vibrance - 1.0) + 1.0
        gray = (x * self.lum_weights).sum(-1, keepdim=True)
        x = gray + (x - gray) * boost

        lum = (x * self.lum_weights).sum(-1, keepdim=True)
        shadow_mask = torch.sigmoid((0.5 - lum) * 10.0)
        highlight_mask = 1.0 - shadow_mask
        shadow_factor = shadow_mask * self.shadows
        highlight_factor = highlight_mask * self.highlights
        x = x + x * (shadow_factor + highlight_factor)

        angle = self.hue_shift * np.pi / 180.0
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        rot_matrix = torch.tensor([
            [0.299 + 0.701*cos_a + 0.168*sin_a, 0.587 - 0.587*cos_a + 0.330*sin_a, 0.114 - 0.114*cos_a - 0.497*sin_a],
            [0.299 - 0.299*cos_a - 0.328*sin_a, 0.587 + 0.413*cos_a + 0.035*sin_a, 0.114 - 0.114*cos_a + 0.292*sin_a],
            [0.299 - 0.300*cos_a + 1.250*sin_a, 0.587 - 0.588*cos_a - 1.050*sin_a, 0.114 + 0.886*cos_a - 0.203*sin_a]
        ], device=x.device, dtype=x.dtype)
        x = torch.matmul(x, rot_matrix.T)

        return torch.clamp(x, 0, 1)


# =============================================================================
# Optimized LearnableTransform Variants
# =============================================================================


class LearnableTransformOpt1(nn.Module):
    """Optimization 1: Cache rotation matrix, compute once."""

    def __init__(self, config: TransformConfig | None = None):
        super().__init__()
        config = config or TransformConfig()
        self.config = config
        self.rotation_repr = config.rotation_repr

        learnable = list(config.learnable)
        if 'rotation' in learnable:
            learnable.remove('rotation')
            learnable.append('rotation_axis_angle')

        translation = torch.tensor(config.translation, dtype=torch.float32)
        if 'translation' in learnable:
            self.translation = nn.Parameter(translation)
        else:
            self.register_buffer('translation', translation)

        scale = torch.tensor(config.scale, dtype=torch.float32)
        if 'scale' in learnable:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer('scale', scale)

        if self.rotation_repr == 'axis_angle':
            rotation = torch.tensor(config.rotation_axis_angle, dtype=torch.float32)
            if 'rotation_axis_angle' in learnable:
                self.rotation_axis_angle = nn.Parameter(rotation)
            else:
                self.register_buffer('rotation_axis_angle', rotation)

        self.to(config.device)

    def forward(self, means, scales, quats):
        new_means = means * self.scale
        new_scales = scales * self.scale

        # OPTIMIZATION: Compute rotation matrix directly from axis-angle (once)
        rot_matrix = self._axis_angle_to_rotation_matrix(self.rotation_axis_angle)

        # Apply rotation to points using the matrix
        new_means = torch.matmul(new_means, rot_matrix.T)

        # Convert rotation matrix to quaternion for quaternion multiplication
        rot_quat = self._rotation_matrix_to_quat(rot_matrix)
        new_quats = self._quat_multiply(rot_quat.unsqueeze(0), quats)

        new_means = new_means + self.translation
        return new_means, new_scales, new_quats

    def _axis_angle_to_rotation_matrix(self, axis_angle):
        """Convert axis-angle to rotation matrix directly."""
        angle_sq = torch.sum(axis_angle * axis_angle)
        angle = torch.sqrt(angle_sq + 1e-12)

        # Rodrigues formula
        k = axis_angle / (angle + 1e-12)
        K = torch.stack([
            torch.stack([torch.zeros_like(angle), -k[2], k[1]]),
            torch.stack([k[2], torch.zeros_like(angle), -k[0]]),
            torch.stack([-k[1], k[0], torch.zeros_like(angle)])
        ])

        sin_angle = torch.sin(angle)
        cos_angle = torch.cos(angle)
        I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)

        return I + sin_angle * K + (1 - cos_angle) * (K @ K)

    def _rotation_matrix_to_quat(self, R):
        """Convert rotation matrix to quaternion."""
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = 0.5 / torch.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
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

    def _quat_multiply(self, q1, q2):
        """Multiply quaternions."""
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return torch.stack([w, x, y, z], dim=-1)


class LearnableTransformOpt2(nn.Module):
    """Optimization 2: torch.compile on forward."""

    def __init__(self, config: TransformConfig | None = None):
        super().__init__()
        config = config or TransformConfig()
        self.config = config
        self.rotation_repr = config.rotation_repr

        learnable = list(config.learnable)
        if 'rotation' in learnable:
            learnable.remove('rotation')
            learnable.append('rotation_axis_angle')

        translation = torch.tensor(config.translation, dtype=torch.float32)
        if 'translation' in learnable:
            self.translation = nn.Parameter(translation)
        else:
            self.register_buffer('translation', translation)

        scale = torch.tensor(config.scale, dtype=torch.float32)
        if 'scale' in learnable:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer('scale', scale)

        if self.rotation_repr == 'axis_angle':
            rotation = torch.tensor(config.rotation_axis_angle, dtype=torch.float32)
            if 'rotation_axis_angle' in learnable:
                self.rotation_axis_angle = nn.Parameter(rotation)
            else:
                self.register_buffer('rotation_axis_angle', rotation)

        self.to(config.device)

    def forward(self, means, scales, quats):
        return self._forward_impl(means, scales, quats)

    @torch.compile(mode="reduce-overhead")
    def _forward_impl(self, means, scales, quats):
        new_means = means * self.scale
        new_scales = scales * self.scale

        rot_matrix = self._axis_angle_to_rotation_matrix(self.rotation_axis_angle)
        new_means = torch.matmul(new_means, rot_matrix.T)
        rot_quat = self._rotation_matrix_to_quat(rot_matrix)
        new_quats = self._quat_multiply(rot_quat.unsqueeze(0), quats)

        new_means = new_means + self.translation
        return new_means, new_scales, new_quats

    def _axis_angle_to_rotation_matrix(self, axis_angle):
        angle_sq = torch.sum(axis_angle * axis_angle)
        angle = torch.sqrt(angle_sq + 1e-12)

        k = axis_angle / (angle + 1e-12)
        K = torch.stack([
            torch.stack([torch.zeros_like(angle), -k[2], k[1]]),
            torch.stack([k[2], torch.zeros_like(angle), -k[0]]),
            torch.stack([-k[1], k[0], torch.zeros_like(angle)])
        ])

        sin_angle = torch.sin(angle)
        cos_angle = torch.cos(angle)
        I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)

        return I + sin_angle * K + (1 - cos_angle) * (K @ K)

    def _rotation_matrix_to_quat(self, R):
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = 0.5 / torch.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
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

    def _quat_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return torch.stack([w, x, y, z], dim=-1)


# =============================================================================
# Main Benchmark
# =============================================================================


def run_benchmarks():
    """Run optimization benchmarks."""
    logger.info("=" * 70)
    logger.info("OPTIMIZATION BENCHMARKS")
    logger.info("=" * 70)
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info("")

    n = 1_000_000
    means, scales, quats, opacities, sh0 = create_test_tensors(n)

    # =========================================================================
    # LearnableColor Benchmarks
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("LearnableColor Optimizations")
    logger.info("=" * 70)

    config = ColorGradingConfig(learnable=['brightness', 'saturation', 'contrast'])

    # Baseline
    baseline = LearnableColor(config).cuda()
    baseline_ms = benchmark(lambda: baseline(sh0))
    baseline_throughput = (n / baseline_ms) * 1000 / 1e6
    logger.info(f"Baseline:           {baseline_ms:.3f} ms ({baseline_throughput:.1f}M/sec)")

    # Opt1: Luminance buffer
    opt1 = LearnableColorOpt1(config).cuda()
    opt1_ms = benchmark(lambda: opt1(sh0))
    opt1_throughput = (n / opt1_ms) * 1000 / 1e6
    speedup = baseline_ms / opt1_ms
    status = "[OK]" if speedup > 1.05 else "[--]"
    logger.info(f"Opt1 (lum buffer):  {opt1_ms:.3f} ms ({opt1_throughput:.1f}M/sec) {speedup:.2f}x {status}")

    # Opt2: Remove conditionals
    opt2 = LearnableColorOpt2(config).cuda()
    opt2_ms = benchmark(lambda: opt2(sh0))
    opt2_throughput = (n / opt2_ms) * 1000 / 1e6
    speedup = baseline_ms / opt2_ms
    status = "[OK]" if speedup > 1.05 else "[--]"
    logger.info(f"Opt2 (no cond):     {opt2_ms:.3f} ms ({opt2_throughput:.1f}M/sec) {speedup:.2f}x {status}")

    # Opt3: Temperature optimization
    opt3 = LearnableColorOpt3(config).cuda()
    opt3_ms = benchmark(lambda: opt3(sh0))
    opt3_throughput = (n / opt3_ms) * 1000 / 1e6
    speedup = baseline_ms / opt3_ms
    status = "[OK]" if speedup > 1.05 else "[--]"
    logger.info(f"Opt3 (temp bcast):  {opt3_ms:.3f} ms ({opt3_throughput:.1f}M/sec) {speedup:.2f}x {status}")

    # Opt4: torch.compile
    try:
        opt4 = LearnableColorOpt4(config).cuda()
        # Extra warmup for compilation
        for _ in range(50):
            opt4(sh0)
        torch.cuda.synchronize()
        opt4_ms = benchmark(lambda: opt4(sh0))
        opt4_throughput = (n / opt4_ms) * 1000 / 1e6
        speedup = baseline_ms / opt4_ms
        status = "[OK]" if speedup > 1.05 else "[--]"
        logger.info(f"Opt4 (compile):     {opt4_ms:.3f} ms ({opt4_throughput:.1f}M/sec) {speedup:.2f}x {status}")
    except Exception as e:
        logger.info(f"Opt4 (compile):     FAILED - {e}")

    # =========================================================================
    # LearnableTransform Benchmarks
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("LearnableTransform Optimizations")
    logger.info("=" * 70)

    config = TransformConfig(learnable=['translation', 'scale'])

    # Baseline
    baseline = LearnableTransform(config).cuda()
    baseline_ms = benchmark(lambda: baseline(means, scales, quats))
    baseline_throughput = (n / baseline_ms) * 1000 / 1e6
    logger.info(f"Baseline:           {baseline_ms:.3f} ms ({baseline_throughput:.1f}M/sec)")

    # Opt1: Rotation caching
    opt1 = LearnableTransformOpt1(config).cuda()
    opt1_ms = benchmark(lambda: opt1(means, scales, quats))
    opt1_throughput = (n / opt1_ms) * 1000 / 1e6
    speedup = baseline_ms / opt1_ms
    status = "[OK]" if speedup > 1.05 else "[--]"
    logger.info(f"Opt1 (rot cache):   {opt1_ms:.3f} ms ({opt1_throughput:.1f}M/sec) {speedup:.2f}x {status}")

    # Opt2: torch.compile
    try:
        opt2 = LearnableTransformOpt2(config).cuda()
        # Extra warmup for compilation
        for _ in range(50):
            opt2(means, scales, quats)
        torch.cuda.synchronize()
        opt2_ms = benchmark(lambda: opt2(means, scales, quats))
        opt2_throughput = (n / opt2_ms) * 1000 / 1e6
        speedup = baseline_ms / opt2_ms
        status = "[OK]" if speedup > 1.05 else "[--]"
        logger.info(f"Opt2 (compile):     {opt2_ms:.3f} ms ({opt2_throughput:.1f}M/sec) {speedup:.2f}x {status}")
    except Exception as e:
        logger.info(f"Opt2 (compile):     FAILED - {e}")

    logger.info("\n" + "=" * 70)
    logger.info("RESULTS: Optimizations marked [OK] show >5% improvement")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_benchmarks()
