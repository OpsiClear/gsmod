"""PyTorch GPU acceleration for gsmod - Ultra-fast processing with CUDA.

Performance (RTX 3090 Ti, 1M Gaussians):
  - Peak speedup: 183x over CPU (sphere filter: 105ms -> 0.57ms)
  - Average speedup: 43x across all operations
  - Throughput: 1.09 billion Gaussians/sec
  - Color operations: 15-31x speedup
  - Transform operations: 86-93x speedup
  - Format-aware SH/RGB with lazy conversion optimization

Classes:
    GSTensorPro: GPU tensor wrapper with processing operations
    ColorGPU: GPU-accelerated color adjustments
    TransformGPU: GPU-accelerated 3D transforms
    FilterGPU: GPU-accelerated filtering
    PipelineGPU: Unified GPU pipeline

Example:
    >>> from gsmod.torch import GSTensorPro
    >>> from gsmod import ColorValues, FilterValues, TransformValues
    >>>
    >>> # Load data directly to GPU
    >>> gstensor = GSTensorPro.from_ply("scene.ply", device='cuda')
    >>>
    >>> # Same API as CPU
    >>> gstensor.filter(FilterValues(min_opacity=0.1, sphere_radius=5.0))
    >>> gstensor.transform(TransformValues.from_translation(1, 0, 0))
    >>> gstensor.color(ColorValues(brightness=1.2, saturation=1.3))
    >>>
    >>> # Save back to disk
    >>> gstensor.to_ply("output.ply")
"""

from gsmod.torch.color import ColorGPU
from gsmod.torch.filter import FilterGPU
from gsmod.torch.gstensor_pro import GSTensorPro

# Learnable modules for training pipelines
from gsmod.torch.learn import (
    # Configuration dataclasses for learnable modules
    ColorGradingConfig,
    # Learnable modules with gradient support
    LearnableColor,
    LearnableFilter,
    LearnableFilterConfig,
    LearnableGSTensor,
    LearnableOpacity,
    LearnableTransform,
    OpacityConfig,
    TransformConfig,
)
from gsmod.torch.pipeline import PipelineGPU
from gsmod.torch.transform import TransformGPU

__all__ = [
    # Inference (fast, in-place operations)
    "GSTensorPro",
    "ColorGPU",
    "TransformGPU",
    "FilterGPU",
    "PipelineGPU",
    # Training (gradient support)
    "LearnableColor",
    "LearnableTransform",
    "LearnableOpacity",
    "LearnableFilter",
    "LearnableGSTensor",
    # Configuration dataclasses
    "ColorGradingConfig",
    "TransformConfig",
    "OpacityConfig",
    "LearnableFilterConfig",
]
