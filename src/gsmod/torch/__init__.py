"""PyTorch GPU acceleration for gsmod.

This module provides GPU-accelerated processing for Gaussian Splatting data
using PyTorch tensors. All operations leverage GPU parallelism for massive
performance gains over CPU processing.

Classes:
    GSTensorPro: GPU tensor wrapper with processing operations
    ColorGPU: GPU-accelerated color adjustments
    TransformGPU: GPU-accelerated 3D transforms
    FilterGPU: GPU-accelerated filtering
    PipelineGPU: Unified GPU pipeline

Performance:
    - Color: 10-100x faster than CPU
    - Transform: 20-50x faster than CPU
    - Filter: 50-100x faster than CPU

Example:
    >>> from gsmod.torch import GSTensorPro, PipelineGPU
    >>> import gsply
    >>>
    >>> # Load data to GPU
    >>> data = gsply.plyread("scene.ply")
    >>> gstensor = GSTensorPro.from_gsdata(data, device='cuda')
    >>>
    >>> # Create GPU pipeline
    >>> pipeline = (
    ...     PipelineGPU()
    ...     .within_sphere(radius=1.0)
    ...     .translate([1, 0, 0])
    ...     .brightness(1.2)
    ...     .saturation(1.3)
    ... )
    >>>
    >>> # Apply on GPU (ultra-fast)
    >>> result = pipeline(gstensor, inplace=True)
"""

from gsmod.torch.color import ColorGPU
from gsmod.torch.filter import FilterGPU
from gsmod.torch.gstensor_pro import GSTensorPro

# Learnable modules for training pipelines
from gsmod.torch.learn import (
    # Configs (will be deprecated)
    ColorGradingConfig,
    GSTensorProLearn,
    # New names
    LearnableColor,
    # Backwards compatibility aliases
    LearnableColorGrading,
    LearnableFilter,
    LearnableFilterConfig,
    LearnableGSTensor,
    LearnableTransform,
    SoftFilter,
    SoftFilterConfig,
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
    # Training (gradient support) - new names
    "LearnableColor",
    "LearnableTransform",
    "LearnableFilter",
    "LearnableGSTensor",
    # Configs
    "ColorGradingConfig",
    "TransformConfig",
    "LearnableFilterConfig",
    # Backwards compatibility aliases (deprecated)
    "LearnableColorGrading",
    "SoftFilter",
    "GSTensorProLearn",
    "SoftFilterConfig",
]
