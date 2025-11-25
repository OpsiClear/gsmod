"""Type aliases for gsmod.

Provides unified type hints for array-like parameters across all modules.
"""

from collections.abc import Sequence

import numpy as np

# 3D vector type (position, size, center, etc.)
Vector3 = tuple[float, float, float] | Sequence[float] | np.ndarray

# Quaternion type (w, x, y, z)
Quaternion = tuple[float, float, float, float] | Sequence[float] | np.ndarray

# General array-like type
ArrayLike = Sequence[float] | np.ndarray
