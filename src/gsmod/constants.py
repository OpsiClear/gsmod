"""
Constants and default values for gsmod pipelines.

Centralizes magic numbers and configuration defaults for better maintainability.
"""

from __future__ import annotations

# =============================================================================
# Color Pipeline Constants
# =============================================================================

# LUT (Look-Up Table) configuration
DEFAULT_LUT_SIZE = 1024  # 0.1% color precision (1024 bins per channel)
MIN_LUT_SIZE = 256  # Minimum for acceptable quality
MAX_LUT_SIZE = 4096  # Maximum before diminishing returns

# Color adjustment defaults (neutral values)
DEFAULT_TEMPERATURE = 0.5  # Neutral color temperature
DEFAULT_BRIGHTNESS = 1.0  # No brightness change
DEFAULT_CONTRAST = 1.0  # No contrast change
DEFAULT_GAMMA = 1.0  # Linear gamma
DEFAULT_SATURATION = 1.0  # No saturation change
DEFAULT_SHADOWS = 1.0  # No shadow adjustment
DEFAULT_HIGHLIGHTS = 1.0  # No highlight adjustment

# Color adjustment ranges
TEMPERATURE_MIN = 0.0  # Cool/blue
TEMPERATURE_MAX = 1.0  # Warm/orange
BRIGHTNESS_MIN = 0.0  # Black
BRIGHTNESS_MAX = 5.0  # Very bright (practical limit)
CONTRAST_MIN = 0.0  # Flat
CONTRAST_MAX = 5.0  # Very high contrast
GAMMA_MIN = 0.1  # Very bright (inverted)
GAMMA_MAX = 5.0  # Very dark
SATURATION_MIN = 0.0  # Grayscale
SATURATION_MAX = 5.0  # Very saturated
SHADOWS_MIN = 0.0  # Black shadows
SHADOWS_MAX = 5.0  # Very bright shadows
HIGHLIGHTS_MIN = 0.0  # Dark highlights
HIGHLIGHTS_MAX = 5.0  # Very bright highlights

# =============================================================================
# Filter Pipeline Constants
# =============================================================================

# Filter defaults
DEFAULT_OPACITY_THRESHOLD = 0.0  # Keep all by default
DEFAULT_MAX_SCALE = 10.0  # Based on typical Gaussian splat scenes

# Filter ranges
OPACITY_MIN = 0.0  # Fully transparent
OPACITY_MAX = 1.0  # Fully opaque
SCALE_MIN = 0.0  # No scale filtering

# Recommended percentiles for max_scale calculation
SCALE_PERCENTILE_DEFAULT = 99.5  # Filter top 0.5% of scales
SCALE_PERCENTILE_MIN = 0.0
SCALE_PERCENTILE_MAX = 100.0

# =============================================================================
# Transform Pipeline Constants
# =============================================================================

# Default transform values (identity transforms)
DEFAULT_TRANSLATION = (0.0, 0.0, 0.0)  # No translation
DEFAULT_SCALE = 1.0  # No scaling

# =============================================================================
# General Constants
# =============================================================================

# Computation device
DEFAULT_DEVICE = "cpu"  # CPU-only (NumPy/Numba)

# Spatial dimensions
SPATIAL_DIMS = 3  # X, Y, Z
QUATERNION_DIMS = 4  # w, x, y, z

# Valid filter types
VALID_FILTER_TYPES = {"none", "sphere", "cuboid", "ellipsoid", "frustum"}

# Valid rotation formats
VALID_ROTATION_FORMATS = {"quaternion", "matrix", "axis_angle", "euler"}
