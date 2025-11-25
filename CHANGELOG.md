# Changelog

All notable changes to gsmod will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2025-01-24

### Added
- **Opacity Adjustment Module** (`gsmod.opacity`)
  - `OpacityValues` config dataclass for opacity scaling
  - Format-aware opacity adjustment supporting both linear [0,1] and PLY (logit) formats
  - Factory methods: `OpacityValues.fade()` and `OpacityValues.boost()`
  - Multiplicative composition: combine multiple opacity adjustments
- **Learnable Opacity Module** (`gsmod.torch.learn`)
  - `OpacityConfig` dataclass for learnable opacity configuration
  - `LearnableOpacity` nn.Module with full gradient support
  - Format-aware opacity scaling for both linear and PLY (logit) formats
  - Factory methods: `from_values()` and `to_values()` for OpacityValues integration
  - Integration with `LearnableGSTensor.apply_opacity()` for training pipelines
- **Unified Processing Interface** (`gsmod.processing`)
  - `GaussianProcessor` class for auto-dispatching between CPU and GPU backends
  - Single API works with both GSData/GSDataPro (CPU) and GSTensor/GSTensorPro (GPU)
  - Methods: `color()`, `transform()`, `filter()`, `opacity()`, and `process()` (batch processing)
  - Factory function `get_processor()` for singleton access
- **Shared Rotation Utilities** (`gsmod.shared.rotation`)
  - Unified rotation functions with auto-dispatch between NumPy and PyTorch
  - Functions: `quaternion_multiply()`, `quaternion_to_rotation_matrix()`, `euler_to_quaternion()`, etc.
  - Reduces code duplication across CPU and GPU backends
- **Enhanced Protocol Definitions** (`gsmod.protocols`)
  - Added `ColorProcessor`, `TransformProcessor`, `FilterProcessor` protocols
  - Generic type support in `PipelineStage` protocol for CPU/GPU compatibility
  - Added `is_neutral()` method to pipeline protocol
- **Opacity Support in GSDataPro and GSTensorPro**
  - New `.opacity(OpacityValues)` method on both GSDataPro and GSTensorPro
  - Handles PLY (logit) and linear opacity formats automatically
  - Supports fade (reduce opacity) and boost (increase opacity) operations
- **Filter Include/Exclude Modes**
  - New `invert` parameter in `FilterValues` (default: False)
  - `invert=False`: Include mode - keep only what matches (default behavior)
  - `invert=True`: Exclude mode - remove what matches, keep everything outside
  - Supported in CPU, GPU, and unified processor implementations
  - Chain multiple filters with different invert settings for complex patterns
  - Examples:
    - `FilterValues(sphere_radius=5.0, invert=True)` - keep points outside sphere
    - Hollow shell: include r=3.0, then exclude r=1.5
    - Box with hole: include box, then exclude sphere
- **Expanded Preset Library**
  - 39 color presets organized by category (up from 10)
  - Film Stock: KODAK_PORTRA, FUJI_VELVIA, KODAK_EKTACHROME, ILFORD_HP5, CINESTILL_800T
  - Seasonal: SPRING_FRESH, SUMMER_BRIGHT, AUTUMN_WARM, WINTER_COLD
  - Time of Day: SUNRISE, MIDDAY_SUN, SUNSET, BLUE_HOUR, OVERCAST
  - Artistic: HIGH_KEY, LOW_KEY, TEAL_ORANGE, BLEACH_BYPASS, CROSS_PROCESS, FADED_PRINT, SEPIA_TONE
  - Technical: LIFT_SHADOWS, COMPRESS_HIGHLIGHTS, INCREASE_CONTRAST, DESATURATE_MILD, ENHANCE_COLORS
  - 12 opacity presets: FADE_SUBTLE through FADE_HEAVY, BOOST_MILD through BOOST_STRONG, GHOST_EFFECT, TRANSLUCENT
  - New function: `get_opacity_preset(name)` for loading opacity presets by name
- **Histogram-Based Learning** (`gsmod.histogram`)
  - `HistogramResult.learn_from()`: Gradient-based color adjustment learning API
  - Automatically learns brightness, contrast, saturation, gamma to match target histogram
  - `HistogramResult.to_color_values()`: Rule-based adjustment suggestions (vibrant, dramatic, bright, dark, neutral)
  - GPU acceleration support - works with both CPU and CUDA tensors
  - Uses `MomentMatchingLoss` for fast and accurate histogram matching

### Changed
- **Rotation utilities refactored**
  - Moved rotation conversion functions from `transform/api.py` to `gsmod.shared.rotation`
  - `transform/api.py` now imports from shared module (canonical implementations)
  - Maintains backward compatibility (all existing imports still work)
- **Improved format property access**
  - Updated FilterGPU to use `is_opacities_ply` and `is_scales_ply` properties (from gsply 0.2.8+)
  - Replaced direct `_format` dict access with public API properties
- **Enhanced format tracking**
  - GSDataPro and GSTensorPro now use `copy_format_from()` when available
  - Better handling of masks, mask_names, and _base attributes in clone/copy operations
- **Fixed TransformValues.is_neutral()**
  - Now uses `np.allclose()` for robust float comparison
  - Prevents false negatives from floating-point precision errors

### Fixed
- **CRITICAL: GPU filter rotation matrix transpose bug** (`src/gsmod/torch/learn.py`)
  - Fixed missing transpose (.T) in `_get_ellipsoid_rotation_matrix()` and `_get_box_rotation_matrix()`
  - GPU rotated filters were oriented backwards compared to CPU (using R instead of R^T for inverse)
  - Now correctly transposes rotation matrices to match CPU implementation
  - Affects: `LearnableFilter` with ellipsoid/box geometry, gradient-based learning with rotated filters
- **CRITICAL: Axis-angle to rotation matrix conversion bug** (`src/gsmod/torch/learn.py`)
  - Fixed incorrect axis normalization in `_axis_angle_to_rotation_matrix()`
  - Was building skew-symmetric matrix from unnormalized axis_angle, then dividing by angle
  - Now correctly normalizes axis first, then builds K, matching Rodrigues' formula and CPU implementation
  - Rotation matrices now mathematically correct and match CPU exactly
  - Affects: All GPU rotated filter operations (ellipsoid, box, frustum)
- Format property access in GPU filters now uses public API (`is_opacities_ply`, `is_scales_ply`)
- TransformValues identity check now correctly handles array/tuple comparisons
- Improved robustness of neutral/identity detection across all value types

## [0.3.0] - 2025-01-20

### Added
- **GPU Acceleration** (PyTorch CUDA)
  - `GSTensorPro` class with format-aware SH/RGB operations
  - `ColorGPU` pipeline with lazy format conversion
  - `TransformGPU` pipeline for GPU-accelerated transforms
  - `FilterGPU` pipeline for GPU-accelerated filtering
  - `PipelineGPU` unified GPU pipeline
  - Up to 183x speedup over CPU, 1.09B Gaussians/sec on RTX 3090 Ti
- `docs/GPU_API_REFERENCE.md` - Comprehensive GPU API documentation

### Changed
- Project structure reorganized similar to gsply
  - `color.py` -> `color/` module (pipeline.py, presets.py, kernels.py)
  - `transforms.py` -> `transform/` module (api.py, pipeline.py, kernels.py)
  - `filter/` module expanded (api.py, pipeline.py, masks.py, bounds.py, config.py, kernels.py)
- Updated all documentation to reflect new module structure
- Improved error messages with more detailed context

### Removed
- **`apply_pre_activations`** - Use `gsply.apply_pre_activations()` instead
  - The function is now provided solely by gsply to avoid code duplication
  - Migration: `from gsmod import apply_pre_activations` -> `import gsply; gsply.apply_pre_activations(...)`
- Removed unused `_scale_numpy_fallback()` function from transform/api.py
- Removed debug/temporary test files from project root
- Cleaned up stale pycache files

### Fixed
- Updated README.md project structure to match actual file organization
- Fixed test file references in CLAUDE.md (test_color.py -> test_color_pipeline.py, etc.)
- Fixed module references in documentation (transforms.py -> transform/)

## [0.2.0] - 2025-01-15

### Added
- Color processing pipeline with LUT-based operations
- Transform pipeline with Numba-optimized kernels
- Filter pipeline with spatial and property-based filtering
- FilterMasks API for multi-layer mask management
- Parameterized templates for efficient parameter variation
- Scene composition utilities (concatenate, merge, split)
- ColorPreset class with built-in presets (cinematic, warm, cool, etc.)

### Performance
- Color: 1,389M colors/sec (zero-copy API)
- Transform: 698M Gaussians/sec
- Filter: 62M Gaussians/sec (full pipeline)

## [0.1.0] - 2025-01-01

### Added
- Initial release
- Basic color adjustments (brightness, contrast, gamma, saturation)
- Basic 3D transforms (translate, rotate, scale)
- gsply integration for GSData I/O
