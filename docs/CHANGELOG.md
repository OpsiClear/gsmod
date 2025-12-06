# Changelog

All notable changes to gsmod will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.6] - 2025-12-06

### Changed
- **Enhanced SH-Aware Color Processing** (CPU and GPU)
  - Brightness, Contrast, Saturation, Temperature, Gamma, Hue Shift now apply to BOTH sh0 (DC) and shN (higher-order SH coefficients)
  - Improved SH mode handling: brightness/contrast use corrected formulas for rendered RGB behavior
  - Vibrance now correctly converts SH->RGB->SH (required for adaptive saturation)
  - Split toning (shadow/highlight tints) remain DC-only (additive operations)
  - Conditional clamping: only applied when is_sh0_rgb=True (SH coefficients can be negative or >1)
  - Automatic SH->RGB->SH conversion for operations requiring RGB format
- **Dependencies Updated**
  - gsply requirement updated from `0.2.11` to `>=0.2.13`
  - torch moved to optional dependency (install separately: `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128`)
- **Code Consolidation**
  - `_axis_angle_to_rotation_matrix_numpy()` moved to `shared/rotation.py` (single canonical implementation)
  - `filter/api.py` and `filter/apply.py` now import from shared module
- **Documentation Updated**
  - CLAUDE.md: Updated module references (`color/` module structure, distributed `*/kernels.py` files)
  - WORKFLOWS.md: Updated API references to current exports (GSDataPro, ColorValues, Filter, Pipeline)

### Fixed
- **CRITICAL: Format Tracking in GSTensorPro**
  - Fixed format auto-detection incorrectly changing format during initialization
  - `_format` now properly passed in all factory methods (from_gsdata, from_gstensor, clone, to)
  - Prevents LINEAR scales from being incorrectly detected as PLY log-scales
  - Affects: All GSTensorPro creation and conversion operations
- **Numerical Stability Improvements**
  - LUT kernels: clamp to [0,1] before conversion to int() to avoid NaN/Inf issues
  - Ellipsoid filter: guard against division by zero with minimum radius (1e-7)
  - Logit conversion: use log subtraction form for better numerical stability
  - Consistent epsilon values: standardized to 1e-7 across codebase
- **FilterValues.is_neutral() Bug Fix**
  - Now correctly checks `invert` parameter (invert=True is NOT neutral)
- **GPU Opacity/Scale Filtering**
  - Improved threshold conversion with proper clamping to avoid boundary issues
  - Optimized: compute max_scales once when needed for both min/max filters

### Performance
- GPU scale filtering: eliminated redundant max_scales computation (2x evaluation -> 1x)

## [0.1.5] - 2025-12-02

### Added
- **Spherical Harmonics (SH) Color Support** (`gsmod.color.sh_kernels`, `gsmod.color.sh_utils`)
  - Full SH-aware color operations matching GPU ground truth behavior
  - Brightness and Saturation: Apply to BOTH sh0 (DC) and shN (higher-order SH bands)
  - All other operations: Apply to sh0 ONLY (contrast, gamma, temperature, tint, hue, vibrance, fade)
  - Numba-optimized kernels: `apply_scale_to_sh_numba()`, `apply_matrix_to_sh_numba()`, `apply_contrast_to_sh_numba()`, etc.
  - SH utilities: `sh_to_rgb()`, `compute_luminance()`, matrix builders for saturation/temperature/hue
  - CPU/GPU consistency: CPU now matches GPU SH handling exactly
- **Triton GPU Kernels** (`gsmod.torch.triton_kernels`)
  - Fused GPU kernels for maximum performance on NVIDIA GPUs
  - Kernels: `triton_adjust_brightness()`, `triton_adjust_contrast()`, `triton_adjust_gamma()`, `triton_adjust_saturation()`, `triton_adjust_temperature()`
  - Graceful fallback to PyTorch operations if Triton unavailable
  - Block-based parallelization optimized for GPU architecture
- **GSTensorPro.from_gstensor() Factory Method**
  - Convert gsply GSTensor to GSTensorPro while preserving format state
  - Recommended workflow: load with gsply, convert formats, then wrap with GSTensorPro
  - Preserves `is_sh0_rgb`, `is_scales_ply`, `is_opacities_ply` format tracking
  - Optional device/dtype conversion during wrapping
- **Format-Aware CPU Filtering** (`gsmod.filter.apply`)
  - CPU filters now handle PLY format (logit opacities, log scales) correctly
  - Automatic threshold conversion: linear -> logit/log when filtering PLY format data
  - Matches GPU FilterGPU behavior for CPU/GPU consistency
  - Uses `is_opacities_ply` and `is_scales_ply` format properties from gsply 0.2.11
- **Transform Log-Space Kernel** (`gsmod.transform.kernels`)
  - Added `elementwise_add_scalar_numba()` for log-space scale transforms
  - Enables efficient scale operations in PLY format (log space) where multiplication becomes addition
- **New Benchmarks**
  - `benchmarks/benchmark_sh_color.py`: NumPy vs Numba performance for SH operations
  - `benchmarks/benchmark_triton.py`: PyTorch vs Triton kernel performance comparison
  - `benchmarks/benchmark_gpu_vs_cpu.py`: Comprehensive GPU vs CPU performance analysis

### Changed
- **Color Application Refactored** (`gsmod.color.apply`)
  - `apply_color_values()` now accepts `shN` parameter and returns tuple `(sh0, shN)`
  - Matches GPU ground truth: brightness/saturation on both sh0+shN, all else on sh0 only
  - Removed LUT-based implementation in favor of direct operations for better SH support
  - Breaking change: Return type changed from `np.ndarray` to `tuple[np.ndarray, np.ndarray | None]`
- **GPU Color Methods Enhanced** (`gsmod.torch.gstensor_pro`)
  - All color adjustment methods now use Triton kernels when available
  - Shadows/highlights now use smoothstep curves matching supersplat shader
  - Better visual quality with smooth shadow/highlight transitions
  - Temperature and tint operations preserve format awareness
- **Dependencies Updated**
  - gsply requirement updated from `0.2.10` to `0.2.11`
  - Supports latest format query properties and performance improvements

### Performance
- SH color operations: 10-30x faster with Numba kernels vs pure NumPy
- GPU color operations: Additional 10-20% speedup with Triton kernels (when available)
- CPU-GPU consistency: No performance penalty for matching behavior

### Fixed
- CPU color operations now correctly handle SH formats matching GPU behavior
- Shadow/highlight curves now use proper smoothstep interpolation (matches supersplat)
- Format-aware filtering eliminates opacity/scale threshold bugs in PLY format

### Documentation
- Updated CLAUDE.md with SH color operation semantics
- Documented brightness/saturation special case (both sh0 and shN)
- Added GPU ground truth reference in color module docstrings

## [0.1.4] - 2025-11-26

### Added
- **Auto-Correction Module** (`gsmod.color.auto`)
  - Industry-standard automatic color correction algorithms (Photoshop/Lightroom/iOS Photos style)
  - `auto_enhance()`: Combined enhancement (exposure + contrast + white balance), like iOS Photos Auto
  - `auto_contrast()`: Percentile-based histogram stretching (0.1% clipping), like Photoshop Auto Contrast
  - `auto_exposure()`: 18% gray midtone targeting (0.45 in gamma space)
  - `auto_white_balance()`: Gray World and White Patch methods
  - `compute_optimal_parameters()`: Minimal adjustments to reach target statistics
  - `AutoCorrectionResult`: Dataclass with computed adjustments, converts to ColorValues via `.to_color_values()`
  - Self-referential analysis (no target histogram required)
- **Perceptual Loss Functions** (`gsmod.histogram.loss`)
  - `PerceptualColorLoss`: Comprehensive loss addressing flat histogram problem
    - Contrast preservation (penalizes reduction below threshold)
    - Dynamic range matching (5th/95th percentiles)
    - Parameter regularization (keeps values near neutral)
  - `ContrastPreservationLoss`: Standalone contrast preservation loss
  - `ParameterBoundsLoss`: Soft penalty for extreme parameter values
  - `create_balanced_loss()`: Factory function for balanced defaults

### Changed
- **Filter Atomic Class Architecture** (`gsmod.filter.atomic.Filter`)
  - Rewritten to use `FilterValues` internally for fused kernel path
  - AND operations (`Filter & Filter`) now merge FilterValues for single kernel execution
  - OR/NOT operations correctly fall back to mask combination approach
  - Added internal helpers: `_from_values()` and `_from_mask_fn()`
  - Factory methods simplified to construct FilterValues directly
- **Pipeline Operation Merging** (`gsmod.pipeline.Pipeline`)
  - Added `_merge_operations()` method to merge consecutive same-type operations
  - Color, transform, and filter operations merged using their `+` operator
  - Reduces number of kernel calls for better performance

### Performance
- Filter AND operations: 2.8x faster via merged FilterValues (single fused kernel)
- Pipeline transform merge: 3.5x faster when consecutive transforms combined
- Pipeline full chain: 1.6x faster overall with operation merging
- Filter.get_mask(): 2.5x faster after removing logger.debug overhead

### Documentation
- Documented expected ~2% color merge difference due to LUT quantization
  - This is mathematically correct behavior (quantization artifacts)
  - Performance benefit outweighs minor precision difference

### Fixed
- Filter.to_values() now correctly supports AND combinations (returns merged FilterValues)
- OR and NOT combinations correctly raise ValueError (cannot be represented as single FilterValues)

## [0.1.3] - 2025-11-26

### Added
- **Unified Pipeline Class** (`gsmod.pipeline.Pipeline`)
  - CPU pipeline class matching GPU PipelineGPU interface
  - Fluent API for chaining color, transform, and filter operations
  - Method chaining: `.brightness()`, `.saturation()`, `.translate()`, `.scale()`, `.min_opacity()`, etc.
  - Operations accumulated and executed in order when called
  - Single unified interface for all processing operations
- **Filter Atomic Class** (`gsmod.filter.atomic.Filter`)
  - Immutable filter class with boolean operators (&, |, ~)
  - Factory methods: `min_opacity()`, `max_opacity()`, `min_scale()`, `max_scale()`, `sphere()`, `box()`, `ellipsoid()`, `frustum()`
  - Combine filters with logical operators for complex patterns
  - Direct mask computation via `.get_mask()` method
  - Apply filters via callable interface: `filter(data, inplace=False)`
- **Extended GSDataPro Filter Methods**
  - Individual filter methods: `filter_min_opacity()`, `filter_max_opacity()`, `filter_min_scale()`, `filter_max_scale()`
  - Geometry filters: `filter_within_sphere()`, `filter_outside_sphere()`, `filter_within_box()`, `filter_outside_box()`
  - Advanced geometry: `filter_within_ellipsoid()`, `filter_outside_ellipsoid()`, `filter_within_frustum()`, `filter_outside_frustum()`
  - Transform methods: `translate()`, `scale_uniform()`, `scale_nonuniform()`, `rotate_quaternion()`, `rotate_euler()`, `rotate_axis_angle()`, `transform_matrix()`
  - Color adjustment: `adjust_brightness()` and other individual color methods
- **GPU Filter Enhancements** (`gsmod.torch.filter.FilterGPU`)
  - Rotated box filtering: `within_rotated_box()`, `outside_rotated_box()`
  - Ellipsoid filtering: `within_ellipsoid()`, `outside_ellipsoid()`
  - Frustum filtering: `within_frustum()`, `outside_frustum()`
  - Optimized kernels: `_filter_rotated_box()`, `_filter_ellipsoid()`, `_filter_frustum()`
  - Axis-angle to rotation matrix conversion: `_axis_angle_to_rotation_matrix()`

### Changed
- **Filter Architecture Simplified**
  - Atomic Filter class now uses single fused kernel path
  - Removed redundant kernel implementations
  - Improved performance through kernel consolidation
- **Full CPU-GPU Parity**
  - All filter operations now available on both CPU and GPU
  - Consistent API between GSDataPro and GSTensorPro
  - Unified behavior across backends

### Fixed
- **Test Suite Stability**
  - All 498 tests passing (with 55 skipped GPU tests when CUDA unavailable)
  - Improved test coverage to 61% overall
  - Enhanced equivalence tests between CPU and GPU implementations

### Performance
- Filter atomic operations use optimized Numba kernels for 40-100x speedup
- Unified Pipeline reduces method call overhead
- GPU filters maintain 100-180x speedup over CPU for large datasets

## [0.1.2] - 2025-01-25

### Changed
- **Dependencies**
  - Updated gsply requirement from `>=0.2.8` to `==0.2.10` (exact version pin)
- **Style Harmonization with gsply**
  - All docstrings now use `:returns:` instead of `:return:` (consistent with gsply)
  - Enhanced module-level docstrings with performance notes and examples
  - `color/__init__.py`: Added performance metrics (1,091M colors/sec)
  - `transform/__init__.py`: Added performance metrics (698M Gaussians/sec)
  - `filter/__init__.py`: Added performance metrics (46M Gaussians/sec)
  - `torch/__init__.py`: Added GPU benchmark details (183x speedup, 1.09B Gaussians/sec)
- **Configuration Class Naming**
  - Renamed `GsproConfig` to `GsmodConfig` for consistency with project name
  - Updated all references in codebase (config module, AGENTS.md, benchmarks, examples)
  - Export name updated: import `GsmodConfig` from `gsmod.config`
- **Configuration Class Documentation**
  - Fixed misleading "will be deprecated" comment on ColorGradingConfig
  - Clarified that config classes (ColorGradingConfig, TransformConfig, etc.) are canonical and actively used
  - Reorganized torch module imports for better clarity

### Removed
- **Deprecated Backward Compatibility Aliases** (Breaking Change)
  - Removed `LearnableColorGrading` (use `LearnableColor` instead)
  - Removed `SoftFilter` (use `LearnableFilter` instead)
  - Removed `GSTensorProLearn` (use `LearnableGSTensor` instead)
  - Removed `SoftFilterConfig` (use `LearnableFilterConfig` instead)
  - **Impact**: Zero usage found in codebase, tests, benchmarks, or documentation
  - **Migration**: Update imports to use new standardized names with "Learnable" prefix
- **Property Aliases from ColorValues** (Breaking Change)
  - Removed `black_level`, `white_level` (use `brightness`, `contrast`)
  - Removed `lift`, `gain` (use `shadows`, `highlights`)
  - Removed `exposure` (use `brightness`)
  - Removed `midtones` (use `gamma`)
  - Removed `vibrancy` (use `vibrance`)
  - Removed `blacks`, `whites` (use `shadows`, `highlights`)
  - **Impact**: Zero usage found in codebase
  - **Rationale**: Simplifies API, aligns with gsply's design philosophy of no property aliases

### Fixed
- Removed confusing "legacy" comment from Color pipeline import
- Improved clarity of module organization and export structure

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
