# Changelog

All notable changes to gsmod will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.7] - 2025-12-09

### Changed
- **Style Consistency Improvements**
  - Unified `inplace` defaults: All methods now default to `inplace=True` for consistency
    - GSTensorPro methods (35 methods)
    - Filter.__call__(), FilterGPU.__call__(), FilterProcessor.apply() protocol
    - Exception: `compose_with_transforms()` remains `inplace=False` (composition utility)
  - Parameter naming unification: `factor` -> `scale` in scale methods, `value` -> `gamma` in adjust_gamma
  - Dict/list comprehensions used for cleaner code in GSTensorPro (apply_color_preset, compute_histogram)
- **TransformValues Non-Uniform Scale Support**
  - `TransformValues.scale` now stores 3-tuple `(sx, sy, sz)` for per-axis scaling
  - `from_scale()` accepts both uniform (float) and per-axis (tuple) scales
  - `from_matrix()` extracts per-axis scales from column norms
  - `is_neutral()` properly checks 3-tuple scale against `[1.0, 1.0, 1.0]`
- **Rotation/Scale Center Point Support**
  - New `center` parameter on TransformValues for rotation/scale center
  - Factory methods `from_rotation_euler()`, `from_rotation_axis_angle()`, `from_scale()` accept `center`
  - `to_matrix()` correctly applies T(center) @ SR @ T(-center) transformation
  - LearnableTransform correctly handles center in forward pass
- **Transform Pipeline Enhancements**
  - New `Transform.from_srt()` factory for standard Scale-Rotate-Translate order
  - New `rotate_euler_deg()` method for degree-based rotation
  - Improved docstrings with center point usage examples
- **Docstring Format Standardized**
  - All docstrings now use Sphinx reST format (`:param:`, `:returns:`) consistently
  - Removed Google-style Args/Returns in favor of Sphinx style
- **TransformConfig/LearnableTransform Updates**
  - `TransformConfig.scale` now supports float or 3-tuple
  - LearnableTransform handles center point for rotation/scale around arbitrary points
- **Numba Kernel Optimizations**
  - Filter kernels refactored with dict/list comprehensions
  - Color kernels use cleaner iteration patterns

### Fixed
- LearnableTransform now correctly applies SRT order with center support

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
  - torch moved to optional dependency
- **Code Consolidation**
  - `_axis_angle_to_rotation_matrix_numpy()` moved to `shared/rotation.py` (single canonical implementation)

### Fixed
- **CRITICAL: Format Tracking in GSTensorPro**
  - Fixed format auto-detection incorrectly changing format during initialization
  - `_format` now properly passed in all factory methods (from_gsdata, from_gstensor, clone, to)
- **Numerical Stability Improvements**
  - LUT kernels: clamp to [0,1] before conversion to int() to avoid NaN/Inf issues
  - Ellipsoid filter: guard against division by zero with minimum radius (1e-7)
  - Logit conversion: use log subtraction form for better numerical stability
- **FilterValues.is_neutral() Bug Fix**
  - Now correctly checks `invert` parameter (invert=True is NOT neutral)

## [0.1.5] - 2025-12-02

### Added
- **Spherical Harmonics (SH) Color Support** (`gsmod.color.sh_kernels`, `gsmod.color.sh_utils`)
  - Full SH-aware color operations matching GPU ground truth behavior
  - Numba-optimized kernels for SH processing
- **Triton GPU Kernels** (`gsmod.torch.triton_kernels`)
  - Fused GPU kernels for maximum performance on NVIDIA GPUs
  - Graceful fallback to PyTorch operations if Triton unavailable
- **GSTensorPro.from_gstensor() Factory Method**
  - Convert gsply GSTensor to GSTensorPro while preserving format state
- **Format-Aware CPU Filtering** (`gsmod.filter.apply`)
  - CPU filters now handle PLY format (logit opacities, log scales) correctly
- **Transform Log-Space Kernel** (`gsmod.transform.kernels`)
  - Added `elementwise_add_scalar_numba()` for log-space scale transforms

### Changed
- Color application refactored for better SH support
- GPU color methods enhanced with Triton kernels when available

### Performance
- SH color operations: 10-30x faster with Numba kernels vs pure NumPy
- GPU color operations: Additional 10-20% speedup with Triton kernels

## [0.1.4] - 2025-11-26

### Added
- **Auto-Correction Module** (`gsmod.color.auto`)
  - Industry-standard automatic color correction (Photoshop/Lightroom/iOS Photos style)
  - `auto_enhance()`: Combined enhancement like iOS Photos Auto
  - `auto_contrast()`: Percentile-based histogram stretching (Photoshop style)
  - `auto_exposure()`: 18% gray midtone targeting
  - `auto_white_balance()`: Gray World and White Patch methods
  - `compute_optimal_parameters()`: Minimal adjustments to reach targets
  - `AutoCorrectionResult`: Dataclass with `.to_color_values()` conversion
- **Perceptual Loss Functions** (`gsmod.histogram.loss`)
  - `PerceptualColorLoss`: Addresses flat histogram problem with contrast preservation
  - `ContrastPreservationLoss`: Standalone contrast preservation
  - `ParameterBoundsLoss`: Soft penalty for extreme values
  - `create_balanced_loss()`: Factory for balanced defaults

### Changed
- Filter atomic class rewritten to use FilterValues internally (2.8x faster AND operations)
- Pipeline operation merging for consecutive same-type operations (3.5x faster)

### Performance
- Filter AND: 2.8x faster via merged FilterValues
- Pipeline transforms: 3.5x faster with operation merging
- Filter.get_mask(): 2.5x faster after logger.debug optimization

## [0.1.3] - 2025-11-26

### Added
- **Unified Pipeline Class** (`gsmod.pipeline.Pipeline`)
  - CPU pipeline matching PipelineGPU interface
  - Fluent API for chaining color, transform, filter operations
- **Filter Atomic Class** (`gsmod.filter.atomic.Filter`)
  - Immutable filters with boolean operators (&, |, ~)
  - Factory methods for all filter types
- Full CPU-GPU parity for all filter operations

## [0.1.2] - 2025-01-25

### Changed
- Updated gsply requirement from `>=0.2.8` to `==0.2.10` (exact version pin)
- Style harmonization with gsply (`:returns:` docstrings, enhanced module documentation)
- Renamed `GsproConfig` to `GsmodConfig` for consistency with project name
- Enhanced module-level docstrings with performance metrics

### Removed
- **Breaking**: Removed deprecated backward compatibility aliases
  - `LearnableColorGrading` (use `LearnableColor`)
  - `SoftFilter` (use `LearnableFilter`)
  - `GSTensorProLearn` (use `LearnableGSTensor`)
  - `SoftFilterConfig` (use `LearnableFilterConfig`)
- **Breaking**: Removed property aliases from `ColorValues`
  - `black_level`, `white_level`, `lift`, `gain`, `exposure`, `midtones`, `vibrancy`, `blacks`, `whites`
  - Use canonical names: `brightness`, `contrast`, `shadows`, `highlights`, `gamma`, `vibrance`

## [0.1.1] - 2025-01-24

### Added
- Opacity adjustment module with format-aware opacity scaling
- OpacityValues config dataclass with fade() and boost() factory methods
- GaussianProcessor unified interface for auto-dispatching CPU/GPU operations
- Shared rotation utilities module for code reuse between backends
- Enhanced protocol definitions with ColorProcessor, TransformProcessor, FilterProcessor
- Opacity support in GSDataPro and GSTensorPro classes

### Changed
- Rotation utilities moved to shared module for better code organization
- Format property access now uses public API (is_opacities_ply, is_scales_ply)
- Improved format tracking in GSDataPro and GSTensorPro
- Fixed TransformValues.is_neutral() with robust float comparison

## [0.3.0] - 2025-01-20

### Added
- GSDataPro: Primary API with direct `.color()`, `.filter()`, `.transform()` methods
- GSTensorPro: GPU tensor wrapper with same API as GSDataPro
- Config values: `ColorValues`, `FilterValues`, `TransformValues` dataclasses
- Presets: Built-in color, filter, and transform presets
- Histogram computation with `HistogramResult`
- Scene composition utilities: `concatenate`, `compose_with_transforms`, `merge_scenes`
- Format verification with `FormatVerifier`
- GPU acceleration with up to 183x speedup

### Changed
- Unified parameter semantics between CPU and GPU pipelines
- Filter radius now uses absolute world units (not relative to scene bounds)
- Improved Numba JIT compilation with automatic warmup

### Performance
- Color: 1,091M Gaussians/sec
- Transform: 698M Gaussians/sec
- GPU: 1.09B Gaussians/sec on RTX 3090 Ti

## [0.2.0] - 2025-01-15

### Added
- Color pipeline with LUT optimization
- Transform pipeline with matrix fusion
- Numba-optimized kernels for quaternion operations
- Basic filtering capabilities

## [0.1.0] - 2025-01-01

### Added
- Initial release
- Basic color adjustments
- Geometric transformations
- NumPy backend
