# Changelog

All notable changes to gsmod will be documented in this file.

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

## [0.3.0] - 2024

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

## [0.2.0] - 2024

### Added
- Color pipeline with LUT optimization
- Transform pipeline with matrix fusion
- Numba-optimized kernels for quaternion operations
- Basic filtering capabilities

## [0.1.0] - 2024

### Added
- Initial release
- Basic color adjustments
- Geometric transformations
- NumPy backend
