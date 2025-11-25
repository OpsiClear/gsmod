# Changelog

All notable changes to gsmod will be documented in this file.

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
