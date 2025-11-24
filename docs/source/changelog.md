# Changelog

All notable changes to gsmod will be documented in this file.

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
