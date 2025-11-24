# Changelog

All notable changes to gsmod will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
