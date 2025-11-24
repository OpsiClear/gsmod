# AGENTS.md

This file provides comprehensive guidance for AI coding agents working with the gsmod codebase. It contains operational details, architecture patterns, testing requirements, and performance considerations specific to agent-driven development.

## Project Overview

**gsmod** (Gaussian Splatting Processing) is a pure Python library for ultra-fast processing of 3D Gaussian Splatting data with both CPU (Numba) and GPU (PyTorch) backends. The library provides three main operation domains:

**CPU Operations (NumPy + Numba):**
1. **Color Processing**: LUT-based color adjustments (1,389M colors/sec)
2. **3D Transforms**: Geometric operations with Numba-optimized kernels (698M Gaussians/sec)
3. **Spatial Filtering**: Volume and property-based filtering (62M Gaussians/sec)

**GPU Operations (PyTorch CUDA):**
1. **Color Processing**: Format-aware operations with lazy conversion (up to 31x faster)
2. **3D Transforms**: GPU-parallelized transforms (up to 93x faster)
3. **Spatial Filtering**: GPU-accelerated filtering (up to 183x faster, 1.09B Gaussians/sec)

**Key Technologies:**
- Python 3.10+ (uses modern type syntax: `list[int]`, `X | Y`)
- NumPy 1.24+ for array operations
- Numba 0.59+ for JIT compilation (required dependency)
- PyTorch 2.0+ for GPU operations (optional, for GPU acceleration)
- gsply for GSData/GSTensor I/O and data structures
- pytest for testing
- ruff for linting/formatting

**Architecture Pattern:**
- **Simplified API (Recommended)**: GSDataPro/GSTensorPro with config value objects
- Config value dataclasses: ColorValues, FilterValues, TransformValues
- Value composition with `__add__` (e.g., `WARM + ColorValues(brightness=1.2)`)
- Built-in presets (CINEMATIC, STRICT_FILTER, DOUBLE_SIZE, etc.)
- Legacy pipeline API still available for advanced use cases (mask computation)
- Zero-copy APIs for maximum performance (`inplace=True` default)

## Development Environment Setup

### Package Management

**CRITICAL**: Always use `uv run` to execute Python scripts, never use `python` directly.

```bash
# Correct
uv run benchmarks/benchmark_color.py
uv run pytest tests/ -v

# Incorrect - DO NOT USE
python benchmarks/benchmark_color.py
python -m pytest tests/
```

### Installation

```bash
# Install in development mode with all dependencies
pip install -e .[dev]

# Verify Numba installation (required)
python -c "import numba; print(f'numba version: {numba.__version__}')"
```

### Directory Structure

```
C:\Users\opsiclear\Projects\gsmod\
├── src\gsmod\                 # Main library code
│   ├── __init__.py            # Public API exports
│   ├── gsdata_pro.py          # GSDataPro class (simplified API)
│   ├── pipeline.py            # Legacy Pipeline class
│   ├── compose.py             # Scene composition utilities
│   ├── params.py              # Parameterized pipeline support
│   ├── protocols.py           # Protocol definitions
│   ├── validators.py          # Input validation utilities
│   ├── constants.py           # Global constants
│   ├── utils.py               # Shared utilities
│   ├── config\                # Config values and presets (NEW)
│   │   ├── __init__.py
│   │   ├── values.py          # ColorValues, FilterValues, TransformValues
│   │   ├── presets.py         # Built-in presets (WARM, CINEMATIC, etc.)
│   │   ├── operations.py      # OperationSpec definitions
│   │   ├── color.py           # ColorConfig
│   │   ├── filter.py          # FilterConfig
│   │   ├── transform.py       # TransformConfig
│   │   └── config.py          # GsproConfig singleton
│   ├── color\                 # Color processing pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py        # Color class with LUT compilation
│   │   ├── kernels.py         # Numba-optimized color kernels
│   │   └── presets.py         # ColorPreset class
│   ├── transform\             # 3D transformation pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py        # Transform class
│   │   ├── api.py             # Quaternion utilities
│   │   └── kernels.py         # Numba-optimized transform kernels
│   └── filter\                # Filtering pipeline
│       ├── __init__.py
│       ├── pipeline.py        # Filter class with atomic design
│       ├── api.py             # Core filter implementation
│       ├── kernels.py         # Numba-optimized filter kernels
│       ├── bounds.py          # Scene bounds calculation
│       └── config.py          # FilterConfig dataclass
│   └── torch\                 # GPU-accelerated operations (PyTorch)
│       ├── __init__.py
│       ├── gstensor_pro.py    # GSTensorPro class with format-aware ops
│       ├── color.py           # ColorGPU pipeline
│       ├── transform.py       # TransformGPU pipeline
│       ├── filter.py          # FilterGPU pipeline
│       └── pipeline.py        # PipelineGPU unified class
├── tests\                     # Comprehensive test suite
│   ├── test_gsdata_pro.py     # Simplified API tests (NEW)
│   ├── test_config_values.py  # Config value dataclass tests (NEW)
│   ├── test_color_pipeline.py
│   ├── test_transform_pipeline.py
│   ├── test_filter.py         # Atomic Filter API tests
│   ├── test_filter_mask.py    # Filter.get_mask() tests
│   ├── test_pipeline.py
│   ├── test_numba_kernels.py
│   ├── test_parameterized_pipelines.py
│   ├── test_compose.py
│   ├── test_gsdata_integration.py
│   ├── test_gpu.py            # GPU tensor and pipeline tests
│   ├── test_format_handling.py # Format-aware operation tests
│   └── test_equivalence.py    # CPU-GPU equivalence tests
├── benchmarks\                # Performance benchmarks
│   ├── run_all_benchmarks.py
│   ├── benchmark_color.py
│   ├── benchmark_transform.py
│   └── benchmark_filter.py
├── .github\workflows\         # CI/CD configuration
│   ├── test.yml               # Multi-platform testing
│   ├── lint.yml               # Code quality checks
│   ├── build.yml              # Package building
│   ├── publish.yml            # PyPI publishing
│   └── benchmark.yml          # Performance tracking
├── docs\                      # Documentation
│   └── GPU_API_REFERENCE.md   # GPU API reference
├── pyproject.toml             # Package configuration
├── CLAUDE.md                  # Human-oriented project guide
├── AGENTS.md                  # This file
└── README.md                  # User-facing documentation
```

## Testing Requirements

### Running Tests

**CRITICAL**: All tests MUST pass before committing code changes.

```bash
# Run all tests with coverage (REQUIRED before commits)
pytest tests/ -v --cov=gsmod --cov-report=term-missing

# Run specific test file
pytest tests/test_color_pipeline.py -v

# Run specific test function
pytest tests/test_color_pipeline.py::test_color_stacking -v

# Run tests matching pattern
pytest tests/ -v -k "filter"
```

### Test Coverage Requirements

- Minimum coverage: 80% (enforced in CI/CD)
- New features MUST include tests
- Tests MUST cover both success and failure cases
- Performance-critical paths MUST have benchmark tests

### Test Structure

Tests mirror the source structure:

```
src/gsmod/color/pipeline.py    -> tests/test_color_pipeline.py
src/gsmod/transform/pipeline.py -> tests/test_transform_pipeline.py
src/gsmod/filter/pipeline.py    -> tests/test_filter.py
src/gsmod/pipeline.py           -> tests/test_pipeline.py
src/gsmod/color/kernels.py      -> tests/test_numba_kernels.py
```

**Test Naming Convention:**
- Test files: `test_<module>.py`
- Test functions: `test_<feature>_<scenario>`
- Test classes: `Test<Feature>` (optional, use for grouping)

### Creating Test Data

```python
import numpy as np
from gsmod import GSDataPro

# Create synthetic GSDataPro for testing
def create_test_data(n: int = 100) -> GSDataPro:
    """Create synthetic Gaussian data for testing."""
    pro = GSDataPro.__new__(GSDataPro)
    pro.means = np.random.randn(n, 3).astype(np.float32)
    pro.scales = np.abs(np.random.randn(n, 3).astype(np.float32)) * 0.1
    pro.quats = np.random.randn(n, 4).astype(np.float32)
    pro.quats = pro.quats / np.linalg.norm(pro.quats, axis=1, keepdims=True)
    pro.opacities = np.random.rand(n).astype(np.float32)
    pro.sh0 = np.random.rand(n, 3).astype(np.float32)
    pro.shN = None
    return pro

# Usage example
def test_basic_color():
    """Test basic color operation."""
    from gsmod import ColorValues

    data = create_test_data(100)
    original_sh0 = data.sh0.copy()

    data.color(ColorValues(brightness=1.5))

    # Colors should be modified
    assert not np.allclose(data.sh0, original_sh0)
```

### Test Categories

1. **Functional Tests**: Verify correctness of operations
   - Input validation (valid ranges, types)
   - Output correctness (expected values, shapes)
   - Edge cases (empty data, extreme values)

2. **Integration Tests**: Verify component interactions
   - Pipeline composition
   - GSData integration
   - Multi-operation workflows

3. **Performance Tests**: Verify optimization behavior
   - Operation stacking optimization
   - Cache hit rates (parameterized templates)
   - Zero-copy behavior (inplace=True)

## Code Quality Standards

### Linting and Formatting

**CRITICAL**: Code MUST pass ruff checks before committing.

```bash
# Check formatting (DO NOT COMMIT if this fails)
ruff format --check src/ tests/

# Auto-fix formatting
ruff format src/ tests/

# Check linting (DO NOT COMMIT if this fails)
ruff check src/ tests/

# Auto-fix linting issues
ruff check --fix src/ tests/
```

### Ruff Configuration

From `pyproject.toml`:
- Line length: 100 characters
- Target: Python 3.10+
- Ignored rules:
  - E501: Line too long (handled by formatter)
  - N803/N806/N812: Allow uppercase variable names (e.g., `R` for rotation matrix)

### Type Hints

**REQUIRED**: All public functions MUST have type hints.

```python
# Correct - Python 3.12+ syntax
def process_colors(
    colors: np.ndarray,
    brightness: float = 1.0,
    contrast: float = 1.0
) -> np.ndarray:
    """Process colors with brightness and contrast."""
    ...

# Correct - Union syntax
def handle_input(data: GSData | np.ndarray) -> GSData:
    """Handle GSData or raw array input."""
    ...

# Incorrect - Old syntax (DO NOT USE)
from typing import List, Union
def old_style(items: List[int]) -> Union[str, None]:  # BAD
    ...
```

### Logging vs Printing

**CRITICAL**: Use logging module, NOT print() statements.

```python
import logging

logger = logging.getLogger(__name__)

# Correct
logger.debug("Compiling LUT with size=%d", lut_size)
logger.info("Pipeline compiled with %d operations", len(operations))
logger.warning("Non-contiguous array detected, performance may degrade")
logger.error("Invalid opacity threshold: %f (must be in [0, 1])", threshold)

# Incorrect - DO NOT USE (except in benchmark scripts)
print("Compiling LUT...")  # BAD
print(f"Pipeline has {len(ops)} operations")  # BAD
```

**Exception**: Benchmark scripts in `benchmarks/` may use print() for output readability.

### Code Style

```python
# Use double quotes for strings
name = "gsmod"

# Use trailing commas in multi-line collections
operations = [
    ("temperature", 0.6),
    ("brightness", 1.2),
    ("contrast", 1.1),  # <- trailing comma
]

# Use descriptive variable names (not single letters except in math contexts)
lut_size = 256  # Good
n = len(data)   # Good for loop counter
x = data        # Bad (unless in mathematical context)

# Constants in UPPER_CASE
DEFAULT_LUT_SIZE = 1024  # 0.1% color precision (1024 bins per channel)
MAX_OPACITY = 1.0
```

## Architecture Deep Dive

### LUT-Based Color Pipeline Architecture

The color pipeline uses a two-phase architecture for maximum performance:

**Phase 1: LUT-Capable Operations** (Pre-compiled into 1D lookup tables)
- `temperature`: Color temperature adjustment (-1 to 1, additive composition)
- `tint`: Green/magenta balance (-1 to 1, additive composition)
- `brightness`: Multiplicative brightness scaling (multiplicative composition)
- `contrast`: Contrast expansion/contraction (multiplicative composition)
- `gamma`: Non-linear gamma correction (multiplicative exponent composition)
- `fade`: Black point lift for film look (0 to 1, additive composition)

**Phase 2: Dependent Operations** (Applied after LUT lookup)
- `saturation`: Color saturation adjustment (depends on luminance)
- `vibrance`: Selective saturation boost (depends on current saturation)
- `hue_shift`: Rotate colors around color wheel (RGB rotation matrix)
- `shadows`: Shadow region adjustment (depends on luminance threshold)
- `highlights`: Highlight region adjustment (depends on luminance threshold)
- `shadow_tint`: Split toning for shadows (hue + saturation, depends on luminance)
- `highlight_tint`: Split toning for highlights (hue + saturation, depends on luminance)

**Key Optimization: Interleaved LUT Layout**

```python
# Traditional layout (cache-unfriendly)
r_lut = [r0, r1, r2, ..., r1023]  # 1024 floats
g_lut = [g0, g1, g2, ..., g1023]  # 1024 floats
b_lut = [b0, b1, b2, ..., b1023]  # 1024 floats

# Interleaved layout (cache-friendly, 1.73x faster)
# Stored as 2D array [lut_size, 3] where RGB values for index i are adjacent
interleaved_lut = np.stack([r_lut, g_lut, b_lut], axis=1)  # [1024, 3]
# Access: lut[i, 0] = R, lut[i, 1] = G, lut[i, 2] = B
# All 3 channel values for index i are adjacent in memory (row-major order)
```

**Performance Impact:**
- Interleaved layout: 1.73x speedup from improved cache locality
- Single fused kernel: Eliminates intermediate array allocations
- Zero-copy API (`inplace=True`): 6.6x faster than standard API

### Operation Stacking and Optimization

The pipeline automatically optimizes repeated operations:

```python
# User code
pipeline = (
    Color()
    .brightness(1.2)     # 20% increase
    .brightness(1.5)     # 50% increase
    .gamma(1.05)         # First gamma
    .gamma(1.05)         # Second gamma
    .saturation(1.3)     # 30% increase
    .saturation(1.2)     # 20% increase
)

# Optimized to (happens during compile())
{
    "brightness": 1.2 * 1.5 = 1.8,      # Multiplicative composition
    "gamma": 1.05 * 1.05 = 1.1025,      # Multiplicative composition
    "saturation": 1.3 * 1.2 = 1.56,     # Multiplicative composition
}
```

**Composition Rules:**
- Additive: `temperature(0.3).temperature(0.2)` -> `temperature(0.5)`
- Multiplicative: `brightness(1.2).brightness(1.5)` -> `brightness(1.8)`
- Exponent: `gamma(1.05).gamma(1.05)` -> `gamma(1.1025)`

### Lazy Compilation with Dirty Flag

```python
class Color:
    def __init__(self):
        self._is_dirty = True  # Track if recompilation needed
        self._phase1_operations = []  # List of (op_type, params) tuples
        self._saturation_operations = []  # Phase 2: list of values
        self._vibrance_operations = []
        self._hue_shift_operations = []
        self._shadows_operations = []
        self._highlights_operations = []
        self._compiled_lut = None

    def brightness(self, value: float) -> Self:
        # Mark dirty - recompilation needed
        self._is_dirty = True
        # Append operation (will be optimized on compile())
        self._phase1_operations.append(("brightness", {"value": value}))
        return self

    def compile(self) -> Self:
        if not self._is_dirty and self._compiled_lut is not None:
            return self  # Skip if already compiled

        # Optimize stacked operations
        optimized = self._optimize_phase1_operations()

        # Build interleaved LUT [lut_size, 3]
        self._compiled_lut = self._build_lut(optimized)

        self._is_dirty = False
        return self

    def __call__(self, data: GSData, inplace: bool = True) -> GSData:
        # Auto-compile if dirty
        if self._is_dirty:
            self.compile()

        # Apply compiled LUT
        return self.apply(data, inplace)
```

### Parameterized Templates with Caching

For workflows requiring runtime parameter variation (animation, A/B testing):

```python
# Create parameterized template
template = Color.template(
    brightness=Param("b", default=1.2, range=(0.5, 2.0)),
    contrast=Param("c", default=1.1, range=(0.5, 2.0)),
    saturation=Param("s", default=1.2, range=(0.5, 2.0))
)

# Cache key: (b=1.5, c=1.2, s=1.4) -> compiled LUT
# Cache hit rate: ~67% for animation (repeated parameter values)
result = template(data, params={"b": 1.5, "c": 1.2, "s": 1.4})

# Cache implementation (internal)
# Color pipeline uses _lut_cache, Filter pipeline uses _filter_cache
cache_key = tuple(params[name] for name in self._param_order)  # Pre-sorted order (O(1))
if cache_key in self._lut_cache:
    self._compiled_lut = self._lut_cache[cache_key]  # Cache hit (49x faster)
else:
    # Temporarily update operations with params, compile, then cache
    lut = compile_lut(params)
    self._lut_cache[cache_key] = lut.copy()
    return lut
```

### Numba Kernel Design

**Critical Performance Considerations:**

1. **Use `nogil=True`** for true parallelism (releases GIL)
2. **Use `parallel=True`** with `prange` for multi-core utilization
3. **Use `fastmath=True`** for aggressive floating-point optimizations
4. **Use `cache=True`** to cache compiled machine code

```python
from numba import njit, prange

@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def fused_color_pipeline_interleaved_lut_numba(
    colors: np.ndarray,      # [N, 3] float32
    lut: np.ndarray,          # [lut_size, 3] float32 (2D interleaved array)
    saturation: float,
    vibrance: float,
    hue_shift_deg: float,
    shadows: float,
    highlights: float,
    out: np.ndarray,          # [N, 3] float32 (pre-allocated)
    m00: float, m01: float, m02: float,  # Pre-computed hue rotation matrix
    m10: float, m11: float, m12: float,
    m20: float, m21: float, m22: float,
) -> None:
    """Fused color pipeline: LUT lookup + Phase 2 operations."""
    N = colors.shape[0]
    lut_size = lut.shape[0]
    lut_max = lut_size - 1

    for i in prange(N):  # Parallel loop
        # LUT lookup (Phase 1) - interleaved access
        r_in = colors[i, 0]
        g_in = colors[i, 1]
        b_in = colors[i, 2]

        # Clamp to [0, 1] and convert to LUT index
        r_idx = min(max(int(r_in * lut_max), 0), lut_max)
        g_idx = min(max(int(g_in * lut_max), 0), lut_max)
        b_idx = min(max(int(b_in * lut_max), 0), lut_max)

        # Interleaved lookup (2D array access - RGB values adjacent in memory)
        r = lut[r_idx, 0]
        g = lut[g_idx, 1]
        b = lut[b_idx, 2]

        # Phase 2: Saturation, Vibrance, Hue Shift, Shadows, Highlights
        # (Full implementation includes all Phase 2 operations)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        if saturation != 1.0:
            r = lum + saturation * (r - lum)
            g = lum + saturation * (g - lum)
            b = lum + saturation * (b - lum)
            r = min(max(r, 0.0), 1.0)
            g = min(max(g, 0.0), 1.0)
            b = min(max(b, 0.0), 1.0)
            lum = 0.299 * r + 0.587 * g + 0.114 * b  # Recalc for shadows/highlights

        # Shadows/Highlights (conditional on luminance)
        if lum < 0.5:  # Shadow region
            if shadows != 1.0:
                factor = shadows - 1.0
                r = r + r * factor
                g = g + g * factor
                b = b + b * factor
        else:  # Highlight region
            if highlights != 1.0:
                factor = highlights - 1.0
                r = r + r * factor
                g = g + g * factor
                b = b + b * factor

        # Store result
        out[i, 0] = min(max(r, 0.0), 1.0)
        out[i, 1] = min(max(g, 0.0), 1.0)
        out[i, 2] = min(max(b, 0.0), 1.0)
```

### Transform Pipeline Architecture

3D transformations use fused 4x4 homogeneous transformation matrices:

```python
# Fused transform (FAST - single matrix multiply)
T = translate_matrix @ rotate_matrix @ scale_matrix
new_means = (T @ homogeneous_coords)[:, :3]
new_quaternions = quat_multiply(rotation_quat, old_quaternions)
new_scales = old_scales * scale_factor

# Individual transforms (SLOW - 3 separate operations)
means = translate(means, [1, 0, 0])
means = rotate(means, quaternion)
means = scale(means, 2.0)
```

**Performance:** Fused approach is 2-3x faster due to:
- Single matrix multiply vs 3 separate operations
- Better memory locality (process each Gaussian completely)
- Custom 3x3 matrix multiply (9x faster than BLAS for small matrices)

### Filter Pipeline Architecture

The Filter class uses an **atomic design** with factory methods and Python operators:

**Design Pattern:**
- Each filter is an atomic operation created via factory methods
- Filters are combined using Python operators: `&` (AND), `|` (OR), `~` (NOT)
- Nested combinations are automatically flattened for efficiency

```python
from gsmod import Filter

# Create atomic filters with factory methods
opacity = Filter.min_opacity(0.3)
scale = Filter.max_scale(2.0)
sphere = Filter.sphere(radius=5.0, center=[0, 0, 0])
box = Filter.box(size=[4, 4, 4])

# Combine with operators
combined = opacity & scale & sphere  # AND logic
either = opacity | sphere            # OR logic
outside = ~sphere                    # Invert

# Get mask without filtering
mask = combined.get_mask(data)

# Apply filter directly
filtered = combined(data, inplace=False)
```

**Numba-Optimized Kernels:**

```python
# Fused opacity+scale filter (1.95x faster than separate filters)
@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def fused_opacity_scale_filter(
    opacities: np.ndarray,
    scales: np.ndarray,
    min_opacity: float,
    max_scale: float,
    mask: np.ndarray  # Output boolean mask
) -> None:
    N = len(opacities)
    for i in prange(N):
        opacity_pass = opacities[i, 0] >= min_opacity
        scale_pass = (scales[i, 0] <= max_scale and
                     scales[i, 1] <= max_scale and
                     scales[i, 2] <= max_scale)
        mask[i] = opacity_pass and scale_pass
```

**Volume Filters:**

```python
# Sphere filter with center and radius
sphere = Filter.sphere(center=[0, 0, 0], radius=5.0)

# Box filter with optional rotation
box = Filter.box(center=[0, 0, 0], size=[4, 4, 4])
rotated_box = Filter.box(size=[4, 4, 4], rotation=[0, 45, 0])  # Euler angles

# Ellipsoid filter with semi-axes
ellipsoid = Filter.ellipsoid(center=[0, 0, 0], radii=[2, 3, 4])

# Frustum filter for camera culling
frustum = Filter.frustum(apex=[0, 0, 0], direction=[0, 0, 1], fov=60, near=0.1, far=10)
```

## Common Development Patterns

### Adding a New Color Operation

1. **Determine operation phase:**
   - Phase 1 (LUT-capable): temperature, brightness, contrast, gamma
   - Phase 2 (dependent): saturation, vibrance, hue_shift, shadows, highlights

2. **Add operation to Color class** (`src/gsmod/color/pipeline.py`):

```python
def new_operation(self, value: float) -> Self:
    """
    Apply new color operation.

    Args:
        value: Operation parameter (describe range and default)

    Returns:
        Self for method chaining
    """
    # Validate input
    validate_range(value, 0.0, 2.0, "value")

    # Mark dirty (trigger recompilation)
    self._is_dirty = True

    # Append operation (will be optimized during compile())
    self._phase1_operations.append(("new_operation", {"value": value}))

    return self
```

3. **Update LUT compiler** (if Phase 1 operation):

```python
def _build_lut(self, operations: dict) -> np.ndarray:
    # ... existing code ...

    # Apply new operation
    if "new_operation" in operations:
        value = operations["new_operation"]
        lut_r = apply_new_operation(lut_r, value)
        lut_g = apply_new_operation(lut_g, value)
        lut_b = apply_new_operation(lut_b, value)
```

4. **Update Numba kernel** (if Phase 2 operation) in `src/gsmod/color/kernels.py`:

```python
@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def fused_color_phase2_numba(..., new_param: float, ...):
    # Add new operation logic
    if new_param != 1.0:
        # Apply operation
        r = transform(r, new_param)
        g = transform(g, new_param)
        b = transform(b, new_param)
```

5. **Add tests** in `tests/test_color_pipeline.py`:

```python
def test_new_operation():
    """Test new color operation."""
    data = create_test_data(100)
    pipeline = Color().new_operation(1.5)
    result = pipeline(data, inplace=False)

    # Verify correctness
    assert result.colors.shape == data.colors.shape
    # Add specific value checks

def test_new_operation_stacking():
    """Test stacking of new operation."""
    pipeline = Color().new_operation(1.2).new_operation(1.5)
    # Verify operations are stored as list
    assert len(pipeline._phase1_operations) == 2
    # Compile to verify optimization: 1.2 * 1.5 = 1.8
    pipeline.compile()
    optimized = pipeline._optimize_phase1_operations()
    assert optimized["new_operation"] == 1.8
```

6. **Add to public API** in `src/gsmod/__init__.py` (if needed)

7. **Run tests and benchmarks:**

```bash
pytest tests/test_color_pipeline.py -v
uv run benchmarks/benchmark_color.py
```

### Adding a New GPU Operation

1. **Add to GSTensorPro class** (`src/gsmod/torch/gstensor_pro.py`):

```python
def new_operation(self, param: float, inplace: bool = False) -> GSTensorPro:
    """Apply new GPU operation (format-aware).

    :param param: Operation parameter
    :param inplace: If True, modify in-place; if False, return new object
    :returns: GSTensorPro object
    """
    if inplace:
        # Initialize format if not present
        if not hasattr(self, '_format'):
            self._format = {}

        # Check if operation needs RGB format
        if self._format.get("sh0", "sh") != "rgb":
            self.to_rgb(inplace=True)  # Only if needed

        # Apply GPU operation
        self.sh0 = torch.some_operation(self.sh0, param)
        self._base = None  # Invalidate base
        return self

    result = self.clone()
    return result.new_operation(param, inplace=True)
```

2. **Add to GPU pipeline class** (`src/gsmod/torch/color.py` or similar):

```python
def new_operation(self, param: float = 1.0) -> ColorGPU:
    """Add new operation to pipeline."""
    self._operations.append(("new_operation", param))
    return self
```

3. **Update pipeline execution** in `__call__`:

```python
elif op_name == "new_operation":
    data.new_operation(op_value, inplace=True)
```

4. **Update format tracking** if the operation requires RGB:

```python
# Add to RGB_REQUIRED_OPS if operation needs RGB
RGB_REQUIRED_OPS = {"saturation", "temperature", "vibrance", "new_operation"}
```

5. **Add tests** in `tests/test_gpu.py` and `tests/test_format_handling.py`

6. **Run tests:**

```bash
pytest tests/test_gpu.py tests/test_format_handling.py -v
uv run benchmarks/benchmark_format_aware.py
```

### Adding a New Transform Function

1. **Add to Transform class** (`src/gsmod/transform/pipeline.py`):

```python
def new_transform(self, param: float) -> Self:
    """Apply new geometric transformation."""
    validate_positive(param, "param")

    # Add operation to queue
    self._operations.append(("new_transform", {"param": param}))

    return self
```

2. **Implement kernel** in `src/gsmod/transform/kernels.py`:

```python
@njit(parallel=True, fastmath=True, cache=True)
def apply_new_transform_numba(
    means: np.ndarray,
    param: float,
    out_means: np.ndarray
) -> None:
    N = means.shape[0]
    for i in prange(N):
        # Apply transformation
        out_means[i, 0] = transform_x(means[i, 0], param)
        out_means[i, 1] = transform_y(means[i, 1], param)
        out_means[i, 2] = transform_z(means[i, 2], param)
```

3. **Add tests** in `tests/test_transform_pipeline.py`

4. **Update benchmarks** in `benchmarks/benchmark_transform.py`

### Adding a New Filter Operation

1. **Add factory method to Filter class** (`src/gsmod/filter/pipeline.py`):

```python
@classmethod
def new_filter(cls, threshold: float = 0.5) -> Self:
    """Create filter by new criterion.

    Args:
        threshold: Filter threshold value

    Returns:
        Filter instance
    """
    f = cls()
    f._type = "new_filter"
    f._params = {"threshold": float(threshold)}
    return f
```

2. **Add mask computation in `_compute_atomic_mask`**:

```python
def _compute_atomic_mask(self, data: GSData) -> np.ndarray:
    # ... existing cases ...

    elif self._type == "new_filter":
        threshold = self._params["threshold"]
        return apply_new_filter_numba(data.attribute, threshold)
```

3. **Implement kernel** in `src/gsmod/filter/kernels.py`:

```python
@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def apply_new_filter_numba(
    data: np.ndarray,
    threshold: float,
) -> np.ndarray:
    N = len(data)
    mask = np.empty(N, dtype=np.bool_)
    for i in prange(N):
        mask[i] = data[i] >= threshold
    return mask
```

4. **Add tests** in `tests/test_filter.py` and `tests/test_filter_mask.py`:

```python
def test_new_filter():
    """Test new filter operation."""
    data = create_test_data(100)
    f = Filter.new_filter(0.5)
    mask = f.get_mask(data)

    # Verify correctness
    assert mask.dtype == np.bool_
    assert mask.sum() < len(data)

def test_new_filter_combination():
    """Test combining new filter with others."""
    f = Filter.new_filter(0.5) & Filter.min_opacity(0.3)
    mask = f.get_mask(data)
    # Combined mask should be subset
```

## Performance Optimization Guidelines

### Benchmarking New Features

**CRITICAL**: Always benchmark performance-critical code changes.

```bash
# Run full benchmark suite
cd benchmarks
uv run run_all_benchmarks.py

# Run specific benchmark
uv run benchmark_color.py

# Compare before/after performance
git stash  # Save changes
uv run benchmark_color.py > baseline.txt
git stash pop
uv run benchmark_color.py > optimized.txt
diff baseline.txt optimized.txt
```

### Performance Targets

Based on current benchmarks (must maintain or improve):

**Color Processing (100K colors):**
- Zero-copy API: >= 1,000 M colors/sec (0.1ms)
- Standard API: >= 200 M colors/sec (0.5ms)

**3D Transforms (1M Gaussians):**
- Combined transform: >= 650 M Gaussians/sec (1.5ms)

**Filtering (1M Gaussians):**
- Individual filters: >= 300 M Gaussians/sec (3.3ms)
- Full pipeline: >= 50 M Gaussians/sec (20ms)

### Common Performance Pitfalls

**1. Non-Contiguous Arrays (45% slowdown)**

```python
# BAD - Forces Numba to use slow fallback
non_contiguous = data.colors[:, [0, 2, 1]]  # Column reordering
result = apply_kernel(non_contiguous)  # 45% slower!

# GOOD - Ensure contiguity before processing
if not colors.flags['C_CONTIGUOUS']:
    colors = np.ascontiguousarray(colors)
result = apply_kernel(colors)  # Full speed
```

**2. Unnecessary Array Copies**

```python
# BAD - Creates unnecessary copies
result = pipeline(data, inplace=False)  # Allocates new GSData
result = pipeline(result, inplace=False)  # Another allocation

# GOOD - Use inplace=True for zero-copy performance
pipeline(data, inplace=True)  # Modifies data in-place
```

**3. Repeated Compilation (cache misses)**

```python
# BAD - Recompiles LUT every call
for frame in range(100):
    brightness = 1.0 + 0.2 * random.random()
    Color().brightness(brightness)(data, inplace=True)  # Slow!

# GOOD - Use parameterized template with caching
template = Color.template(brightness=Param("b", default=1.2, range=(0.8, 1.4)))
for frame in range(100):
    brightness = 1.0 + 0.2 * random.random()
    template(data, params={"b": brightness}, inplace=True)  # 49x faster on cache hits
```

**4. Missing Numba Optimizations**

```python
# BAD - Missing performance flags
@njit()
def slow_kernel(data):
    ...

# GOOD - All performance flags enabled
@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def fast_kernel(data):
    ...
```

### Memory Management

**Prefer inplace=True (default) for production workflows:**

```python
from gsmod import GSDataPro, ColorValues, FilterValues, TransformValues

# Memory-efficient (zero-copy) - default behavior
data = GSDataPro.from_ply("scene.ply")
data.filter(FilterValues(sphere_radius=0.8))   # inplace=True default
data.transform(TransformValues.from_rotation_euler(0, 45, 0))
data.color(ColorValues(brightness=1.2))
data.to_ply("output.ply")

# Memory-inefficient (creates copies)
data = GSDataPro.from_ply("scene.ply")
data = data.filter(FilterValues(sphere_radius=0.8), inplace=False)  # Copy 1
data = data.transform(TransformValues.from_scale(2.0), inplace=False)  # Copy 2
data = data.color(ColorValues(brightness=1.2), inplace=False)  # Copy 3
data.to_ply("output.ply")
```

## gsply Integration

gsmod uses gsply for GSData I/O and data structures. Key integration points:

### GSData Structure

```python
from gsply import GSData

# GSData attributes
data.means       # [N, 3] float32 - Gaussian centers
data.scales      # [N, 3] float32 - Gaussian sizes
data.quaternions # [N, 4] float32 - Gaussian orientations (w, x, y, z)
data.colors      # [N, 3] float32 - RGB colors [0, 1]
data.opacities   # [N, 1] float32 - Opacity values [0, 1]
```

### Contiguity Considerations

**IMPORTANT**: PLY files loaded via `gsply.plyread()` have non-contiguous arrays (zero-copy I/O optimization).

```python
# Check if data is contiguous
if not data.is_contiguous():
    logger.warning("Non-contiguous arrays detected")

# Manual conversion (ONLY for 100+ repeated operations)
data.make_contiguous(inplace=True)  # Use sparingly!
```

**When to use `make_contiguous()`:**
- ONLY for iterative workflows with 100+ operations on same data
- NOT for single-pass pipelines (conversion overhead > operation savings)

**Benchmark data (100K Gaussians):**
- Conversion cost: 3.2ms
- Color operation: 0.1ms
- Transform operation: 3.2ms
- **Verdict**: Conversion overhead exceeds savings for typical pipelines

### GSData Operations

```python
# Concatenation (6.15x faster than np.concatenate)
merged = GSData.concatenate([data1, data2, data3])

# Slicing with boolean mask
filtered = data[mask]  # or data.copy_slice(mask)

# Adding data
data.add(other_data)  # 1.9x faster than manual np.concatenate
```

### GSData Pre-Activation Stage

Some training/export pipelines store Gaussians in log-domain form (log scales, logit opacities, non-unit quaternions). Before running GPU splatting or color/transform/filter stages, normalize once on CPU via gsply's pre-activation helper:

```python
import gsply

data = gsply.plyread("scene_logits.ply")

# Single pass: scales = exp+clip, opacities = sigmoid, quats = normalized
gsply.apply_pre_activations(
    data,
    min_scale=1e-4,
    max_scale=100.0,
    min_quat_norm=1e-8,
    inplace=True,
)
```

Implementation details:
- Numba kernel `processes ~1M Gaussians in ~1.3 ms (≈750M/s).
- Helper auto-coerces arrays to float32 + contiguous layout, which is critical because `gsply.plyread()` returns zero-copy strided buffers.
- Parameter validation prevents degenerate ranges (e.g., `max_scale < min_scale`).
- Call this once per dataset; the pipeline can stay in-place to keep zero-copy behavior.

## CI/CD Pipeline

GitHub Actions workflows in `.github/workflows/`:

### test.yml (Multi-Platform Testing)

**Trigger:** Push/PR to master, main, develop branches

**Matrix:**
- OS: ubuntu-latest, windows-latest, macos-latest
- Python: 3.10, 3.11, 3.12, 3.13

**Steps:**
1. Checkout code
2. Setup Python
3. Install dependencies: `pip install -e .[dev]`
4. Verify Numba: `python -c "import numba; print(numba.__version__)"`
5. Run tests: `pytest tests/ -v --cov=gsmod --cov-report=xml`
6. Upload coverage to Codecov (Ubuntu + Python 3.12 only)

**Required Checks:**
- All tests MUST pass
- Coverage MUST be >= 80%

### lint (Code Quality)

**Trigger:** Push/PR to master, main, develop branches

**Steps:**
1. Lint with ruff: `ruff check src/ tests/`
2. Format check: `ruff format --check src/ tests/`
3. Type check: `mypy src/ --ignore-missing-imports` (non-blocking)

**Required Checks:**
- Ruff lint MUST pass
- Ruff format MUST pass

### build.yml (Package Building)

**Trigger:** Push/PR to master, main

**Steps:**
1. Build wheel: `python -m build`
2. Validate: `twine check dist/*`
3. Test installation: `pip install dist/*.whl`

### publish.yml (PyPI Publishing)

**Trigger:** GitHub Release created

**Steps:**
1. Build package
2. Publish to PyPI (with PYPI_API_TOKEN secret)

### benchmark.yml (Performance Tracking)

**Trigger:** Push to master, manual dispatch

**Steps:**
1. Run benchmark suite: `uv run benchmarks/run_all_benchmarks.py`
2. Archive results as artifacts

## Common Workflows

### Making Code Changes

1. **Create feature branch:**
   ```bash
   git checkout -b feature/new-operation
   ```

2. **Make changes with tests:**
   - Add feature code
   - Add tests
   - Update documentation if needed

3. **Verify locally:**
   ```bash
   # Format code
   ruff format src/ tests/

   # Check linting
   ruff check --fix src/ tests/

   # Run tests
   pytest tests/ -v --cov=gsmod --cov-report=term-missing

   # Run benchmarks if performance-critical
   cd benchmarks
   uv run benchmark_color.py  # or relevant benchmark
   ```

4. **Commit changes:**
   ```bash
   git add .
   git commit -m "feat: add new color operation"
   ```

5. **Push and create PR:**
   ```bash
   git push origin feature/new-operation
   # Create PR on GitHub
   ```

### Investigating Performance Issues

1. **Profile with benchmark:**
   ```bash
   cd benchmarks
   uv run benchmark_color.py > baseline.txt
   ```

2. **Check for common issues:**
   - Non-contiguous arrays
   - Missing Numba flags (parallel, fastmath, cache, nogil)
   - Unnecessary array copies
   - Cache misses in parameterized templates

3. **Compare optimizations:**
   ```bash
   # Make changes
   uv run benchmark_color.py > optimized.txt
   diff baseline.txt optimized.txt
   ```

4. **Validate with tests:**
   ```bash
   pytest tests/ -v
   ```

### Debugging Numba Kernels

```python
# Disable JIT for debugging (ONLY for development)
from numba import config
config.DISABLE_JIT = True

# Now kernel runs as pure Python (slow but debuggable)
result = fused_color_pipeline_numba(...)

# Can use print(), breakpoints, etc.
import pdb; pdb.set_trace()
```

**CRITICAL**: Never commit with `DISABLE_JIT = True`.

## Error Handling and Validation

### Input Validation Patterns

```python
from gsmod.validators import validate_range, validate_positive

def brightness(self, value: float) -> Self:
    """Apply brightness adjustment."""
    # Validate input (raises ValueError if invalid)
    validate_range(value, 0.0, 5.0, "brightness")

    # ... rest of implementation

def scale(self, factor: float | np.ndarray) -> Self:
    """Apply scaling transformation."""
    # Validate positive values
    if isinstance(factor, (int, float)):
        validate_positive(factor, "scale factor")
    else:
        if np.any(factor <= 0):
            raise ValueError("Scale factors must be positive")

    # ... rest of implementation
```

### Error Messages

Use descriptive error messages with context:

```python
# Good error messages
raise ValueError(
    f"Invalid opacity threshold: {threshold:.4f}. "
    f"Must be in range [0.0, 1.0]"
)

raise TypeError(
    f"Expected GSData, got {type(data).__name__}. "
    f"Use gsply.plyread() to load PLY files."
)

# Bad error messages
raise ValueError("Invalid value")  # Not descriptive
raise Exception("Error")  # Wrong exception type
```

## Security Considerations

### Safe File I/O

```python
# Always validate file paths
from pathlib import Path

def load_scene(path: str | Path) -> GSData:
    path = Path(path)

    # Validate extension
    if path.suffix != ".ply":
        raise ValueError(f"Expected .ply file, got {path.suffix}")

    # Validate existence
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return gsply.plyread(str(path))
```

### Avoid Arbitrary Code Execution

```python
# Never use eval() or exec()
# BAD
user_input = "brightness(1.2).contrast(1.5)"
eval(f"Color().{user_input}")  # DANGEROUS!

# GOOD - Use explicit parameter passing
params = {"brightness": 1.2, "contrast": 1.5}
pipeline = Color()
for op, value in params.items():
    if op == "brightness":
        pipeline = pipeline.brightness(value)
    elif op == "contrast":
        pipeline = pipeline.contrast(value)
```

## Troubleshooting Common Issues

### Tests Failing

```bash
# Check which tests are failing
pytest tests/ -v

# Run specific failing test with verbose output
pytest tests/test_color_pipeline.py::test_brightness -vv

# Check coverage gaps
pytest tests/ -v --cov=gsmod --cov-report=html
# Open htmlcov/index.html to see uncovered lines
```

### Linting Errors

```bash
# Auto-fix most issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/

# Check remaining issues
ruff check src/ tests/
```

### Performance Regression

```bash
# Compare benchmarks
cd benchmarks
git stash  # Save changes
uv run benchmark_color.py > baseline.txt
git stash pop
uv run benchmark_color.py > current.txt
diff baseline.txt current.txt
```

### Numba Compilation Errors

```python
# Common issues:
# 1. Type inference failure - add explicit types
@njit
def kernel(data: np.ndarray) -> np.ndarray:  # Good - explicit types
    ...

# 2. Unsupported operations - check Numba docs
# 3. Non-contiguous arrays - use np.ascontiguousarray()
```

## Additional Resources

### Documentation Files

- `README.md`: User-facing documentation and API reference
- `CLAUDE.md`: Human-oriented project guide
- `AGENTS.md`: This file (agent-specific instructions)
- `benchmarks/README.md`: Benchmark suite documentation

### Related Projects

- **gsply**: GSData I/O library (required dependency)
  - GitHub: https://github.com/OpsiClear/gsply
  - Version: 0.3.0+ (supports concatenation, contiguity management)

### External Documentation

- Numba: https://numba.pydata.org/
- NumPy: https://numpy.org/doc/stable/
- pytest: https://docs.pytest.org/
- ruff: https://docs.astral.sh/ruff/

## Quick Reference

### Essential Commands

```bash
# Testing
pytest tests/ -v --cov=gsmod --cov-report=term-missing

# Code Quality
ruff format src/ tests/
ruff check --fix src/ tests/

# Benchmarking
cd benchmarks && uv run run_all_benchmarks.py

# Build Package
python -m build
```

### Performance Targets (Must Maintain)

- Color: >= 1,000 M colors/sec (zero-copy)
- Transform: >= 650 M Gaussians/sec (1M batch)
- Filter: >= 50 M Gaussians/sec (full pipeline)

### File Templates

**Test file:**
```python
"""Tests for <module>."""

import numpy as np
import pytest
from gsply import GSData

from gsmod import <Class>


def create_test_data(n: int = 100) -> GSData:
    """Create synthetic test data."""
    means = np.random.randn(n, 3).astype(np.float32)
    scales = np.random.rand(n, 3).astype(np.float32) * 0.01
    quaternions = np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32)
    colors = np.random.rand(n, 3).astype(np.float32)
    opacities = np.random.rand(n, 1).astype(np.float32)
    return GSData(means=means, scales=scales, quaternions=quaternions,
                  colors=colors, opacities=opacities)


def test_basic_operation():
    """Test basic operation."""
    data = create_test_data(100)
    pipeline = <Class>().<operation>(1.2)
    result = pipeline(data, inplace=False)

    assert result is not None
    assert result.colors.shape == data.colors.shape


def test_operation_stacking():
    """Test operation stacking optimization."""
    pipeline = <Class>().<operation>(1.2).<operation>(1.5)
    # Verify optimization
    # assert ...


def test_invalid_input():
    """Test invalid input handling."""
    with pytest.raises(ValueError):
        <Class>().<operation>(-1.0)  # Invalid value
```

**Numba kernel:**
```python
"""Numba-optimized kernels for <module>."""

import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def kernel_name(
    input_data: np.ndarray,
    param: float,
    out: np.ndarray
) -> None:
    """
    Kernel description.

    Args:
        input_data: Input array [N, D]
        param: Operation parameter
        out: Output buffer [N, D] (pre-allocated)
    """
    N = input_data.shape[0]

    for i in prange(N):
        # Process each element
        out[i] = process(input_data[i], param)
```

---

**Document Version:** 2.0
**Last Updated:** 2025-11-21
**Compatibility:** gsmod v0.3.0+
**API Notes:** Uses simplified API (GSDataPro, GSTensorPro, config value objects)
