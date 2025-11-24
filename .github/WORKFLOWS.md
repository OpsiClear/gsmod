# GitHub Workflows Documentation

## Overview

This repository uses GitHub Actions for CI/CD. The workflows are based on the gsply project patterns but adapted for gsmod's structure and requirements.

## Workflows

### 1. Test (`test.yml`)

**Trigger**: Push/PR to master, main, develop branches

**Purpose**: Run comprehensive tests across multiple platforms and Python versions

**Matrix**:
- OS: Ubuntu, Windows
- Python: 3.12, 3.13
- Total: 4 combinations

**Steps**:
1. Install dependencies (including Numba)
2. Verify Numba installation
3. Run pytest with coverage
4. Upload coverage to Codecov (Ubuntu + Python 3.12 only)

**Lint Job**:
- Runs ruff linting and formatting checks
- Runs mypy type checking (continue-on-error)

**Status**: [OK] Production ready

---

### 2. Build (`build.yml`)

**Trigger**: Push/PR to master/main, push tags (v*)

**Purpose**: Build distribution packages and verify installation

**Jobs**:

#### Build Job
1. Build source distribution (.tar.gz) and wheel (.whl)
2. Check package with twine
3. Upload artifacts

#### Verify Install Job
- Matrix: Ubuntu, Windows, macOS
- Downloads built artifacts
- Installs from wheel
- Tests imports:
  - `import gsmod`
  - Main APIs: ColorLUT, transform, Pipeline, ColorPreset
  - Filter APIs: filter_gaussians, apply_filter

**Status**: [OK] Production ready

---

### 3. Publish (`publish.yml`)

**Trigger**: GitHub Release published

**Purpose**: Publish package to PyPI and attach artifacts to release

**Jobs**:

#### Build Job
- Builds distribution packages

#### Publish to PyPI Job
- Uses trusted publishing (no API token needed)
- Requires PyPI environment configured in GitHub repo settings
- Publishes to https://pypi.org/p/gsmod

#### GitHub Release Job
- Signs artifacts with Sigstore
- Uploads signed artifacts to GitHub release

**Status**: [OK] Production ready

**Setup Required**:
1. Configure PyPI trusted publisher in PyPI settings
2. Create "pypi" environment in GitHub repo settings

**Note**: PyPI Test publishing is **NOT** included per user request

---

### 4. Benchmark (`benchmark.yml`)

**Trigger**: Push/PR to master/main, manual workflow_dispatch

**Purpose**: Run performance benchmarks and report results

**Benchmarks Run**:
1. `benchmark_color_optimization.py` - Color processing performance
2. `benchmark_learnable_cpu.py` - Learnable parameters CPU performance

**Features**:
- Uploads benchmark results as artifacts
- Comments results on PRs automatically
- Continue-on-error for each benchmark (won't fail entire workflow)

**Status**: [OK] Production ready

---

## Workflow Comparison: gsmod vs gsply

| Feature | gsply | gsmod |
|---------|-------|-------|
| Numba | Optional (with/without matrix) | Required (always installed) |
| Test data creation | Creates PLY test files | No test data needed |
| Python versions | 3.10-3.13 | 3.12-3.13 |
| Platforms | Ubuntu, Windows, macOS | Ubuntu, Windows, macOS |
| PyPI Test | Included | **Excluded** (per request) |
| Benchmark PR comments | Yes | Yes |

---

## Artifacts

### Test Workflow
- **coverage.xml** - Uploaded to Codecov

### Build Workflow
- **python-package-distributions** - Contains .tar.gz and .whl files

### Benchmark Workflow
- **benchmark-results** - Contains:
  - benchmark_results.txt
  - benchmark_optimizations_results.txt
  - benchmark_filter_results.txt

### Publish Workflow
- **Signed artifacts** - Attached to GitHub release

---

## Setup Instructions

### For Codecov (Optional)
1. Go to https://codecov.io
2. Connect your GitHub repository
3. No token needed (public repos)

### For PyPI Publishing (Required for releases)

#### 1. Configure Trusted Publisher on PyPI
1. Go to https://pypi.org/manage/account/publishing/
2. Add new publisher:
   - Project name: `gsmod`
   - Owner: `OpsiClear` (or your GitHub org)
   - Repository: `gsmod`
   - Workflow: `publish.yml`
   - Environment: `pypi`

#### 2. Configure GitHub Environment
1. Go to repository Settings > Environments
2. Create environment named `pypi`
3. Add protection rules if desired (require reviewers, etc.)

### For Manual Workflows
- Go to Actions tab
- Select "Benchmark" workflow
- Click "Run workflow"

---

## Local Testing

### Test Workflow
```bash
# Install dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/ -v --cov=gsmod --cov-report=xml --cov-report=term

# Lint
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

### Build Workflow
```bash
# Install build tools
uv pip install build twine

# Build
uv run python -m build

# Check
uv run twine check dist/*

# Test install
uv pip install dist/*.whl
uv run python -c "import gsmod; print(gsmod.__version__)"
```

### Benchmark Workflow
```bash
cd benchmarks

# Run benchmarks
uv run benchmark_color_optimization.py
uv run benchmark_learnable_cpu.py
```

---

## Troubleshooting

### Test failures on specific platforms
- Check if Numba has issues on that platform/Python version
- Review test output in Actions logs
- Run locally: `pytest tests/ -v`

### Build artifacts missing
- Ensure `python -m build` completes successfully
- Check twine check output

### PyPI publish fails
- Verify trusted publisher is configured correctly
- Check environment name matches (`pypi`)
- Ensure release is actually published (not draft)

### Benchmarks fail
- Continue-on-error is enabled, so workflow passes
- Check artifact uploads for partial results
- Run benchmarks locally to debug

---

## Maintenance

### Updating Python Versions
Edit matrix in `test.yml` and `build.yml`:
```yaml
matrix:
  python-version: ['3.12', '3.13', '3.14']  # Add new version
```

### Updating Action Versions
Check for updates periodically:
- `actions/checkout@v4` → v5
- `actions/setup-python@v5` → v6
- `codecov/codecov-action@v4` → v5
- etc.

### Adding New Benchmarks
Add to `benchmark.yml`:
```yaml
- name: Run new benchmark
  run: |
    cd benchmarks
    python benchmark_new.py 2>&1 | tee ../benchmark_new_results.txt
  continue-on-error: true
```

---

## Migration from Old CI

**Removed**: `.github/workflows/ci.yml`
**Replaced by**: `.github/workflows/test.yml`

**Key Improvements**:
1. More Python versions (added 3.13)
2. Explicit Numba verification
3. Updated action versions (v4 → v5)
4. Better coverage upload (only on one matrix combination)
5. Separated concerns (test, build, publish, benchmark)

---

## Status

All workflows are production-ready and follow best practices:

- [OK] test.yml - Comprehensive testing
- [OK] build.yml - Build verification
- [OK] publish.yml - PyPI publishing (no test.pypi)
- [OK] benchmark.yml - Performance tracking

**Created**: 2025-11-13
**Based on**: gsply workflows (adapted)
