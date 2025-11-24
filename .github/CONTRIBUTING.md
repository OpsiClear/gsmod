# Contributing to gsmod

Thank you for your interest in contributing to gsmod! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/gsmod.git
   cd gsmod
   ```
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests:
   ```bash
   pytest tests/ -v --cov=gsmod --cov-report=term-missing
   ```

4. Check code quality:
   ```bash
   ruff format src/ tests/
   ruff check --fix src/ tests/
   ```

5. Run benchmarks (if performance-related):
   ```bash
   cd benchmarks
   python run_all_benchmarks.py
   ```

6. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: description of changes"
   ```

7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

8. Create a Pull Request

## Code Style

- Use Python 3.10+ type hints (`list[int]`, `X | Y` syntax)
- Follow PEP 8 style guidelines (enforced by ruff)
- Use descriptive variable names
- Add docstrings to all public functions (`:params` style)
- Keep functions focused and concise
- Do not add comments for simple things

## Testing

- Write tests for all new features
- Ensure all existing tests pass
- Aim for >80% test coverage
- Test on multiple Python versions (3.10, 3.11, 3.12, 3.13)
- Tests MUST pass before committing

## Performance

- Benchmark performance-critical changes
- Maintain or improve performance targets:
  - Color: >= 1,000 M colors/sec (zero-copy)
  - Transform: >= 650 M Gaussians/sec (1M batch)
  - Filter: >= 50 M Gaussians/sec (full pipeline)
- Use Numba optimizations (parallel=True, fastmath=True, cache=True, nogil=True)
- Avoid non-contiguous arrays
- Use inplace=True for production workflows

## Documentation

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features
- Update AGENTS.md for architecture changes
- Update CLAUDE.md for high-level changes

## Pull Request Process

1. Ensure all tests pass
2. Ensure code passes ruff checks
3. Update documentation
4. Fill out the PR template completely
5. Request review

## Performance Benchmarks

Before submitting performance improvements:

1. Run the benchmark suite:
   ```bash
   cd benchmarks
   python run_all_benchmarks.py
   ```

2. Document the improvement:
   - Before/after timings
   - Test data size
   - System specs

3. Include results in PR description

## Code Review

- Be respectful and constructive
- Respond to feedback promptly
- Make requested changes in new commits
- Squash commits before merge (if requested)

## Release Process

Only maintainers can create releases:

1. Update version in `pyproject.toml` and `src/gsmod/__init__.py`
2. Update CHANGELOG or release notes
3. Commit and push to main
4. Create GitHub release with tag: `v0.2.0`
5. Write release notes
6. GitHub Actions will build and publish to PyPI

## Questions?

- Open an issue for questions
- Check existing issues and PRs
- Read the documentation (README.md, AGENTS.md, CLAUDE.md)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

