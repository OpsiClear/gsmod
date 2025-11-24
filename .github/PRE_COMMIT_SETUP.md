# Pre-commit Setup Guide

This project uses [pre-commit](https://pre-commit.com/) to automatically check code quality before commits.

## Installation

### 1. Install pre-commit

Pre-commit is included in the dev dependencies:

```bash
pip install -e ".[dev]"
```

Or install it separately:

```bash
pip install pre-commit
```

### 2. Install the git hooks

```bash
pre-commit install
```

This will install the pre-commit hook that runs automatically before each commit.

Optionally, also install the pre-push hook:

```bash
pre-commit install --hook-type pre-push
```

## Usage

### Automatic (Recommended)

Once installed, pre-commit will run automatically on staged files before each commit:

```bash
git add .
git commit -m "Your commit message"
# Pre-commit hooks run automatically
```

If any hook fails, the commit will be aborted and files will be auto-fixed (where possible). Review the changes, stage them, and commit again.

### Manual Run

Run on all files:

```bash
pre-commit run --all-files
```

Run on specific files:

```bash
pre-commit run --files src/gsmod/pipeline.py
```

Run a specific hook:

```bash
pre-commit run ruff --all-files
pre-commit run ruff-format --all-files
```

## What Gets Checked

### Ruff (Python Linting & Formatting)
- **Linting**: Checks for code quality issues (unused imports, undefined names, etc.)
- **Formatting**: Ensures consistent code style (line length, quotes, spacing)
- **Auto-fix**: Many issues are fixed automatically

### File Cleanup
- **Trailing whitespace**: Removes trailing spaces
- **End of file**: Ensures files end with a newline
- **Line endings**: Normalizes line endings

### File Validation
- **YAML/TOML/JSON**: Validates syntax
- **Large files**: Prevents accidentally committing large files (>1MB)
- **Merge conflicts**: Detects merge conflict markers

### Python-Specific
- **AST check**: Validates Python syntax
- **Debug statements**: Detects leftover debug code
- **Test naming**: Ensures test files follow pytest conventions

## Updating Hooks

Update to the latest hook versions:

```bash
pre-commit autoupdate
```

## Bypassing Hooks (Not Recommended)

In rare cases where you need to bypass hooks:

```bash
git commit --no-verify -m "Emergency fix"
```

**Note**: CI will still run all checks, so bypassing locally only delays the inevitable.

## Troubleshooting

### Hooks are slow on first run

Pre-commit installs hook environments on first run. Subsequent runs are much faster.

### Hook fails but I don't see why

Run with verbose output:

```bash
pre-commit run --all-files --verbose
```

### I want to skip a specific hook

Edit `.pre-commit-config.yaml` and comment out the hook, or use `SKIP`:

```bash
SKIP=ruff git commit -m "Skip ruff for this commit"
```

### Clear hook cache

If hooks are misbehaving:

```bash
pre-commit clean
pre-commit install --install-hooks
```

## CI Integration

Pre-commit runs automatically in CI on all pull requests. The same checks that run locally will run in CI, so passing pre-commit locally means your PR will pass CI checks.

## More Information

- [Pre-commit documentation](https://pre-commit.com/)
- [Ruff documentation](https://docs.astral.sh/ruff/)

