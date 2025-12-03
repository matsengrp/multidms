# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`multidms` is a Python package for joint modeling of multiple deep mutational scanning (DMS) experiments. It uses JAX for high-performance computing and automatic differentiation to fit global-epistasis models that estimate individual mutation effects and how they differ between experimental conditions.

## Development Commands

### Core Development Workflow
```bash
# Install development dependencies
pip install -e ".[dev]"

# Code quality checks
ruff check .              # Lint code
black .                   # Format code

# Testing
pytest --doctest-modules -vv    # Run all tests including doctests
pytest tests/                   # Run unit tests only
pytest --doctest-modules multidms tests  # Full test suite (as used in CI)

# Documentation
make -C docs clean        # Clean docs build
make -C docs html         # Build documentation

# Version management
bumpver update --patch    # Bump patch version
bumpver update --minor    # Bump minor version
bumpver update --major    # Bump major version
```

### Testing Notes
- The test suite is minimal with only basic Data class tests in `tests/test_data.py`
- Doctests are integrated throughout the codebase and run with pytest
- CI runs tests on Python 3.9, 3.10, 3.11 on Ubuntu and macOS

## Package Architecture

### Core Classes and Entry Points
- **`multidms.Data`** - Handles data preprocessing and one-hot encoding of variant substitutions
- **`multidms.Model`** - Main model class for fitting DMS experiments using JAX-based optimization
- **`multidms.ModelCollection`** - Interface for fitting multiple models in parallel
- **`multidms.fit_models`** - Function for parallel model fitting across collections

### Key Modules
- **`multidms.biophysical`** - Core biophysical model equations, transformations, and mathematical foundations
- **`multidms.model_collection`** - Parallel model fitting and analysis workflows
- **`multidms.plot`** - Interactive plotting functionality using matplotlib/seaborn/altair
- **`multidms.utils`** - Data transformation utilities and helper functions

### Dependencies and Architecture Patterns
- **JAX ecosystem**: Core computational framework with jaxopt for optimization
- **Data handling**: pandas for DataFrames, numpy for arrays (version pinned ≤1.26.0)
- **Optimization**: Uses generalized lasso with bit-flipping algorithms via pylops/pyproximal
- **Visualization**: Multi-library approach (matplotlib, seaborn, altair) for different plot types
- **Scientific computing**: scipy for statistical functions, polyclonal for related modeling

### Code Style and Conventions
- **Formatting**: Black with line length 89 (matches ruff configuration)
- **Linting**: Ruff with specific rule selections (E, F, UP, D) and custom ignores for docstring styles
- **Documentation**: NumPy-style docstrings throughout
- **Type hints**: Used where appropriate, with typing_extensions for compatibility

### Development Patterns
- Models compose biophysical equations from `multidms.biophysical` module
- Heavy use of JAX transformations (jit, grad, vmap) for performance
- Parameter initialization and transformation handled through dedicated methods
- Cross-validation and simulation validation workflows built into model classes

### File Organization
- Main package code in `multidms/` with flat module structure
- Jupyter notebooks in `notebooks/` demonstrate usage and validation
- Sphinx documentation in `docs/` with linked notebook examples
- Minimal test suite in `tests/` (expansion needed)

### CI/CD and Release Process
- GitHub Actions handle testing, linting, documentation builds
- Automated PyPI publishing on tagged releases
- Version management via bumpver tool with coordinated updates across files
- Multi-platform testing ensures compatibility across development environments

## Active Technologies
- Python 3.9, 3.10, 3.11 (multi-version CI support) + JAX ≥0.4.29, jaxopt, equinox, pandas ≥2.2.0, numpy ≤1.26.0 (001-jaxmodels-refactor)
- N/A (library operates on in-memory pandas DataFrames) (001-jaxmodels-refactor)
- Python 3.9, 3.10, 3.11 (CI tested on all three versions) + JAX ≥0.4.29, equinox, jaxopt, pandas ≥2.2.0, numpy ≤1.26.0 (001-jaxmodels-refactor)
- N/A (in-memory DataFrame processing) (001-jaxmodels-refactor)

## Recent Changes
- 001-jaxmodels-refactor: Added Python 3.9, 3.10, 3.11 (multi-version CI support) + JAX ≥0.4.29, jaxopt, equinox, pandas ≥2.2.0, numpy ≤1.26.0
