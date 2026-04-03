# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`multidms` is a Python package for jointly modeling deep mutational scanning (DMS) experiments. It estimates individual mutation effects and condition-specific shifts across experiments that may have different wildtype sequences, using global epistasis models implemented in JAX.

## Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run full test suite (unit tests + doctests)
pytest --doctest-modules -vv

# Run only unit tests
pytest tests/

# Run a single test file
pytest tests/test_model.py -vv

# Run a single test function
pytest tests/test_model.py::test_function_name -vv

# Lint
ruff check .

# Format
black .

# Build docs
make -C docs clean && make -C docs html

# Version bump (commits + tags, does NOT push)
bumpver update --patch  # or --minor or --major
```

### pixi (alternative)

```bash
pixi install              # one-command env setup
pixi run test             # pytest --doctest-modules multidms tests
pixi run lint             # ruff check .
pixi run fmt              # black .
pixi run fmt-check        # black --check .
pixi run docs             # build Sphinx docs
pixi run -e py39 test     # test against Python 3.9
```

## Architecture

### Dual API: Legacy wrapper vs JAX-native

The package has two layers:

1. **`jaxmodels`** — The core JAX-native API. Uses equinox modules, BCOO sparse arrays, and jaxopt for optimization. Key classes:
   - `jaxmodels.Data` (equinox Module) — holds JAX arrays for one condition
   - `jaxmodels.Latent` — models latent phenotypes
   - `jaxmodels.Model` — global epistasis model with fitting/loss functions

2. **`data.py` / `model.py`** — The user-facing wrapper API. `Data` handles pandas DataFrames, one-hot encoding via `binarymap`, and multi-condition bookkeeping. `Model` wraps `jaxmodels` with a friendlier interface. Conversion between layers: `jaxmodels.Data.from_multidms()`.

### Key module roles

- **`model_collection.py`** — `ModelCollection` and `fit_models()` for parallel fitting across parameter grids using `ThreadPoolExecutor`. Includes cross-validation, mutation DataFrames, and Altair visualization methods.
- **`plot.py`** — Interactive Altair-based visualizations (heatmaps, lineplots) for mutation effects.
- **`utils.py`** — Mutation string parsing (`split_sub`, `split_subs`), parameter transforms, difference matrices.

### Data flow

User provides a pandas DataFrame with columns for condition, substitutions, and functional scores → `Data` class creates `BinaryMap` one-hot encodings per condition → `jaxmodels.Data.from_multidms()` converts to JAX sparse arrays → `jaxmodels.Model` fits via gradient-based optimization → results extracted back through `Model` wrapper.

### Key design patterns

- **Sparse computation**: Variant-mutation matrices use JAX BCOO sparse format for memory efficiency.
- **Multi-condition modeling**: Conditions share a reference beta vector; non-reference conditions get shift parameters capturing condition-specific effects.
- **Loss types**: `functional_score_loss` (regression on scores) and `count_loss` (likelihood on pre/post selection counts).
- **GE nonlinearity**: `Identity` (linear) or `Sigmoid` (nonlinear global epistasis).

## Code Style

- **Formatter**: Black (line length default 88)
- **Linter**: Ruff with E, F, UP, D rules; line length 89; Google-style docstrings
- **Docstrings**: Google convention (via ruff pydocstyle)
- **Type hints**: Uses `jaxtyping` for JAX array shape/dtype annotations (e.g., `Float[Array, "n_variants"]`)

## CI

GitHub Actions runs on push to main and PRs: ruff lint → black format check → pytest with doctests → docs build. Tested on Python 3.9/3.10/3.11 across ubuntu and macos.

## Active Technologies
- Python 3.9+ (matches existing CI matrix) + multidms (this package), snakemake, papermill, jupyter, matplotlib, seaborn, pandas, numpy, pyyaml (002-simulation-pipeline)
- CSV intermediate files + pickle for fitted model collections; all in `experiments/simulation/results/` (002-simulation-pipeline)
- Python 3.9+ (matches existing CI matrix) + multidms (this package), snakemake, papermill, jupyter, matplotlib, seaborn, pandas, numpy, pyyaml, requests (for data download) (003-spike-pipeline)
- CSV intermediate files + pickle for fitted model collections; all in `experiments/scv2-spike/results/` (003-spike-pipeline)

## Recent Changes
- 002-simulation-pipeline: Added Python 3.9+ (matches existing CI matrix) + multidms (this package), snakemake, papermill, jupyter, matplotlib, seaborn, pandas, numpy, pyyaml
