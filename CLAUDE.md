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
pixi run dashboard        # launch interactive marimo dashboard (read-only)
pixi run dashboard-edit   # launch marimo dashboard in edit mode
```

## Architecture

### Dual API: Legacy wrapper vs JAX-native

The package has two layers:

1. **`jaxmodels`** — The core JAX-native API. Uses equinox modules, BCOO sparse arrays, and jaxopt for optimization. Key classes:
   - `jaxmodels.Data` (equinox Module) — holds JAX arrays for one condition
   - `jaxmodels.Latent` — models latent phenotypes
   - `jaxmodels.Model` — global epistasis model with fitting/loss functions. `Model.α` is a shared scalar (not per-condition).

2. **`data.py` / `model.py`** — The user-facing wrapper API. `Data` handles pandas DataFrames, one-hot encoding via `binarymap`, and multi-condition bookkeeping. `Model` wraps `jaxmodels` with a friendlier interface. Conversion between layers: `jaxmodels.Data.from_multidms()`.

### Key module roles

- **`model_collection.py`** — `ModelCollection` and `fit_models()` for parallel fitting across parameter grids using `ThreadPoolExecutor`. Includes cross-validation, mutation DataFrames, and thin visualization wrappers that delegate to `plot.py`.
- **`plot.py`** — All interactive Altair-based visualizations. Every public function takes a DataFrame and returns an `alt.Chart`. Class methods on `Data`, `Model`, and `ModelCollection` are thin wrappers that delegate here.
- **`utils.py`** — Mutation string parsing (`split_sub`, `split_subs`), parameter transforms, difference matrices.

### Data flow

User provides a pandas DataFrame with columns for condition, substitutions, and functional scores → `Data` class creates `BinaryMap` one-hot encodings per condition → `jaxmodels.Data.from_multidms()` converts to JAX sparse arrays → `jaxmodels.Model` fits via gradient-based optimization → results extracted back through `Model` wrapper.

### Key design patterns

- **Sparse computation**: Variant-mutation matrices use JAX BCOO sparse format for memory efficiency.
- **Multi-condition modeling**: Conditions share a reference beta vector; non-reference conditions get shift parameters capturing condition-specific effects.
- **Loss types**: `functional_score_loss` uses per-variant `.mean()` Huber loss so conditions contribute equally regardless of variant count. `count_loss` uses `.sum()` (total NLL). Hyperparameter values are calibrated to the `.mean()` scale. As a consequence, `ModelCollection.fit_models["total_loss_*"]` columns are already per-variant averages (one per condition, plus `"total"` averaged over conditions) — do not divide again by variant count when plotting or you will produce doubly-normalized values.
- **GE nonlinearity**: `Identity` (linear) or `Sigmoid` (nonlinear global epistasis). Alpha is shared across all conditions to prevent per-condition sigmoid degeneracy.
- **`beta0_ridge`**: standard ridge on β0 — sums `β0**2` across **all** conditions (reference included). Not a penalty on inter-condition differences, despite the name. See `multidms.jaxmodels._beta_ridge_penalty`.
- **Plotting separation**: Rendering logic is fully separated from data preparation. Classes own data extraction (`get_*_df`, `_prepare_*_df`); `plot.py` owns chart construction.
- **Interactive dashboard**: `experiments/dashboard.py` is a marimo app for exploring `ModelCollection` results interactively. Launch with `pixi run dashboard`.

## Code Style

- **Formatter**: Black (line length default 88)
- **Linter**: Ruff with E, F, UP, D rules; line length 89; Google-style docstrings
- **Docstrings**: Google convention (via ruff pydocstyle)
- **Type hints**: Uses `jaxtyping` for JAX array shape/dtype annotations (e.g., `Float[Array, "n_variants"]`)

## CI

GitHub Actions runs on push to main and PRs: ruff lint → black format check → pytest with doctests → docs build. Tested on Python 3.9/3.10/3.11 across ubuntu and macos.

## Remote Pipeline Execution

Before launching any remote pipeline:

1. **Create a local worktree** (if not on main): `git worktree add ../multidms-wt-<branch> <branch>`
2. **Scout first**: `bip scout` — pick a server with <20% CPU
3. **Launch** (from the worktree): `pixi run remote-pipeline -- <pipeline> <profile> host=<server>`
   - pipeline: `simulation` or `spike`
   - profile: `test`, `experimental`, or `prod`
   - Auto-creates remote worktree at `$remote_dir/../multidms-worktrees/<branch>/`
   - Auto-generates `output_dir=results-<profile>-<branch>`
   - Runs `pixi install` then snakemake in the remote worktree
4. **Monitor**: `pixi run remote-status -- <pipeline> <profile> host=<server>`
5. **Fetch results**: `pixi run run-pull -- <pipeline> <profile> host=<server>`
6. **Preserve results** (optional): Copy results dir to main clone before removing worktrees
7. **Clean up**: Remove local worktree, remote worktree, and tmux session

Convention: `output_dir` is always `results-<profile>-<branch>` (e.g., `results-prod-fix-alpha`).

**`beta_clip_range` in YAML configs**: use a single `null` (scalar) to disable clipping — not `[null, null]`, which deserializes to a 2-tuple of `None`s and crashes at `jnp.clip` argument unpacking.

Never skip step 2. Never leave tmux sessions running after fetching results.

## Active Technologies
- marimo (interactive dashboard for exploring ModelCollection results)
- Python 3.9+ (matches existing CI matrix) + multidms (this package), snakemake, papermill, jupyter, matplotlib, seaborn, pandas, numpy, pyyaml (002-simulation-pipeline)
- CSV intermediate files + pickle for fitted model collections; all in `experiments/simulation/results/` (002-simulation-pipeline)
- Python 3.9+ (matches existing CI matrix) + multidms (this package), snakemake, papermill, jupyter, matplotlib, seaborn, pandas, numpy, pyyaml, requests (for data download) (003-spike-pipeline)
- CSV intermediate files + pickle for fitted model collections; all in `experiments/scv2-spike/results/` (003-spike-pipeline)
- Snakemake experiment pipeline in `experiments/loss-normalization/` for validating `.mean()` loss normalization against V0.4.0 hyperparameter anchors (fusionreg × l2reg 2D grid)

## Recent Changes
- 002-simulation-pipeline: Added Python 3.9+ (matches existing CI matrix) + multidms (this package), snakemake, papermill, jupyter, matplotlib, seaborn, pandas, numpy, pyyaml
