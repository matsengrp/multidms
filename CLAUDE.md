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
- **Fitting strategies**: `fit_models()` fits each `(dataset, hyperparameter)` combination independently in parallel (CPU processes or GPU devices). `fit_models_path()` fits the same combinations sequentially along the `fusionreg` axis, warm-starting each step from the previous fit's `(β, β0, α)`. Both return the same DataFrame schema and feed into `ModelCollection`. Use the path fitter when a strong shift lasso distorts data-poor conditions under independent fitting. In the spike pipeline, the choice is driven by `spike.fitting.strategy: "independent" | "continuation"` in the active YAML config.
- **`plot.py`** — All interactive Altair-based visualizations. Every public function takes a DataFrame and returns an `alt.Chart`. Class methods on `Data`, `Model`, and `ModelCollection` are thin wrappers that delegate here.
- **`utils.py`** — Mutation string parsing (`split_sub`, `split_subs`), parameter transforms, difference matrices.

### Data flow

User provides a pandas DataFrame with columns for condition, substitutions, and functional scores → `Data` class creates `BinaryMap` one-hot encodings per condition → `jaxmodels.Data.from_multidms()` converts to JAX sparse arrays → `jaxmodels.Model` fits via gradient-based optimization → results extracted back through `Model` wrapper.

### Key design patterns

- **Sparse computation**: Variant-mutation matrices use JAX BCOO sparse format for memory efficiency.
- **Multi-condition modeling**: Conditions share a reference beta vector; non-reference conditions get shift parameters capturing condition-specific effects.
- **Loss types**: `functional_score_loss` uses per-variant `.mean()` Huber loss so conditions contribute equally regardless of variant count. `count_loss` uses `.sum()` (total NLL). Hyperparameter values are calibrated to the `.mean()` scale. As a consequence, `ModelCollection.fit_models["total_loss_*"]` columns are already per-variant averages (one per condition, plus `"total"` averaged over conditions) — do not divide again by variant count when plotting or you will produce doubly-normalized values.
- **GE nonlinearity**: `Identity` (linear) or `Sigmoid` (nonlinear global epistasis). Alpha is shared across all conditions to prevent per-condition sigmoid degeneracy.
- **Plotting separation**: Rendering logic is fully separated from data preparation. Classes own data extraction (`get_*_df`, `_prepare_*_df`); `plot.py` owns chart construction.
- **Interactive dashboard**: `experiments/dashboard.py` is a marimo app for exploring `ModelCollection` results interactively. It discovers every `fit_collection.pkl` below the directory it is launched from (cwd), so it can explore any fitted collection, not just pipeline outputs. Launch with `pixi run dashboard`.

## Code Style

- **Formatter**: Black (line length default 88)
- **Linter**: Ruff with E, F, UP, D rules; line length 89; Google-style docstrings
- **Docstrings**: Google convention (via ruff pydocstyle)
- **Type hints**: Uses `jaxtyping` for JAX array shape/dtype annotations (e.g., `Float[Array, "n_variants"]`)

## CI

GitHub Actions runs on push to main and PRs: ruff lint → black format check → pytest with doctests → docs build. Tested on Python 3.9/3.10/3.11 across ubuntu and macos.

## Remote Pipeline Execution

### CRITICAL: set `n_processes` to at least the number of fits

**JAX/XLA leaks executable JIT mappings across sequential fits in one process.**
A worker that runs more than ~5 fits dies with `LLVM ERROR: Unable to allocate
section memory` and `JaxRuntimeError: INTERNAL: Failed to materialize symbols`
— on an idle host with 1.4 TB free. It is not a memory shortage: the failing
request was 980 bytes. The XLA dylib counter climbs (`xla_jit_dylib_12` →
`_19` → `_31`) and mappings are never released.

Failures land **by position in the work queue, not by hyperparameter.** A
serial run of the spike 20-fit grid passed indices 0–4 and failed 5–19, across
both replicates and every `fusionreg` including 0.0. Under a multi-worker pool
this surfaces as a misleading `ModelCollectionFitError: Failed fitting 1 of 20
parameter sets` — which reads like one bad parameter set but is just the first
worker to exhaust its address space.

**Rule: `n_processes >= n_fits`,** where `n_fits = len(fusionreg_values) ×
n_datasets`. `fit_models()` calls `p.map` with no `chunksize`, so it resolves
to `ceil(n_fits / (n_processes × 4)) == 1` and every fit gets a fresh process.

| pipeline | grid | `n_processes` | fits/worker | safe? |
|---|---|---|---|---|
| spike prod | 10 fusionreg × 2 reps = 20 | **20** | 1 | yes |
| spike prod (old) | 20 | 6 | 3–4 | **no — leaks** |
| simulation prod | 10 fusionreg × 6 datasets = 60 | 30 | 2 | yes, under the ~5 threshold |

Never set `n_processes: null` — auto-selection picks workers by core count
alone, ignoring both memory and this leak. Raising `n_processes` is cheap here:
each worker holds one `Data` object (~2–3 GB), so 20 workers is ~60 GB on a
1511 GB host.

If a fit ever needs more workers than the host has cores, fix the leak instead
of splitting the grid — the real repair is disposing of XLA state between fits
(or `maxtasksperchild=1` on the pool in `model_collection.py`).

### Launch procedure

1. **Create a local worktree** (if not on main): `git worktree add ../multidms-wt-<branch> <branch>`
2. **Scout first**: `bip scout` — pick a server with <20% CPU. Check available
   RAM and cores too, not just the CPU percentage.
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

Never skip step 2. Never leave tmux sessions running after fetching results.

### Manual launch (when `remote-pipeline` cannot reach GitHub)

`remote-pipeline` runs a remote `git fetch origin`, which fails when the
1Password SSH agent refuses to sign for **forwarded** sessions (`ssh-add -l`
lists the key, but every signature dies with `signing failed ... from agent`).
Retrying never fixes it. Push straight into the remote clone over the SSH
channel you already have:

```bash
git push <host>:/fh/fast/matsen_e/shared/multidms/multidms <branch>:refs/heads/<branch>
ssh <host> 'git -C <clone> worktree add <wt-dir> <branch>'
```

If the branch is already checked out in the remote worktree the push is
rejected; push to a temp ref and fast-forward instead:

```bash
git push <host>:<clone> <branch>:refs/heads/<branch>-incoming
ssh <host> 'git -C <wt-dir> merge --ff-only <branch>-incoming'
```

Then symlink the env into the new worktree (`.pixi/envs` **and** `pixi.lock` —
the lock is gitignored, so a fresh worktree lacks it and pixi re-solves every
platform from scratch) and drive snakemake from a here-doc'd script on the
remote, so `cd` sits on its own line where it cannot be dropped:

```bash
ssh <host> 'cat > ~/run-<branch>.sh <<"EOF"
#!/bin/bash -l
set -euo pipefail
export PATH="$HOME/.pixi/bin:$PATH"
cd <wt-dir>
pixi run --frozen snakemake -s experiments/scv2-spike/Snakefile \
  --config output_dir=results-<profile>-<branch> -j1 "$@"
EOF'
ssh <host> 'tmux new-session -d -s smk-<branch> "bash ~/run-<branch>.sh > ~/smk-<branch>.log 2>&1"'
```

`/fh/fast` is a **shared filesystem** — every orca host sees the same worktree,
env, and results. Moving hosts needs no re-sync, no re-download, and completed
rule outputs carry over. Always `--dry-run` first: a rule's declared outputs
are deleted at job start, and `config.yaml` is a rule `input:`, so any edit to
it (even a comment) invalidates the expensive fit.

### Monitoring a running fit

**Never judge liveness by `%CPU`** — `ps` reports a *lifetime average* that
decays slowly, so a deadlocked worker can sit at "60%" for hours. Sample
**cumulative CPU time** twice instead; if `TIME` is identical across a 20 s
gap, the process is doing nothing:

```bash
ssh <host> 'ps -p <pids> -o pid,stat,time --no-headers'   # run twice
```

`stat` of `S` with `wchan` `futex_wait_queue` / `pipe_read` across the whole
tree means a **`multiprocessing.Pool` deadlock**: `_fit_fun` swallows worker
exceptions (`except Exception: return None`, no logging, `model_collection.py`)
and `p.map` blocks forever when a child dies. Count the workers — fewer alive
than `n_processes` confirms it. See also
`~/.claude/.../project_convergence_maxiter100_spawn_deadlock.md`.

To find *which* fit fails, run the grid serially with per-fit capture
(`fit_one_model` in a `try/except` with `traceback.print_exc()`); the pipeline
itself only reports a count.

Other recurring snags:
- **HTTP 429 from `raw.githubusercontent.com`** in `prepare_data` — GitHub
  rate-limits the raw-data download. Seed the cache from a prior run instead:
  `cp -a <other-results>/raw_data <new-results>/raw_data` (verify md5s; do not
  `mkdir -p` the destination first or `cp -a` nests it one level deep).
- **Stale snakemake lock** after a killed run — verify no snakemake is running
  on *any* host sharing the filesystem, then `snakemake --unlock`.
- **`IncompleteFilesException`** after a kill — `--rerun-incomplete` regenerates
  only the interrupted output.
- **SSH `Connection closed by UNKNOWN port 65535`** — sshd `MaxStartups`
  throttling from too many rapid connections. Back off several minutes and
  batch probes into one session; it is not an auth failure.

## Active Technologies
- marimo (interactive dashboard for exploring ModelCollection results)
- Python 3.9+ (matches existing CI matrix) + multidms (this package), snakemake, papermill, jupyter, matplotlib, seaborn, pandas, numpy, pyyaml (002-simulation-pipeline)
- CSV intermediate files + pickle for fitted model collections; all in `experiments/simulation/results/` (002-simulation-pipeline)
- Python 3.9+ (matches existing CI matrix) + multidms (this package), snakemake, papermill, jupyter, matplotlib, seaborn, pandas, numpy, pyyaml, requests (for data download) (003-spike-pipeline)
- CSV intermediate files + pickle for fitted model collections; all in `experiments/scv2-spike/results/` (003-spike-pipeline)
- Snakemake experiment pipeline in `experiments/loss-normalization/` for validating `.mean()` loss normalization against V0.4.0 hyperparameter anchors (fusionreg × l2reg 2D grid)

## Recent Changes
- 002-simulation-pipeline: Added Python 3.9+ (matches existing CI matrix) + multidms (this package), snakemake, papermill, jupyter, matplotlib, seaborn, pandas, numpy, pyyaml
