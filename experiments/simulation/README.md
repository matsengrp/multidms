# Simulation Validation Pipeline

Generates synthetic DMS data with known ground-truth mutational effects and shifts, fits multidms models across a regularization grid, and evaluates model recovery against the truth. Produces publication-quality manuscript figures.

## Pipeline DAG

```
config.yaml
    │
    ▼
simulate_data ──► simulated_muteffects.csv
               ──► simulated_func_scores.csv
    │
    ├──────────────────────────┐
    ▼                          ▼
fit_models                cross_validation
──► fit_collection.pkl    ──► cross_validation_loss.csv
    │
    ▼
evaluate ──► model_vs_truth_beta_shift.csv
          ──► collection_muts.csv
          ──► fit_sparsity.csv
          ──► library_replicate_correlation.csv
          ──► model_vs_truth_variant_phenotype.csv
    │
    ▼
manuscript_figures
──► figures/main_figure.pdf
──► figures/ground_truth_correlation.pdf
──► figures/sparsity_diagnostic.pdf
```

## Running

```bash
# Test profile (<5 minutes)
pixi run sim-test

# Production profile (manuscript defaults)
pixi run sim-prod

# Or directly via Snakemake
snakemake -s experiments/simulation/Snakefile --config profile=test -j4
```

## File Manifest

### Intermediate files (in `results/`)

| File | Produced by | Description |
|------|------------|-------------|
| `simulated_muteffects.csv` | simulate_data | Ground-truth mutation effects (beta, shift) for both homologs |
| `simulated_func_scores.csv` | simulate_data | Variant functional scores across noise levels |
| `fit_collection.pkl` | fit_models | Pickled DataFrame of fitted Model objects across regularization grid |
| `model_vs_truth_beta_shift.csv` | evaluate | Pearson correlation and MAE of predicted vs true beta/shift |
| `collection_muts.csv` | evaluate | Per-mutation predicted vs true parameters at all fusionreg values |
| `fit_sparsity.csv` | evaluate | Fraction of shifts exactly zero per mutation type |
| `library_replicate_correlation.csv` | evaluate | Replicate correlation of mutation parameters |
| `model_vs_truth_variant_phenotype.csv` | evaluate | Variant phenotype prediction accuracy |
| `cross_validation_loss.csv` | evaluate | Training vs validation loss across regularization grid |

### Manuscript figures (in `results/figures/`)

| File | Description |
|------|-------------|
| `main_figure.pdf` | Composite figure: distributions (A) + latent vs functional score (B) + diagnostic metrics grid (C) |
| `ground_truth_correlation.pdf` | Predicted vs true mutation effects scatter |
| `sparsity_diagnostic.pdf` | Inferred shift vs beta for true-zero-shift mutations |

### Notebooks

| Notebook | Purpose | Inline visualizations |
|----------|---------|----------------------|
| `simulate_data` | Generate synthetic DMS data | Beta distribution, shift distribution (#3) |
| `fit_models` | Fit models across regularization grid | Fit summary |
| `evaluate` | Evaluate against ground truth, save metrics | Items #7-11: latent vs func score, pairplot, violin, log convergence, corr vs lambda |
| `visualize` | Diagnostic plots for chosen lasso | Items #5-6: GE sigmoid curves, enrichment vs latent |
| `manuscript_figures` | Publication figures (leaf) | Item #4: heatmap with WT markers; saves items #12-14 as PDF |

## Configuration

- **Production**: `config/config.yaml` — manuscript defaults (genelength=50, 9 fusionreg values, outer maxiter=200 / inner ge+cal maxiter=10)
- **Test**: `config/config_test.yaml` — reduced parameters (genelength=10, 2 fusionreg values, maxiter=20)

Each has a matching `<name>_downstream.yaml` sibling.

### Configuration tiers

The config is split by dependency tier so that a downstream-only edit cannot
invalidate the expensive model fit:

| File | Holds | Do edits invalidate the fit? |
|------|-------|------------------------------|
| `config.yaml` | `seed`, `train_frac`, the simulation parameters, the `fitting:` block, `output_dir` | **Yes** |
| `config_downstream.yaml` | `lasso_choice` (only) | **No** |

Snakemake reruns a job when an `input:` file changes, but not when a `params:`
value changes. (Snakemake 9.21 detects that change by hashing file content, not
by comparing mtimes — so a bare `touch` does not trigger a rerun.) Before the
split, every rule declared the single config as an input, so editing
`lasso_choice` — a key the fit never reads — forced a full refit that recomputed
a byte-identical `fit_collection.pkl`.

Now only `rule evaluate` and `rule manuscript_figures` declare
`config_downstream.yaml` as an `input:`. `simulate_data`, `fit_models`, and
`cross_validation` do not reference it at all.

> Do not "tidy" this by adding `downstream_config` to the `input:` block of
> `simulate_data`, `fit_models`, or `cross_validation`. That would silently
> restore the defect. See issue #287.

Unlike spike, simulation's downstream tier holds `lasso_choice` alone — this
pipeline has no `condition_colors`, `condition_titles`, or `domain_dict`.

## Run log

### 2026-07-29 — prod, issue #287 (`beta0_ridge`/`l2reg` promotion + maxiter split)

| | |
|---|---|
| Host | orca05 (64 cores, 1.5 TB RAM) |
| Branch / commit | `287-config-tier-split` @ `9c4b515` |
| Output dir | `results-prod-287-config-tier-split` |
| Wall-clock | **7519 s (125 min)** |
| `fit_collection.pkl` | 535,372,253 bytes (535 MB), 54 fitted rows |
| Workers | `n_processes: 6`, **~42 GB RSS each** at steady state |

**Parameter set** (verified on the fitted rows, not just the YAML):
`beta0_ridge=0.01`, `l2reg=1e-6`, `maxiter=200` (outer) / `10` (inner
`ge_kwargs`, `cal_kwargs`), fusionreg = the 9-value manuscript ladder.

> These are **convergence-lab-derived** values, not the manuscript's ridge
> weights (manuscript: β ridge 1e-7, α ridge 1e-3). `beta0_ridge` is
> *shift shrinkage* — it penalizes `(β0_d − β0_ref)²`, the intercept
> **differences**, not the intercept magnitudes.

#### AC7 — ground-truth shift recovery at `fusionreg = 8e-5`

Pre-registered gate: no cell may drop more than 0.02 absolute, and the
six-cell mean may not drop.

| measurement_type | library | baseline | new | Δ |
|---|---|---|---|---|
| observed_phenotype | lib_1 | 0.8319 | 0.9657 | **+0.1338** |
| loose_bottle | lib_1 | 0.8071 | 0.9401 | **+0.1329** |
| tight_bottle | lib_1 | 0.7382 | 0.8197 | **+0.0815** |
| observed_phenotype | lib_2 | 0.8174 | 0.9696 | **+0.1522** |
| loose_bottle | lib_2 | 0.8011 | 0.9536 | **+0.1525** |
| tight_bottle | lib_2 | 0.7390 | 0.8282 | **+0.0892** |

Six-cell mean **0.7891 → 0.9128 (+0.1237)**; worst per-cell change is
**+0.0815**. All six cells improved. **AC7: PASS.**

`beta` recovery moved −0.0009 to −0.0030 (e.g. 1.0000 → 0.9970) against
near-perfect baselines. AC7 gates on `shift`, not `beta`.

#### Sizing note — why this does not run on a laptop

Each simulation fit worker holds **~42 GB resident** at steady state, ~6×
spike's footprint. `n_processes: null` selects `min(cpu_count // 2, n_models)`
from **core count alone, with no regard for memory**: on a 36 GB laptop that
chose 7 workers (~294 GB of demand) and kernel-panicked the host mid-run,
losing a complete production fit. A single worker exceeds that machine's
total RAM. Always pin `n_processes` explicitly, and size it against memory
rather than cores.
