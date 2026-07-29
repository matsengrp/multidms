# SARS-CoV-2 Spike Analysis Pipeline

Analyzes deep mutational scanning data for SARS-CoV-2 spike variants across three conditions (Delta, BA.1, BA.2). Downloads raw data from the public repository, applies count-aggregation data preparation, fits multidms models across a regularization grid, and evaluates model quality.

## Pipeline DAG

```
config.yaml
    │
    ▼
prepare_data ──► training_functional_scores.csv
    │
    ├──────────────────────┐
    ▼                      ▼
fit_models            cross_validation
──► fit_collection.pkl ──► cross_validation_loss.csv
    │
    ▼
evaluate ──► mutations_df.csv
          ──► collection_muts.csv
          ──► fit_sparsity.csv
          ──► library_replicate_correlation.csv
```

## Running

```bash
# Test profile (<10 minutes, 10% subsample)
pixi run spike-test

# Production profile (full data, manuscript defaults)
pixi run spike-prod

# Or directly via Snakemake
snakemake -s experiments/scv2-spike/Snakefile --config profile=test -j4
```

## File Manifest

### Intermediate files (in `results/`)

| File | Produced by | Description |
|------|------------|-------------|
| `training_functional_scores.csv` | prepare_data | Variant functional scores after count aggregation, filtering, and clipping |
| `fit_collection.pkl` | fit_models | Pickled DataFrame of fitted Model objects across regularization grid |
| `cross_validation_loss.csv` | cross_validation | Training vs validation loss across regularization grid |
| `mutations_df.csv` | evaluate | Per-mutation beta and shift parameters at chosen lasso strength |
| `collection_muts.csv` | evaluate | Per-mutation parameters at all fusionreg values |
| `fit_sparsity.csv` | evaluate | Shift sparsity fraction across regularization grid |
| `library_replicate_correlation.csv` | evaluate | Replicate correlation of mutation parameters |

### Configuration (in `config/`)

| File | Description |
|------|-------------|
| `config.yaml` | Production profile — fit tier |
| `config_downstream.yaml` | Production profile — downstream tier |
| `config_test.yaml` | Test profile — 10% subsample, 2 fusionreg values, 20 iterations |
| `config_test_downstream.yaml` | Test profile — downstream tier |

Every config variant has a matching `<name>_downstream.yaml` sibling. See
**Configuration tiers** below for which keys live where and why it matters.

## Configuration tiers

The config is split by dependency tier so that a downstream-only edit cannot
invalidate the expensive model fit:

| File | Holds | Do edits invalidate the fit? |
|------|-------|------------------------------|
| `config.yaml` | `seed`, `train_frac`, data sourcing and filtering, the `fitting:` block, `reference`, `output_dir`, `skip_cross_validation`, experiment/condition membership | **Yes** |
| `config_downstream.yaml` | `lasso_choice`, `condition_colors`, `condition_titles`, `domain_dict` | **No** |

Snakemake reruns a job when an `input:` file's mtime changes, not when its
content changes. Before the split, every rule declared the single config as an
input, so editing `lasso_choice` — a key the fit never reads — forced a
multi-hour refit that recomputed a byte-identical `fit_collection.pkl`. Even
`touch config.yaml` was enough.

Now only `rule evaluate` declares `config_downstream.yaml` as an `input:`.
`prepare_data` and `cross_validation` read it too (for plot labels), but
receive its **path via `params:`** — Snakemake does not rerun on `params:`
changes, so a color edit cannot reach the fit.

> Do not "tidy" this by adding `downstream_config` to the `input:` block of
> `prepare_data`, `cross_validation`, or `fit_models`. That would silently
> restore the defect. See issue #287.

Retuning the chosen lasso weight, changing a plot color, or adding a
downstream analysis therefore reuses the cached `fit_collection.pkl`.

### Source notebooks (in `notebooks/`)

| Notebook | Description |
|----------|-------------|
| `prepare_data.ipynb` | Download raw data, count aggregation, filtering, functional score computation, replicate correlations |
| `fit_models.ipynb` | Fit multidms models across fusionreg grid per replicate (independent, parallel) |
| `fit_models_path.ipynb` | Fit a warm-started continuation path along ascending fusionreg (sequential per replicate) |
| `cross_validation.ipynb` | 80/20 train/test CV across regularization grid |
| `evaluate.ipynb` | Convergence diagnostics, sparsity, replicate correlations, GE plots, mutation export |

## Fitting strategy

`spike.fitting.strategy` in the active config selects which notebook
`rule fit_models` runs:

- `"independent"` (default) — each `(replicate, fusionreg)` combination is
  fit from scratch in parallel, as in every prior release.
- `"continuation"` — each replicate's models are fit sequentially along the
  sorted `fusionreg_values` grid, warm-starting `(β, β0, α)` from the
  previous step. Use this when a strong shift lasso distorts the global
  epistasis calibration for data-poor conditions — empirically this was
  the Delta pathology at high `fusionreg` in the independent-fit results.
  The output `fit_collection.pkl` has the same schema, so downstream rules
  are unchanged.
