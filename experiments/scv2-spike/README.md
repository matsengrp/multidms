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
    │                  ──► cv_fit_collection.pkl
    │                  ──► cv_convergence.csv
    ▼                      │
evaluate ──► mutations_df.csv
          ──► collection_muts.csv
          ──► fit_sparsity.csv
          ──► library_replicate_correlation.csv
          ──► fit_convergence.csv
          ──► convergence_trajectory.csv
          ──► ge_landscape_variants.csv
          ──► ge_landscape_curve.csv
          ──► ge_params.csv
    │                      │
    └──────────┬───────────┘
               ▼
      manuscript_figures ──► figures/*.pdf, figures/*.png
                         ──► raw_data/validation/viral_titers.csv
                         ──► raw_data/validation/spike_validation_data.csv
```

`manuscript_figures` reads the exported CSVs only — never
`fit_collection.pkl` — and is **skipped entirely** when
`skip_cross_validation` is true (see below).

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
| `cv_fit_collection.pkl` | cross_validation | Pickled CV fits, written after `add_eval_loss` so it carries validation losses |
| `cv_convergence.csv` | cross_validation | Per-CV-fit convergence; `fit N models, M failed` counts crashes, not convergence |
| `mutations_df.csv` | evaluate | Per-mutation beta and shift parameters at chosen lasso strength |
| `collection_muts.csv` | evaluate | Per-mutation parameters at all fusionreg values, keyed by `dataset_name` and `fusionreg` |
| `fit_sparsity.csv` | evaluate | Shift sparsity fraction across regularization grid |
| `library_replicate_correlation.csv` | evaluate | Replicate correlation of mutation parameters |
| `fit_convergence.csv` | evaluate | Per-fit `converged`, sweep count, and drift (`argmin_sweep`, `drift_frac`) |
| `convergence_trajectory.csv` | evaluate | Tidy per-sweep objective/loss trace; the data behind figure S16 |
| `ge_landscape_variants.csv` | evaluate | Per-variant latent phenotype and fitness at `lasso_choice` (figure S17) |
| `ge_landscape_curve.csv` | evaluate | Fitted sigmoid curve points at `lasso_choice` |
| `ge_params.csv` | evaluate | Per-condition `alpha`, `beta0`, `bundle_sum`, `wildtype_latent` |
| `manuscript_figures.ipynb` | manuscript_figures | Executed figure notebook (the rendering log for the nine figures below) |
| `raw_data/validation/viral_titers.csv` | manuscript_figures | Viral titers for figure 5, fetched from the legacy analysis repo at a pinned commit and cached |
| `raw_data/validation/spike_validation_data.csv` | manuscript_figures | Per-mutation validation measurements for figure 5, same pinned source |

### Manuscript figures (in `results/figures/`)

Each figure is written in every format listed under `spike.figures.formats`
in the downstream config — by default both `.pdf` (what the manuscript build
consumes) and `.png` (what the docs page shows). Every format is a declared
rule output, so a figure that fails to render fails the run instead of
silently vanishing into notebook output.

| Manuscript label | Filename (`.pdf` and `.png`) |
|------------------|------------------------------|
| S6 | `raw_data_summary_barcodes_backgrounds_hist` |
| S7 | `replicate_functional_score_correlation_scatter` |
| S9 | `shrinkage_analysis_trace_plots_beta` |
| S11 | `percent_shifts_under_x_lineplot` |
| S12 | `shift_corr_Delta_BA2` |
| S16 | `convergence_all_lasso_lines` |
| S17 | `global_epistasis_and_prediction_correlations` |
| fig4 | `shift_by_site_heatmap_zoom` |
| fig5 | `validation_titer_fold_change` |

> The S9 filename is deliberate. Two near-identical names exist in the legacy
> repo (`..._trace_plots.pdf` and `..._trace_plots_with_epistasis.pdf`); both
> are unused, and the manuscript build consumes
> `shrinkage_analysis_trace_plots_beta.pdf`. Do not "correct" it to the
> shorter name.

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
| `config_downstream.yaml` | `lasso_choice`, `condition_colors`, `condition_titles`, `domain_dict`, the `figures:` block | **No** |

Snakemake reruns a job when an `input:` file changes, but not when a `params:`
value changes. (Snakemake 9.21 detects that change by hashing file content, not
by comparing mtimes — so a bare `touch` does not trigger a rerun.) Before the
split, every rule declared the single config as an input, so editing
`lasso_choice` — a key the fit never reads — forced a multi-hour refit that
recomputed a byte-identical `fit_collection.pkl`.

Only `rule evaluate` and `rule manuscript_figures` declare
`config_downstream.yaml` as an `input:`. `prepare_data` and `cross_validation`
read it too (for plot labels), but receive its **path via `params:`** —
Snakemake does not rerun on `params:` changes, so a color edit cannot reach
the fit.

> Do not "tidy" this by adding `downstream_config` to the `input:` block of
> `prepare_data`, `cross_validation`, or `fit_models`. That would silently
> restore the defect. See issue #287.

Retuning the chosen lasso weight, changing a plot color, or adding a
downstream analysis therefore reuses the cached `fit_collection.pkl`.

### The figure tier

`spike.figures` — output formats, DPI, the excluded `fusionreg` rung, and the
figure-4/figure-5 knobs — lives in `config_downstream.yaml`, so editing it
re-runs `manuscript_figures` alone and never invalidates
`fit_collection.pkl` (a **~2 h 20 min** refit).

`rule manuscript_figures` keeps that property on the code side too. Note what
is deliberately **absent** from its `input:`: `config=CONFIG_PATH` and
`common=COMMON`. The fit-tier config path still reaches the notebook via
`params:`, which Snakemake does not track. The rule's helper module is
`notebooks/_downstream.py`, a deliberate sibling of `_common.py` rather than
an addition to it — `_common.py` is `input:` on all four fit-tier rules, so a
figure helper added there would make an edit to plotting code invalidate the
fit. `_downstream.py` is `input:` on `manuscript_figures` only.

The rule also reads **CSVs only**. It must never load `fit_collection.pkl`
(1.76 GB); everything it needs was exported by `evaluate` for exactly that
reason.

> `manuscript_figures` is **skipped when `skip_cross_validation` is true.**
> Figure S9's middle panel is the held-out loss trace, which needs
> `cross_validation_loss.csv`. With CV skipped there is no honest way to draw
> it, so the whole rule is dropped rather than emitting an S9 with a missing
> panel.

### Source notebooks (in `notebooks/`)

| Notebook | Description |
|----------|-------------|
| `prepare_data.ipynb` | Download raw data, count aggregation, filtering, functional score computation, replicate correlations |
| `fit_models.ipynb` | Fit multidms models across fusionreg grid per replicate (independent, parallel) |
| `fit_models_path.ipynb` | Fit a warm-started continuation path along ascending fusionreg (sequential per replicate) |
| `cross_validation.ipynb` | 80/20 train/test CV across regularization grid |
| `evaluate.ipynb` | Convergence diagnostics, sparsity, replicate correlations, GE plots, mutation export |
| `manuscript_figures.ipynb` | Render the nine manuscript figures from the exported CSVs |
| `_common.py` | Fit-tier helpers; `input:` on all four fit-tier rules |
| `_downstream.py` | Figure-tier helpers (validation-data fetch, `savefig`, `lasso_slice`, `set_plot_style`); `input:` on `manuscript_figures` only |

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

> **The Delta pathology is substantially an under-convergence artifact.**
> It was characterized when prod ran `maxiter: 50`. Raising the outer
> `maxiter` to 200 (and lowering the inner `ge_kwargs`/`cal_kwargs` maxiter
> to 10) removes most of it under the *independent* fitter — no continuation
> needed. Shift replicate correlation for `shift_Delta` at
> `fusionreg=6.4e-4` goes 0.0778 (baseline) → 0.4047, and the two drops seen
> at `maxiter=50` (−0.111 at 1.6e-4, −0.089 at 3.2e-4 for BA2) both turn
> positive. At a strong lasso the optimizer simply had not converged.
>
> This does not retire `"continuation"` — that strategy was never re-tested
> at `maxiter: 200`, so its marginal value now is unmeasured. Treat the
> paragraph above as motivation recorded at `maxiter: 50`, not as a
> standing result. See issue #287.

## Run log

### 2026-08-12 — prod, issue #292 Stage 1 (tol 1e-6, final)

| | |
|---|---|
| Host | orca03 (64 cores, 1.5 TB RAM) |
| Branch / commit | `292-spike-fit-tuning` @ `9cf9109` |
| Output dir | `results-prod-292-spike-fit-tuning` |
| Wall-clock | **8400 s (2 h 20 min)**, 18:43–21:03 (vs 90 min at tol 1e-5) |
| `fit_collection.pkl` | 1,761,133,465 bytes (1.76 GB), 20 fitted rows |
| `cv_fit_collection.pkl` | 1,385,867,352 bytes (1.39 GB), 20 fitted rows |
| Workers | `n_processes: 6`, **~35–38 GB RSS each** at steady state |

**Parameter set**: `recompute_scale=false`, `tol=1e-6`, `maxiter=500` (outer) /
`10` (inner), `beta0_ridge=0.01`, `l2reg=1e-6`, fusionreg = the 9-value
manuscript ladder plus the excluded `1.28e-3` probe rung.

**Convergence — 18/18 on the analysed ladder**, in 73–319 outer sweeps against
the 500 ceiling. The one non-converged fit is `rep_1` at the excluded `1.28e-3`
rung (see below). Cross-validation converged as well.

> **Correction to the `n_processes` sizing note.** The previous entry recorded
> ~2.5 GB RSS per worker; that was sampled too early. Steady-state RSS here is
> **~35–38 GB per worker**, so six workers need ~230 GB. Fine on a 1.5 TB orca
> (peak system usage 263 GB of 1511 GB), but this is why `n_processes` must
> never be restored to `null` — auto-sizing by core count would pick ~64
> workers and need multiple TB.

**The `1.28e-3` rung is excluded from analysis under gate G5, because it is
unstable rather than slow.** `rep_1` reaches `objective_error` 2.7e-06 by sweep
100, then *diverges*: the error climbs back to ~1.8e-04 and flattens (tail rate
~1.000 over the last 100 sweeps), and the objective ends 10% above its sweep-109
minimum — the only `drift_frac` above the 0.05 threshold anywhere on the ladder.
Raising `maxiter` would not fix this. The earlier tol 1e-5 run stopped at sweep
94, *before* the divergence began, and so reported this rung as converged: that
success was an artifact of the stopping point. Excluding the top rung is exactly
what G5 provides for, and it was never a candidate λ. Across the remaining 18
fits max `drift_frac` is 0.0024.

**Stop-codon sparsity answers the question the rung was added for**: it
saturates at 1.000 by `3.2e-4` and is flat thereafter, so it is *not* still
climbing past `6.4e-4`. Sparsity is monotone non-decreasing in λ for every
`(dataset_name, mut_type, mut_param)` group.

**λ is unchanged at `8.0e-05`, and is robust to the tolerance.** Tightening
from 1e-5 to 1e-6 moved sparsity and replicate correlation by <0.05 everywhere
on the analysed ladder (median |Δ| 0.001 and 0.0006 respectively). The one
substantive change is that validation loss now *minimizes* at 8.0e-05 instead
of 4.0e-05, so all three manuscript criteria agree on the chosen rung — though
the two rungs sit within 0.16% of each other, so that agreement is
corroboration rather than a decisive independent vote.

### 2026-07-28 — prod, issue #287 (`beta0_ridge`/`l2reg` promotion + maxiter split)

| | |
|---|---|
| Host | orca05 (64 cores, 1.5 TB RAM) |
| Branch / commit | `287-config-tier-split` @ `9c4b515` |
| Output dir | `results-prod-287-config-tier-split` |
| Wall-clock | **3643 s (61 min)** |
| `fit_collection.pkl` | 1,584,437,445 bytes (1.58 GB), 18 fitted rows |
| Workers | `n_processes: 6`, ~7 GB RSS each at steady state |

**Parameter set** (verified on the fitted rows, not just the YAML):
`beta0_ridge=0.01`, `l2reg=1e-6`, `maxiter=200` (outer) / `10` (inner
`ge_kwargs`, `cal_kwargs`), fusionreg = the 9-value manuscript ladder.

> These are not the manuscript's ridge weights (manuscript: β ridge 1e-7,
> α ridge 1e-3). Note `beta0_ridge` is **shift shrinkage** — it penalizes
> `(β0_d − β0_ref)²`, the intercept **differences**, not the intercept
> magnitudes.

**Shift replicate correlation vs the pre-change baseline** (delta = new − baseline):

| fusionreg | Δ `shift_Delta` | Δ `shift_Omicron_BA2` |
|---|---|---|
| 0.0 | −0.0149 | +0.0025 |
| 5e-6 | −0.0087 | +0.0123 |
| 1e-5 | −0.0107 | +0.0155 |
| 2e-5 | −0.0013 | +0.0091 |
| **4e-5** (chosen λ) | **+0.0083** | **+0.0105** |
| 8e-5 | +0.0093 | +0.0111 |
| 1.6e-4 | +0.0108 | +0.0054 |
| 3.2e-4 | +0.0142 | +0.0213 |
| 6.4e-4 | **+0.3269** | +0.0248 |

At the three weakest `fusionreg` values `shift_Delta` is slightly below
baseline (−0.015 to −0.009); at the chosen λ=4e-5 both parameters improve.
