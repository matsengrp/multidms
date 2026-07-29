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

- **Production**: `config/config.yaml` — manuscript defaults (genelength=50, 9 fusionreg values, maxiter=100)
- **Test**: `config/config_test.yaml` — reduced parameters (genelength=10, 2 fusionreg values, maxiter=20)

Each has a matching `<name>_downstream.yaml` sibling.

### Configuration tiers

The config is split by dependency tier so that a downstream-only edit cannot
invalidate the expensive model fit:

| File | Holds | Do edits invalidate the fit? |
|------|-------|------------------------------|
| `config.yaml` | `seed`, `train_frac`, the simulation parameters, the `fitting:` block, `output_dir` | **Yes** |
| `config_downstream.yaml` | `lasso_choice` (only) | **No** |

Snakemake reruns a job when an `input:` file's mtime changes, not when its
content changes. Before the split, every rule declared the single config as an
input, so editing `lasso_choice` — a key the fit never reads — forced a full
refit that recomputed a byte-identical `fit_collection.pkl`.

Now only `rule evaluate` and `rule manuscript_figures` declare
`config_downstream.yaml` as an `input:`. `simulate_data`, `fit_models`, and
`cross_validation` do not reference it at all.

> Do not "tidy" this by adding `downstream_config` to the `input:` block of
> `simulate_data`, `fit_models`, or `cross_validation`. That would silently
> restore the defect. See issue #287.

Unlike spike, simulation's downstream tier holds `lasso_choice` alone — this
pipeline has no `condition_colors`, `condition_titles`, or `domain_dict`.
