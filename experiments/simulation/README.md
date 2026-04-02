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
    ▼
fit_models ──► fit_collection.pkl
    │
    ▼
evaluate ──► model_vs_truth_beta_shift.csv
          ──► collection_muts.csv
          ──► fit_sparsity.csv
          ──► library_replicate_correlation.csv
          ──► model_vs_truth_variant_phenotype.csv
          ──► cross_validation_loss.csv
    │
    ├────────────────────┐
    ▼                    ▼
visualize          manuscript_figures
(inline only)      ──► figures/main_figure.pdf
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

- **Production**: `config/config.yaml` — manuscript defaults (genelength=50, 4 fusionreg values, maxiter=100)
- **Test**: `config/config_test.yaml` — reduced parameters (genelength=10, 2 fusionreg values, maxiter=20)
