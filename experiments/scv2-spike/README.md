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
| `config.yaml` | Production profile — manuscript defaults from `profile_beta0_zero_l2reg_10x` |
| `config_test.yaml` | Test profile — 10% subsample, 2 fusionreg values, 20 iterations |

### Source notebooks (in `notebooks/`)

| Notebook | Description |
|----------|-------------|
| `prepare_data.ipynb` | Download raw data, count aggregation, filtering, functional score computation, replicate correlations |
| `fit_models.ipynb` | Fit multidms models across fusionreg grid per replicate |
| `cross_validation.ipynb` | 80/20 train/test CV across regularization grid |
| `evaluate.ipynb` | Convergence diagnostics, sparsity, replicate correlations, GE plots, mutation export |
