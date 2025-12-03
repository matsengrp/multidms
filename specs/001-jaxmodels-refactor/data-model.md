# Data Model: JAX Models Refactoring

**Feature**: 001-jaxmodels-refactor
**Date**: 2025-11-10
**Status**: Partially Outdated - See Update Below
**Last Updated**: 2025-01-28

## Update (2025-01-28)

**Architecture Decision**: `jaxmodels.Data` class is KEPT as internal implementation detail, contrary to original planning document.

**Actual Implementation**:
- `multidms.Data` - User-facing class, provides per-condition data via `arrays["X"]` and `arrays["y"]` dicts (unchanged)
- `jaxmodels.Data` - Internal class for type-safe jaxmodels interface (kept, not removed)
- `multidms.Model` - User-facing wrapper that converts Data → jaxmodels.Data per condition internally
- No new properties added to `multidms.Data` class

See `multidms/model.py` for actual implementation.

## Overview

This document defines the data model for the jaxmodels refactoring, showing the relationships between user-facing classes (`Data`, `Model`, `ModelCollection`) and the internal jaxmodels backend. The model preserves the existing API while delegating computation to jaxmodels.

**NOTE**: Some details below are outdated. Refer to actual implementation in `multidms/model.py` and `multidms/data.py`.

## Core Entities

### 1. Data (User-Facing)

**Purpose**: Represents processed DMS experimental data, handling DataFrame preprocessing and conversion to jaxmodels format.

**Attributes**:
| Attribute | Type | Description | Source |
|-----------|------|-------------|--------|
| `data` | `pd.DataFrame` | Original DataFrame with condition, aa_substitutions, func_score columns | User input |
| `variants_df` | `pd.DataFrame` | Per-variant data with one-hot encodings | Computed from `data` |
| `mutations_df` | `pd.DataFrame` | Per-mutation catalog across all conditions | Computed from `data` |
| `site_map` | `dict` | Mapping of sites to mutation indices | Computed during encoding |
| `non_identical_sites` | `list` | Sites where conditions have different wildtype amino acids | Computed from data |
| `reference` | `str` | Name of reference condition | User input or inferred |
| `alphabet` | `list` | Allowed amino acid alphabet | User input or default |

**Properties for jaxmodels** (new in v2.0):
| Property | Type | Description | Purpose |
|----------|------|-------------|---------|
| `binary_map` | `np.ndarray` | One-hot encoded mutation matrix (n_variants × n_mutations) | Provides X for jaxmodels |
| `targets` | `dict[str, np.ndarray]` | Target values per condition (func_scores or counts) | Provides y for jaxmodels |
| `condition_indices` | `np.ndarray` | Integer condition index per variant | Maps variants to conditions |
| `weights` | `np.ndarray` or `None` | Sample weights from counts (if available) | Optional weights for jaxmodels |

**Methods**:
| Method | Inputs | Outputs | Purpose |
|--------|--------|---------|---------|
| `__init__(...)` | DataFrame, reference, alphabet | Data object | Validate and preprocess data |

**Validation Rules**:
- Required columns: `condition`, `aa_substitutions`, `func_score`
- Optional columns: `pre_counts`, `post_counts`, `pre_count_wt`, `post_count_wt`
- Functional scores must not contain NaN or infinite values (FR-001a)
- Count data required if provided for count_loss (FR-044a)
- Valid substitution format: "A123B" or comma-separated "A123B,C456D"

**State Transitions**:
```
[Raw DataFrame (user pre-aggregated if desired)]
    → validate columns & values
    → one-hot encode substitutions
    → [Data object ready for Model]
```

**Note**: Users are responsible for aggregating/collapsing identical variants or barcodes before creating Data object. Data class accepts data as-is.

**Breaking Change from v1.x**: Data class now directly provides properties (`binary_map`, `targets`, `condition_indices`, `weights`) that jaxmodels needs. No intermediate `jaxmodels.Data` class exists in v2.0.

---

### 2. Model (User-Facing)

**Purpose**: Wrapper for jaxmodels fitting and prediction, maintaining backward-compatible API.

**Attributes**:
| Attribute | Type | Description | Source |
|-----------|------|-------------|--------|
| `data` | `Data` | Preprocessed DMS data | User provides Data object |
| `_jaxmodel` | `jaxmodels.Model` | Internal JAX model instance | Created during `__init__` |
| `params` | PyTree | Current model parameters (beta, beta0, shift, theta) | From `_jaxmodel` after fitting |
| `convergence_trajectory_df` | `pd.DataFrame` | Loss and error over iterations | Populated during `fit()` |
| `loss_type` | `str` | "functional_score_loss" or "count_loss" | User input |
| `ge_type` | `str` | "Identity" or "Sigmoid" | User input |
| `l2reg` | `float` | L2 regularization strength | User input |
| `fusionreg` | `float` | Fusion regularization strength | User input |

**Methods**:
| Method | Inputs | Outputs | Purpose |
|--------|--------|---------|---------|
| `__init__(...)` | Data, loss_type, ge_type, regularization params | Model object | Initialize jaxmodels.Model |
| `fit(...)` | Optimizer params, warmstart options | Self | Run JAX optimization loop |
| `get_mutations_df()` | phenotype_as_effect | DataFrame | Extract mutation parameters (beta, shift) |
| `get_variants_df()` | phenotype_as_effect | DataFrame | Extract variant predictions |
| `add_phenotypes_to_df(...)` | DataFrame | DataFrame | Predict on new data |
| `plot_epistasis(...)` | Condition, ax | Figure/ax | Visualize global epistasis curve |
| `plot_pred_accuracy(...)` | Condition, ax | Figure/ax | Observed vs predicted scatter |
| `mut_param_heatmap(...)` | Parameter, conditions | Altair chart | Interactive heatmap |
| `plot_param_hist(...)` | Parameter, ax | Figure/ax | Parameter distribution histogram |

**Validation Rules**:
- Count data required if `loss_type="count_loss"` (FR-044a)
- Warmstart parameters must match data dimensions (FR-048)
- Valid ge_type values: "Identity", "Sigmoid" (FR-013)
- Valid loss_type values defined in jaxmodels (FR-014)

**State Transitions**:
```
[Unfit Model]
    → fit() called
    → JAX optimization loop
    → parameters updated
    → [Fitted Model ready for prediction/visualization]
```

---

### 3. jaxmodels.Model (Internal Backend)

**Purpose**: JAX/equinox model implementing global epistasis equations and optimization.

**Attributes**:
| Attribute | Type | Description |
|-----------|------|-------------|
| `latent` | `jaxmodels.Latent` | Computes additive mutation effects (φ) |
| `ge` | `jaxmodels.GlobalEpistasis` | Non-linear transformation g(φ) |
| `loss` | `jaxmodels.Loss` | Loss function (Huber or NegBinomial) |
| `params` | PyTree | Current parameter state (beta, beta0, shift, theta) |

**Methods**:
| Method | Inputs | Outputs | Purpose |
|--------|--------|---------|---------|
| `fit(...)` | Data properties (binary_map, targets, etc.), optimizer, init | Updated params | Run optimization |
| `predict(...)` | Data properties, params | Predictions | Generate predictions |

**Note**: This is defined in `jaxmodels.py`. User-facing `Model` delegates computational work to this class. In v2.0, jaxmodels.Model directly accesses Data class properties instead of receiving a separate jaxmodels.Data object.

---

### 4. ModelCollection (User-Facing)

**Purpose**: Container for multiple fitted models enabling comparison and aggregation.

**Attributes**:
| Attribute | Type | Description | Source |
|-----------|------|-------------|--------|
| `models_df` | `pd.DataFrame` | DataFrame where each row contains a fitted Model object plus hyperparameters | Created via `fit_models()` |

**Methods**:
| Method | Inputs | Outputs | Purpose |
|--------|--------|---------|---------|
| `__init__(...)` | DataFrame with model column | ModelCollection object | Validate and store models |
| `split_apply_combine_muts()` | Groupby columns, agg function | DataFrame | Aggregate mutation parameters across models |
| `mut_param_dataset_correlation(...)` | Parameter, replicate col | Figure | Correlation plot between replicates |
| `shift_sparsity(...)` | None | Figure | Visualize shift parameter sparsity |
| `mut_param_heatmap(...)` | Parameter, filters | Altair chart | Aggregated parameter heatmap |
| `mut_param_traceplot(...)` | Mutations, parameter | Altair chart | Track mutations across fits |

**Validation Rules**:
- `models_df` must contain a column with Model objects
- All models must be fitted (have parameters)
- Models should share compatible data structure for aggregation

---

## Entity Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                         User-Facing API                          │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │        Data          │  ← User provides DataFrame
    │                      │  ← Provides binary_map, targets, etc.
    └──────┬───────────────┘
           │ 1
           │ has
           │
           ▼ 1
    ┌──────────────────────┐
    │       Model          │  ← Wraps jaxmodels.Model internally
    │                      │  ← Accesses Data properties directly
    └──────┬───────────────┘
           │ *
           │ collected in
           │
           ▼ 1
    ┌──────────────────────┐
    │  ModelCollection     │  ← Aggregates multiple models
    └──────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Internal (jaxmodels)                        │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────────┐
    │ jaxmodels.Model  │  ← JAX/equinox model
    │                  │  ← Receives Data properties directly
    │  ┌────────────┐  │
    │  │   Latent   │  │  ← Additive effects φ = Σ beta
    │  └────────────┘  │
    │  ┌────────────┐  │
    │  │ GlobalEpi  │  │  ← Non-linear transform g(φ)
    │  └────────────┘  │
    │  ┌────────────┐  │
    │  │    Loss    │  │  ← Huber or NegBinomial
    │  └────────────┘  │
    └──────────────────┘
```

**Data Flow** (v2.0 simplified):
```
DataFrame
    → Data.__init__()
    → Data object (provides binary_map, targets, condition_indices, weights)
    → Model accesses Data properties directly
    → jaxmodels.Model.fit(data.binary_map, data.targets, ...)
    → optimized parameters
    → Model.get_mutations_df()
    → DataFrame (back to pandas)
```

**Key Change from v1.x**: No intermediate `jaxmodels.Data` class. The `multidms.data.Data` class directly provides all arrays that jaxmodels needs.

---

## Parameter Schema

### Global Model Parameters

| Parameter | Shape | Description | Equation Role |
|-----------|-------|-------------|---------------|
| `beta` | (n_mutations,) | Mutation effects in reference condition | φ_ref = Σ beta_i |
| `beta0` | (n_conditions,) | Condition-specific offsets | φ_c = φ_ref + beta0_c |
| `shift` | (n_mutations, n_conditions-1) | Non-reference mutation shifts | φ_c = Σ (beta_i + shift_ic) |
| `theta` | Varies by ge_type | Global epistasis parameters | g(φ; theta) |

**Identity GE**: `theta` is empty (no parameters)
**Sigmoid GE**: `theta = (ge_offset, ge_scale)` per condition

---

## Data Validation & Error Handling

### Input Validation (Data Class)

| Validation | Error Type | Error Message |
|------------|------------|---------------|
| Missing required columns | `ValueError` | "DataFrame missing required columns: {cols}. Expected: condition, aa_substitutions, func_score" |
| NaN/inf functional scores | `ValueError` | "Found {n} rows with invalid func_score at indices: {rows}. Please clean data with dropna() or filtering" |
| Invalid substitution format | `ValueError` | "Invalid substitution format at row {i}: '{sub}'. Expected format: 'A123B' or 'A123B,C456D'" |
| Invalid alphabet | `ValueError` | "Invalid alphabet: {alphabet}. Must be list of valid amino acids" |

### Model Validation (Model Class)

| Validation | Error Type | Error Message |
|------------|------------|---------------|
| Missing count data for count_loss | `ValueError` | "Count data required when loss_type='count_loss'. Please provide pre_counts, post_counts columns" |
| Warmstart shape mismatch | `ValueError` | "Warmstart parameters shape {shape} doesn't match expected {expected_shape} for {n_mutations} mutations and {n_conditions} conditions" |
| Unseen mutations in prediction | `ValueError` | "Cannot predict on unseen mutations: {mutations}. These were not in training data. Please retrain with expanded dataset" |

---

## Performance Considerations

### Memory Usage

| Entity | Memory Scaling | Notes |
|--------|----------------|-------|
| `Data.data` | O(n_variants) | Original DataFrame |
| `Data.variants_df` | O(n_variants × n_mutations) | One-hot encoding (sparse) |
| `jaxmodels.Data.X` | O(n_variants × n_mutations) | JAX sparse array |
| `Model.params.beta` | O(n_mutations) | Dense parameter vector |
| `Model.params.shift` | O(n_mutations × n_conditions) | Dense parameter matrix |

**Memory Limit**: Datasets with millions of variants may exceed available RAM on typical hardware. Users should be warned if memory allocation fails (FR-044b).

### Conversion Overhead

- DataFrame → JAX array conversion: O(n_variants)
- Happens once during `Model.__init__()` (lazy conversion)
- Prediction conversions: O(n_new_variants) per call

---

## Compatibility Matrix

### Version Compatibility

| Component | v1.x (biophysical) | v2.0 (jaxmodels) | Breaking Changes |
|-----------|-------------------|------------------|------------------|
| Data API | ✓ | ✓ | None - fully compatible |
| Model.__init__ | ✓ | ✓ | None - parameter names preserved |
| Model.fit() | ✓ | ✓ | Internal optimizer changed, API same |
| Model.get_mutations_df() | ✓ | ✓ | None - output format identical |
| Model.plot_*() | ✓ | ✓ | None - visualization API preserved |
| ModelCollection | ✓ | ✓ | None - aggregation methods unchanged |
| biophysical.py | ✓ | ✗ ImportError | BREAKING: Module deprecated |

### Migration Path

**Code Changes Required**:
- Remove any direct imports of `multidms.biophysical`
- Update to `multidms` v2.0.0 in requirements
- Re-run existing workflows (results should be nearly identical)

**No Code Changes Required**:
- Core API (`Data`, `Model`, `ModelCollection`) remains compatible
- Notebook workflows should run with minimal modifications
- Hyperparameter names and ranges unchanged

---

**Data Model Complete**: 2025-11-10
**Status**: Ready for contract generation
