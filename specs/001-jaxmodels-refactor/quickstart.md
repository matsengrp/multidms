# Quickstart Guide: multidms v2.0 (jaxmodels)

**Feature**: 001-jaxmodels-refactor
**Date**: 2025-11-10
**Audience**: Developers implementing the refactoring

## Overview

This quickstart demonstrates the core workflow for the refactored multidms v2.0 using the jaxmodels backend. The example shows the minimal path from data loading to model fitting and visualization.

## Prerequisites

```bash
# Install multidms v2.0 with dependencies
pip install multidms>=2.0.0

# Or install from source for development
git checkout 001-jaxmodels-refactor
pip install -e ".[dev]"
```

**Required packages** (auto-installed with multidms):
- JAX ≥0.4.29
- equinox
- jaxopt
- pandas ≥2.2.0
- numpy ≤1.26.0

## Basic Workflow

### Step 1: Prepare Data

```python
import pandas as pd
from multidms import Data

# Load your DMS experimental data
# Required columns: condition, aa_substitutions, func_score
df = pd.DataFrame({
    'condition': ['reference', 'reference', 'variant_A', 'variant_A'],
    'aa_substitutions': ['', 'M1A', '', 'M1A'],
    'func_score': [0.0, 1.2, 0.1, 1.5]
})

# Optional: Add count data for count-based loss
df['pre_counts'] = [1000, 500, 1000, 450]
df['post_counts'] = [1000, 600, 1100, 700]

# Optional: Collapse identical variants before creating Data object
# (Data class does NOT aggregate automatically - users control this)
# df = df.groupby(['condition', 'aa_substitutions'], as_index=False).agg({
#     'func_score': 'mean',
#     'pre_counts': 'sum',
#     'post_counts': 'sum'
# })

# Create Data object (validates and preprocesses)
data = Data(
    df,
    reference='reference'  # Optional: defaults to first condition
)

# Inspect processed data
print(f"Variants: {len(data.variants_df)}")
print(f"Mutations: {len(data.mutations_df)}")
print(f"Conditions: {data.variants_df.condition.nunique()}")
```

**Output**:
```
Variants: 4
Mutations: 1
Conditions: 2
```

### Step 2: Initialize and Fit Model

```python
from multidms import Model

# Initialize model with desired configuration
model = Model(
    data,
    loss_type='functional_score_loss',  # or 'count_loss'
    ge_type='Sigmoid',                   # or 'Identity'
    l2reg=0.01,                          # L2 regularization
    fusionreg=0.1                        # Fusion regularization
)

# Fit the model
model.fit(
    warmstart=True,      # Use Ridge initialization
    maxiter=1000,        # Maximum iterations
    tol=1e-6            # Convergence tolerance
)

# Check convergence
print(f"Final loss: {model.convergence_trajectory_df.loss.iloc[-1]:.6f}")
print(f"Converged: {model.convergence_trajectory_df.error.iloc[-1] < 1e-6}")
```

**Output**:
```
Final loss: 0.045231
Converged: True
```

### Step 3: Extract Parameters

```python
# Get mutation-level parameters
mutations_df = model.get_mutations_df(phenotype_as_effect=True)
print(mutations_df)
```

**Output**:
```
  mutation  condition  beta  shift  predicted_func_score
0      M1A  reference  1.20   0.00                  1.18
1      M1A  variant_A  1.20   0.35                  1.53
```

```python
# Get variant-level predictions
variants_df = model.get_variants_df()
print(variants_df[['condition', 'aa_substitutions', 'latent_phenotype', 'predicted_func_score']])
```

**Output**:
```
   condition aa_substitutions  latent_phenotype  predicted_func_score
0  reference                                0.00                  0.00
1  reference              M1A                1.20                  1.18
2  variant_A                                0.00                  0.10
3  variant_A              M1A                1.55                  1.52
```

### Step 4: Visualize Results

```python
import matplotlib.pyplot as plt

# Plot global epistasis curve
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Epistasis transformation
model.plot_epistasis(condition='reference', ax=axes[0])
axes[0].set_title('Global Epistasis: Reference')

# Prediction accuracy
model.plot_pred_accuracy(condition='variant_A', ax=axes[1])
axes[1].set_title('Prediction Accuracy: Variant A')

plt.tight_layout()
plt.savefig('model_diagnostics.png')
```

```python
# Interactive heatmap (Altair)
heatmap = model.mut_param_heatmap(
    parameter='beta',
    conditions=['reference', 'variant_A']
)
heatmap.save('mutation_effects.html')
```

### Step 5: Predict on New Data

```python
# Create new variants for prediction
new_variants = pd.DataFrame({
    'condition': ['reference', 'variant_A'],
    'aa_substitutions': ['M1A', 'M1A']  # Must use mutations seen in training
})

# Add predictions
predictions = model.add_phenotypes_to_df(new_variants)
print(predictions)
```

**Output**:
```
   condition aa_substitutions  latent_phenotype  predicted_func_score
0  reference              M1A              1.20                  1.18
1  variant_A              M1A              1.55                  1.52
```

## Advanced: Multiple Models (ModelCollection)

```python
from multidms import fit_models, ModelCollection

# Define hyperparameter grid
param_grid = pd.DataFrame({
    'l2reg': [0.001, 0.01, 0.1],
    'fusionreg': [0.0, 0.1, 0.5]
})

# Fit multiple models in parallel
models_df = fit_models(
    data_or_df_dict={'dataset1': data},
    model_params=param_grid,
    n_jobs=4  # Parallel processes
)

# Create collection for analysis
mc = ModelCollection(models_df)

# Aggregate mutation parameters across models
aggregated = mc.split_apply_combine_muts(
    groupby_cols=['mutation'],
    agg_func='mean'
)
print(aggregated)
```

**Output**:
```
  mutation  beta_mean  shift_mean
0      M1A       1.18        0.32
```

## Common Patterns

### Pattern 1: Aggregating Barcode Replicates

```python
# If you have multiple barcodes/replicates per variant, aggregate before Data creation
# Data class does NOT do this automatically

# Mean aggregation
df_collapsed = df.groupby(['condition', 'aa_substitutions'], as_index=False).agg({
    'func_score': 'mean',
    'pre_counts': 'sum',   # Sum counts across replicates
    'post_counts': 'sum'
})

data = Data(df_collapsed)

# Alternatively: Keep replicates separate and fit to barcode-level data
# (no aggregation)
data_barcodes = Data(df)  # Keeps all replicates
model = Model(data_barcodes)
model.fit()  # Fits treating each barcode as separate observation
```

### Pattern 2: Count-Based Loss

```python
# When you have count data and want to model enrichment
model = Model(
    data,  # data must have pre_counts, post_counts columns
    loss_type='count_loss',
    ge_type='Identity'
)
model.fit()
```

### Pattern 3: Identity Global Epistasis

```python
# For linear additive models (no non-linear transformation)
model = Model(
    data,
    ge_type='Identity'  # g(φ) = φ (no transformation)
)
model.fit()
```

### Pattern 4: Manual Parameter Initialization

```python
import numpy as np

# Initialize with specific parameter values
n_mutations = len(data.mutations_df)
n_conditions = data.variants_df.condition.nunique()

model = Model(
    data,
    beta_init=np.random.randn(n_mutations) * 0.1,
    beta0_init=np.zeros(n_conditions)
)
model.fit(warmstart=False)  # Skip Ridge warmstart
```

### Pattern 5: Cross-Validation

```python
from sklearn.model_selection import KFold

# Split data for cross-validation
kf = KFold(n_splits=5)
fold_scores = []

for train_idx, test_idx in kf.split(df):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    train_data = Data(train_df)
    model = Model(train_data, l2reg=0.01)
    model.fit()

    # Evaluate on test set
    test_preds = model.add_phenotypes_to_df(test_df)
    mse = ((test_preds.predicted_func_score - test_df.func_score) ** 2).mean()
    fold_scores.append(mse)

print(f"Mean CV MSE: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
```

## Error Handling Examples

### Handling Invalid Data

```python
# Example: Data with NaN functional scores
bad_df = df.copy()
bad_df.loc[0, 'func_score'] = float('nan')

try:
    data = Data(bad_df)
except ValueError as e:
    print(f"Error: {e}")
    # Output: Found 1 rows with invalid func_score...
    # Clean data before retrying
    clean_df = bad_df.dropna(subset=['func_score'])
    data = Data(clean_df)
```

### Handling Unseen Mutations

```python
# Try to predict on mutations not in training data
new_variants = pd.DataFrame({
    'condition': ['reference'],
    'aa_substitutions': ['X999Z']  # Not in training data!
})

try:
    predictions = model.add_phenotypes_to_df(new_variants)
except ValueError as e:
    print(f"Error: {e}")
    # Output: Cannot predict on unseen mutations: ['X999Z']...
    # Solution: Retrain model with expanded dataset
```

### Handling Count Loss Without Counts

```python
# Example: Requesting count_loss without count data
try:
    model = Model(data, loss_type='count_loss')
except ValueError as e:
    print(f"Error: {e}")
    # Output: Count data required when loss_type='count_loss'...
    # Solution: Add count columns to DataFrame or use functional_score_loss
```

## Migration from v1.x

### Compatible Code (No Changes Needed)

```python
# This code works identically in v1.x and v2.0
from multidms import Data, Model

data = Data(df, reference='ref')
model = Model(data, ge_type='Sigmoid', l2reg=0.01)
model.fit(warmstart=True)
mutations_df = model.get_mutations_df()
```

### Deprecated Code (Requires Changes)

```python
# v1.x: Direct import from biophysical
from multidms.biophysical import some_function  # ❌ Raises ImportError in v2.0

# v2.0: Use Model methods instead
# Most biophysical functions are now internal to jaxmodels
# If you need specific biophysical calculations, use Model class methods
```

### Removed Feature (Manual Preprocessing Required)

```python
# v1.x: Data class aggregated identical variants automatically
data = Data(df, collapse_identical_variants='mean')  # ❌ Parameter removed in v2.0

# v2.0: Aggregate variants yourself before creating Data object
df_collapsed = df.groupby(['condition', 'aa_substitutions'], as_index=False).agg({
    'func_score': 'mean',
    'pre_counts': 'sum',
    'post_counts': 'sum'
})
data = Data(df_collapsed)  # ✓ User handles aggregation
```

### Validation: Comparing v1.x and v2.0

```python
# Run same analysis in both versions
# v1.x results
# model_v1.get_mutations_df().to_csv('v1_results.csv')

# v2.0 results
model_v2 = Model(data, ge_type='Sigmoid', l2reg=0.01)
model_v2.fit()
results_v2 = model_v2.get_mutations_df()

# Compare
# diff = pd.read_csv('v1_results.csv').merge(results_v2, on=['mutation', 'condition'], suffixes=('_v1', '_v2'))
# print(f"Max beta difference: {(diff.beta_v2 - diff.beta_v1).abs().max():.6f}")
# Expect: <0.01 difference due to numerical precision
```

## Performance Benchmarking

```python
import time

# Benchmark fitting time
start = time.time()
model.fit(maxiter=1000)
elapsed = time.time() - start

print(f"Fitting time: {elapsed:.2f} seconds")
print(f"Iterations: {len(model.convergence_trajectory_df)}")
print(f"Time per iteration: {elapsed / len(model.convergence_trajectory_df) * 1000:.2f} ms")

# Expected for 1000 variants, 3 conditions:
# Fitting time: 5-15 seconds (depending on hardware)
# v2.0 should be within 2x of v1.x time
```

## Debugging Tips

### Check Model State

```python
# Is model fitted?
print(f"Model fitted: {model.params is not None}")

# Inspect convergence
if model.params is not None:
    traj = model.convergence_trajectory_df
    print(f"Final loss: {traj.loss.iloc[-1]:.6f}")
    print(f"Final error: {traj.error.iloc[-1]:.6e}")

    # Plot convergence
    import matplotlib.pyplot as plt
    plt.plot(traj.iteration, traj.loss)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Convergence Trajectory')
    plt.show()
```

### Inspect Data Processing

```python
# Check one-hot encoding
print("One-hot encoded mutations:")
print(data.variants_df.filter(regex='mut_').head())

# Check site map
print(f"\nSite map: {data.site_map}")

# Check mutations catalog
print(f"\nMutations:\n{data.mutations_df}")

# NEW in v2.0: Access properties used by jaxmodels
print(f"\nBinary map shape: {data.binary_map.shape}")
print(f"Targets per condition: {list(data.targets.keys())}")
print(f"Condition indices: {data.condition_indices[:5]}")  # First 5
print(f"Sample weights available: {data.weights is not None}")
```

### JAX-Specific Debugging

```python
# Enable JAX debugging (shows full tracebacks)
import jax
jax.config.update('jax_debug_nans', True)
jax.config.update('jax_disable_jit', True)  # Disable JIT for debugging

# Fit model (will show detailed errors if issues occur)
model.fit()
```

## Next Steps

1. **Read the full documentation**: [https://multidms.readthedocs.io](https://multidms.readthedocs.io)
2. **Explore example notebooks**: See `notebooks/` directory for complete workflows
3. **Review API reference**: Full class and method documentation
4. **Migration guide**: Detailed v1.x → v2.0 migration path

## Summary

**Core workflow**:
```python
from multidms import Data, Model

# 1. Load and preprocess data
data = Data(df, reference='ref')

# 2. Initialize and fit model
model = Model(data, ge_type='Sigmoid', l2reg=0.01)
model.fit(warmstart=True)

# 3. Extract results
mutations_df = model.get_mutations_df()
variants_df = model.get_variants_df()

# 4. Visualize
model.plot_epistasis()
model.plot_pred_accuracy()

# 5. Predict on new data
predictions = model.add_phenotypes_to_df(new_df)
```

**Key differences from v1.x**:
- Internal backend changed to jaxmodels (JAX-based optimization)
- `biophysical.py` module deprecated (raises ImportError)
- `jaxmodels.Data` class removed - `Data` provides binary_map, targets directly
- Warmstart now works without count data
- `collapse_identical_variants` parameter removed from Data class (users handle aggregation)
- Performance within 2x of v1.x

**Testing your implementation**:
```bash
# Run full test suite
pytest --doctest-modules -vv

# Check code coverage
pytest --cov=multidms --cov-report=html

# Validate formatting
black . --check
ruff check .
```

---

**Quickstart Complete**: 2025-11-10
**Status**: Ready for implementation
