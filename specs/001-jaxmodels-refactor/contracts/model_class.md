# Contract: Model Class

**Class**: `multidms.Model`
**Purpose**: Fit global epistasis models to DMS data using jaxmodels backend
**Module**: `multidms/model.py`

## Constructor

### `Model.__init__(data, *, loss_type="functional_score_loss", ge_type="Identity", l2reg=0.0, fusionreg=0.0, beta0_ridge=1.0, ...)`

**Purpose**: Initialize a Model for fitting to DMS data.

**Parameters**:
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `data` | `Data` | Yes | - | Preprocessed DMS data |
| `loss_type` | `str` | No | `"functional_score_loss"` | Loss function: "functional_score_loss" or "count_loss" |
| `ge_type` | `str` | No | `"Identity"` | Global epistasis: "Identity" or "Sigmoid" |
| `l2reg` | `float` | No | `0.0` | L2 regularization strength |
| `fusionreg` | `float` | No | `0.0` | Fusion regularization strength |
| `beta0_ridge` | `float` | No | `1.0` | Ridge parameter for beta0 warmstart |
| `beta_init` | `np.ndarray` | No | `None` | Manual beta initialization |
| `beta0_init` | `np.ndarray` | No | `None` | Manual beta0 initialization |
| `alpha_init` | `np.ndarray` | No | `None` | Manual GE parameter initialization |

**Returns**: `Model` object (unfitted)

**Raises**:
- `ValueError`: If `loss_type` not in {"functional_score_loss", "count_loss"}
- `ValueError`: If `ge_type` not in {"Identity", "Sigmoid"}
- `ValueError`: If `loss_type="count_loss"` but count data missing from `data`
- `ValueError`: If manual init parameters have wrong shape

**Example**:
```python
from multidms import Data, Model

data = Data(df)
model = Model(
    data,
    loss_type="functional_score_loss",
    ge_type="Sigmoid",
    l2reg=0.01,
    fusionreg=0.1
)
```

**Post-conditions**:
- `model.data` references the provided Data object
- `model._jaxmodel` is initialized (internal jaxmodels.Model instance)
- `model.params` is None (not yet fitted)

---

## Fitting Method

### `model.fit(*, warmstart=True, maxiter=1000, tol=1e-6, learning_rate=0.01, ...)`

**Purpose**: Fit the model parameters using JAX optimization.

**Parameters**:
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `warmstart` | `bool` | No | `True` | Use Ridge regression for initialization |
| `maxiter` | `int` | No | `1000` | Maximum optimization iterations |
| `tol` | `float` | No | `1e-6` | Convergence tolerance |
| `learning_rate` | `float` | No | `0.01` | Optimizer learning rate |

**Returns**: `self` (fitted Model)

**Raises**:
- `RuntimeError`: If JAX optimization fails with numerical error
- `Warning`: If convergence not achieved within maxiter

**Example**:
```python
model.fit(warmstart=True, maxiter=2000, tol=1e-7)
print(f"Final loss: {model.convergence_trajectory_df.loss.iloc[-1]}")
```

**Post-conditions**:
- `model.params` contains fitted parameters (beta, beta0, shift, theta)
- `model.convergence_trajectory_df` contains loss trajectory
- Model state changes from unfitted → fitted

**Side Effects**:
- Modifies `model.params` in place
- Populates `model.convergence_trajectory_df`
- May print convergence warnings to stdout

**Convergence**:
- Success: `|loss[t] - loss[t-1]| < tol`
- Failure: Maxiter reached without convergence → Warning issued
- Warning message: "Model did not converge. Final error: {err:.6f}, tolerance: {tol:.6f}. Consider increasing maxiter or adjusting hyperparameters."

---

## Prediction Methods

### `model.get_mutations_df(*, phenotype_as_effect=False)`

**Purpose**: Extract mutation-level parameters and predictions.

**Parameters**:
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `phenotype_as_effect` | `bool` | No | `False` | Report predictions as difference from wildtype |

**Returns**: `pd.DataFrame` with mutation parameters

**Raises**:
- `RuntimeError`: If model not fitted (params is None)

**Output Columns**:
- `mutation`: Mutation string (e.g., "A123B")
- `condition`: Condition name
- `beta`: Mutation effect in reference condition
- `shift`: Shift from reference (non-reference conditions only)
- `predicted_func_score`: Predicted functional score for this mutation

**Example**:
```python
muts_df = model.get_mutations_df(phenotype_as_effect=True)
print(muts_df.head())
#   mutation condition  beta  shift  predicted_func_score
# 0   A123B        Ref  0.50   0.00                  0.50
# 1   A123B     Cond_B  0.50   0.20                  0.70
```

---

### `model.get_variants_df(*, phenotype_as_effect=False)`

**Purpose**: Extract variant-level predictions.

**Parameters**:
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `phenotype_as_effect` | `bool` | No | `False` | Report phenotypes relative to wildtype |

**Returns**: `pd.DataFrame` with variant predictions

**Raises**:
- `RuntimeError`: If model not fitted

**Output Columns**:
- `condition`: Condition name
- `aa_substitutions`: Substitution string
- `latent_phenotype`: Additive phenotype φ before global epistasis
- `predicted_func_score`: g(φ) after global epistasis transformation
- `observed_func_score`: Original functional score from data

**Example**:
```python
vars_df = model.get_variants_df()
print(vars_df.head())
#   condition aa_substitutions  latent_phenotype  predicted_func_score  observed_func_score
# 0       Ref           A123B              0.50                  0.48                     0.45
```

---

### `model.add_phenotypes_to_df(df, *, phenotype_as_effect=False)`

**Purpose**: Make predictions on new variants not in training data.

**Parameters**:
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `df` | `pd.DataFrame` | Yes | - | DataFrame with condition, aa_substitutions columns |
| `phenotype_as_effect` | `bool` | No | `False` | Report as effects |

**Returns**: `pd.DataFrame` (input df with added prediction columns)

**Raises**:
- `RuntimeError`: If model not fitted
- `ValueError`: If df missing required columns (condition, aa_substitutions)
- `ValueError`: If df contains mutations not seen during training

**Added Columns**:
- `latent_phenotype`: Predicted latent phenotype
- `predicted_func_score`: Predicted functional score

**Example**:
```python
new_variants = pd.DataFrame({
    'condition': ['Ref', 'Cond_B'],
    'aa_substitutions': ['A123B', 'C456D']
})
predictions = model.add_phenotypes_to_df(new_variants)
```

**Validation**:
- All mutations in `df.aa_substitutions` must exist in `model.data.mutations_df`
- If unseen mutations found: "ValueError: Cannot predict on unseen mutations: {mutations}. These were not in training data. Please retrain with expanded dataset."

---

## Visualization Methods

### `model.plot_epistasis(*, condition=None, ax=None)`

**Purpose**: Plot global epistasis transformation curve g(φ).

**Parameters**:
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `condition` | `str` | No | Reference | Condition to plot |
| `ax` | `matplotlib.axes.Axes` | No | `None` | Axes to plot on (creates new if None) |

**Returns**: `matplotlib.figure.Figure` or `matplotlib.axes.Axes`

**Raises**:
- `RuntimeError`: If model not fitted
- `ValueError`: If condition not in data

**Plot Content**:
- X-axis: Latent phenotype φ
- Y-axis: Functional score g(φ)
- Curve shows non-linear transformation

**Example**:
```python
fig = model.plot_epistasis(condition='Cond_A')
fig.savefig('epistasis.png')
```

---

### `model.plot_pred_accuracy(*, condition=None, ax=None)`

**Purpose**: Scatter plot of observed vs. predicted functional scores.

**Parameters**: Same as `plot_epistasis`

**Returns**: `matplotlib.figure.Figure` or `matplotlib.axes.Axes`

**Raises**: Same as `plot_epistasis`

**Plot Content**:
- X-axis: Observed functional scores
- Y-axis: Predicted functional scores
- Diagonal line shows perfect prediction
- Points colored by variant

**Example**:
```python
fig = model.plot_pred_accuracy(condition='Cond_B')
```

---

### `model.mut_param_heatmap(*, parameter='beta', conditions=None, ...)`

**Purpose**: Interactive heatmap of mutation parameters.

**Parameters**:
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `parameter` | `str` | No | `'beta'` | Parameter to visualize: "beta" or "shift" |
| `conditions` | `list[str]` | No | All | Conditions to include |

**Returns**: `altair.Chart` (interactive Altair visualization)

**Raises**:
- `RuntimeError`: If model not fitted
- `ValueError`: If parameter not in {"beta", "shift"}

**Visualization**:
- Rows: Mutations
- Columns: Conditions (or sites)
- Color: Parameter value
- Interactive: Hover to see values, click to filter

**Example**:
```python
chart = model.mut_param_heatmap(parameter='shift', conditions=['A', 'B'])
chart.save('heatmap.html')
```

---

### `model.plot_param_hist(*, parameter='beta', ax=None)`

**Purpose**: Histogram of parameter distribution.

**Parameters**:
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `parameter` | `str` | Yes | - | Parameter name: "beta", "shift", "beta0" |
| `ax` | `matplotlib.axes.Axes` | No | `None` | Axes object |

**Returns**: `matplotlib.figure.Figure` or `matplotlib.axes.Axes`

**Raises**:
- `RuntimeError`: If model not fitted
- `ValueError`: If parameter name invalid

**Example**:
```python
fig = model.plot_param_hist(parameter='beta')
```

---

## Properties

### `model.params`

**Type**: `PyTree` (jax PyTree structure)

**Description**: Fitted model parameters.

**Structure**:
```python
{
    'beta': jax.numpy.array([...]),      # Shape: (n_mutations,)
    'beta0': jax.numpy.array([...]),     # Shape: (n_conditions,)
    'shift': jax.numpy.array([...]),     # Shape: (n_mutations, n_conditions-1)
    'theta': {...}                        # GE-specific parameters
}
```

**Access**:
```python
if model.params is not None:  # Check if fitted
    beta_values = model.params['beta']
```

---

### `model.convergence_trajectory_df`

**Type**: `pd.DataFrame`

**Description**: Optimization history.

**Columns**:
- `iteration`: Iteration number (0 to maxiter)
- `loss`: Loss value at this iteration
- `error`: Change in loss from previous iteration

**Example**:
```python
print(model.convergence_trajectory_df.tail())
#    iteration      loss     error
# 995       995  0.123456  0.000001
# 996       996  0.123455  0.000001
```

---

## State Management

### Model States

**Unfitted**:
- `model.params` is `None`
- Prediction methods raise `RuntimeError`
- Visualization methods raise `RuntimeError`

**Fitted**:
- `model.params` is populated PyTree
- All methods available
- Can refit by calling `fit()` again (overwrites params)

### State Transitions

```
[Constructed]
    → fit() called
    → [Fitting in progress]
    → convergence or maxiter
    → [Fitted]
```

---

## Warmstart Behavior

### Ridge Regression Initialization

**When warmstart=True**:
1. If count data available: Use counts as sample weights in Ridge
2. If count data unavailable: Run Ridge without weights (equal weighting)
3. Ridge solution initializes beta and beta0
4. Shift initialized to zero
5. GE parameters (theta) initialized to sensible defaults

**Fallback**:
- If Ridge fails: Initialize all parameters to zero
- Warning issued: "Ridge warmstart failed: {error}. Using zero initialization."

---

## Performance Contract

**Time Complexity**:
- `fit()`: O(iterations × n_variants × n_mutations)
- `get_mutations_df()`: O(n_mutations × n_conditions)
- `get_variants_df()`: O(n_variants)
- `add_phenotypes_to_df()`: O(n_new_variants × n_mutations)

**Space Complexity**:
- `params`: O(n_mutations × n_conditions)
- `convergence_trajectory_df`: O(iterations)

**Performance Target**:
- Fitting time ≤ 2× v1.x for datasets with >1000 variants, 3 conditions

---

## Error Handling Summary

| Error Condition | Exception Type | Message Format |
|----------------|----------------|----------------|
| Model not fitted | `RuntimeError` | "Model must be fitted before calling {method}. Call model.fit() first." |
| Count data missing for count_loss | `ValueError` | "Count data required when loss_type='count_loss'. Provide pre_counts, post_counts columns." |
| Warmstart param shape mismatch | `ValueError` | "Warmstart parameter {param} shape {shape} doesn't match expected {expected}." |
| Unseen mutations in prediction | `ValueError` | "Cannot predict on unseen mutations: {mutations}. Retrain with expanded data." |
| Invalid GE type | `ValueError` | "Invalid ge_type '{ge_type}'. Must be 'Identity' or 'Sigmoid'." |
| Convergence failure | `Warning` | "Model did not converge. Final error: {err:.6f}, tolerance: {tol:.6f}." |

---

## Backward Compatibility

**v1.x → v2.0 Compatibility**: ✅ **HIGH** (minor breaking changes)

**Compatible**:
- Constructor parameters (same names, same defaults)
- `fit()` method signature
- `get_mutations_df()` output format
- `get_variants_df()` output format
- All visualization methods

**Breaking Changes**:
- Internal implementation uses jaxmodels (computational backend changed)
- Some numerical differences expected due to JAX vs NumPy/SciPy
- Warmstart behavior slightly different (now works without counts)

**Migration Path**:
- Update code: None required for typical usage
- Validate results: Compare v1.x and v2.0 outputs on same data
- Expected difference: <1% in parameter estimates
