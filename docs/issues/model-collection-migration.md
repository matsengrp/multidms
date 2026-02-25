# Update `Model` and `ModelCollection` for new jaxmodels backend

**Labels:** enhancement, P1

## Summary

After v1.2.0, `multidms.Model` was rewritten to use a new `jaxmodels.py` backend. However, `model_collection.py` was not updated and is completely broken against the new `Model` API. This issue covers all changes to `model.py` and `model_collection.py` needed to make `ModelCollection` functional again.

See `docs/model_collection_migration.md` for the full findings and API audit.

---

## Changes to `multidms/model.py`

### 1. Store fit metadata after fitting

`fit()` currently discards the loss function, its kwargs, and the convergence tolerance after fitting. These are needed by new properties/methods below.

**In `__init__`**, initialize:
```python
self._fit_tol = None
self._loss_fn = None
self._loss_kwargs = None
```

**In `fit()`**, store before calling `jaxmodels.fit()`:
```python
self._fit_tol = tol
self._loss_fn = loss_fn
self._loss_kwargs = loss_kwargs if loss_kwargs is not None else {}
```

### 2. Add `converged` property

`ModelCollection.__init__()` line 309 accesses `model.converged`.

```python
@property
def converged(self) -> bool:
    """Whether the model fitting converged."""
    if self._loss_trajectory is None or len(self._loss_trajectory) < 2:
        return False
    last_two = self._loss_trajectory[-2:]
    error = abs(last_two[-1] - last_two[-2]) / max(
        abs(last_two[-1]), abs(last_two[-2]), 1
    )
    return error < self._fit_tol
```

### 3. Add `conditional_loss` property

`ModelCollection.__init__()` lines 313-315 accesses `model.conditional_loss`.

```python
@property
def conditional_loss(self) -> dict[str, float]:
    """Per-condition loss on training data."""
    if self._jax_model is None:
        raise ValueError("Model has not been fitted. Call fit() first.")
    loss_dict = self._loss_fn(
        self._jax_model, self._jax_data_sets, **self._loss_kwargs
    )
    return {k: float(v) for k, v in loss_dict.items()}
```

### 4. Add `get_df_loss()` method

`ModelCollection.add_validation_loss()` line 516 calls `model.get_df_loss(test_data, conditional=True)`.

```python
def get_df_loss(self, df, conditional=False) -> dict[str, float] | float:
    """Evaluate the model's loss on an arbitrary DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'condition', 'aa_substitutions', 'func_score' columns.
    conditional : bool
        If True, return per-condition loss dict. If False, return total loss.
    """
```

**Implementation**: Reuse encoding logic from `add_phenotypes_to_df()` to build temporary `jaxmodels.Data` objects from the DataFrame, but with actual `func_score` values (not zeros). Then evaluate `self._loss_fn(self._jax_model, temp_data_sets, **self._loss_kwargs)`.

Consider extracting the shared encoding logic (substitution string -> sparse binary matrix) into a private helper `_encode_variants_df()` that both `add_phenotypes_to_df` and `get_df_loss` call.

### 5. Extend `get_mutations_df()` with `times_seen_threshold`

`ModelCollection.split_apply_combine_muts()` passes `times_seen_threshold` via `**kwargs`.

**New signature:**
```python
def get_mutations_df(
    self,
    phenotype_as_effect: bool = True,
    times_seen_threshold: int = 0,
) -> pd.DataFrame:
```

**Filtering logic** (after building the mutations DataFrame):
```python
if times_seen_threshold > 0:
    times_seen_cols = [c for c in mutations_df.columns if c.startswith("times_seen_")]
    mask = mutations_df[times_seen_cols].min(axis=1) >= times_seen_threshold
    mutations_df = mutations_df[mask]
```

Note: the old API also had `return_split` — do NOT add this. ModelCollection always passed `return_split=False`, which is the default behavior of the new API already.

### 6. Add GE visualization support

The old notebooks use `model.get_condition_params()`, `model.model_components["g"]`, and `model.wildtype_df` for plotting the global epistasis curve. Replace with a cleaner interface:

```python
def get_ge_curve(self, grid_min=-5.0, grid_max=5.0, n_points=200) -> pd.DataFrame:
    """Evaluate the global epistasis function over a latent phenotype grid.

    Returns DataFrame with columns: 'latent', 'observed'.
    """
    if self._jax_model is None:
        raise ValueError("Model has not been fitted. Call fit() first.")
    grid = jnp.linspace(grid_min, grid_max, n_points)
    observed = self._jax_model.global_epistasis(grid)
    return pd.DataFrame({"latent": np.array(grid), "observed": np.array(observed)})

@property
def wildtype_latent(self) -> dict[str, float]:
    """Wildtype latent phenotype for each condition."""
    if self._jax_model is None:
        raise ValueError("Model has not been fitted. Call fit() first.")
    result = {}
    for condition in self._data.conditions:
        latent = self._jax_model.φ[condition]
        x_wt = self._jax_data_sets[condition].x_wt
        result[condition] = float(latent.β0 + x_wt @ latent.β)
    return result
```

---

## Changes to `multidms/model_collection.py`

### 7. Rewrite `fit_one_model()`

The current function (lines 50-155) is completely broken — it references the removed `multidms.biophysical` module, uses old parameter names, and passes `warn_unconverged` to `fit()`.

**New signature:**
```python
def fit_one_model(
    dataset,
    ge_type="Sigmoid",
    l2reg=0.0,
    fusionreg=0.0,
    beta0_ridge=0.0,
    loss_type="functional_score_loss",
    maxiter=10,
    tol=1e-6,
    warmstart=True,
    beta0_init=None,
    beta_init=None,
    alpha_init=None,
    beta_clip_range=None,
    ge_kwargs=None,
    cal_kwargs=None,
    loss_kwargs=None,
    verbose=False,
    **kwargs,
):
```

**Implementation:**
1. Save `locals()` for bookkeeping
2. Create `multidms.Model(dataset, ge_type=..., l2reg=..., fusionreg=..., beta0_ridge=..., loss_type=...)`
3. Call `model.fit(warmstart=..., maxiter=..., tol=..., ...)` with fitting parameters
4. Return `pd.Series` with model + all hyperparameters + `dataset_name` + `fit_time`

Also remove old `multidms.biophysical` imports and update `PARAMETER_NAMES_FOR_PLOTTING`:
```python
# OLD
PARAMETER_NAMES_FOR_PLOTTING = {"scale_coeff_lasso_shift": "Lasso Penalty"}
# NEW
PARAMETER_NAMES_FOR_PLOTTING = {"fusionreg": "Fusion Regularization"}
```

### 8. Update `split_apply_combine_muts()`

Lines 409 and 442 pass `return_split=False` to `get_mutations_df()` — remove this argument:
```python
# OLD
fit["model"].get_mutations_df(return_split=False, **kwargs)
# NEW
fit["model"].get_mutations_df(**kwargs)
```

### 9. Update `mut_param_heatmap()` and `mut_param_traceplot()` melt logic

The old API returned a single `beta` column (reference only). The new API returns `beta_{condition}` for all conditions. The melt logic has a special case for `beta` that needs removing:

```python
# OLD — special case for beta (single column)
if mut_param == "beta":
    muts_df_tall = muts_df.assign(condition=self.reference.replace(".", "_"))
else:
    muts_df_tall = muts_df.melt(...)

# NEW — melt works for both beta and shift (both are per-condition now)
muts_df_tall = muts_df.melt(
    id_vars=["wildtype", "site", "mutant"] + addtl_tooltip_stats,
    value_vars=[c for c in muts_df.columns if c.startswith(mut_param)],
    var_name="condition",
    value_name=mut_param,
)
```

Apply the same fix in `mut_param_traceplot()` (lines 838-849).

The `.replace()` logic that strips the prefix to get condition names should work for both `beta_` and `shift_` prefixes since the pattern is identical.

### 10. Update default `id_vars` in loss/trajectory methods

**`get_conditional_loss_df()`** line 544:
```python
# OLD
id_vars = ["dataset_name", "scale_coeff_lasso_shift"]
# NEW
id_vars = ["dataset_name", "fusionreg"]
```

**`convergence_trajectory_df()`** line 563:
```python
# OLD
id_vars=("dataset_name", "scale_coeff_lasso_shift"),
# NEW
id_vars=("dataset_name", "fusionreg"),
```

### 11. Update `add_validation_loss()` — populate total loss

The `total_loss_validation` column (line 512) is initialized but never populated. Add:
```python
self.fit_models.loc[idx, "total_loss_validation"] = sum(condional_df_loss.values())
```

Also clean up `get_conditional_loss_df()` `step_loss` filter (line 546) — the `step_loss` column no longer exists in the new API. The filter is harmless but should be removed for clarity.

### 12. `shift_sparsity()` — no changes needed

Only looks at `shift_` columns, which have the same naming in the new API.

---

## Old → New parameter name mapping (reference)

| Old Parameter | New Equivalent | Notes |
|---|---|---|
| `dataset` | `dataset` | Unchanged |
| `epistatic_model` | `ge_type` | "Identity" or "Sigmoid" only |
| `scale_coeff_lasso_shift` | `fusionreg` | |
| `scale_coeff_ridge_beta` | `l2reg` | |
| `scale_coeff_ridge_alpha_d` | *(dropped)* | No alpha regularization in new API |
| `alpha_d` | *(dropped)* | |
| `init_beta_naught` | `beta0_init` | Now a dict keyed by condition |
| `num_training_steps` | `maxiter` | |
| `iterations_per_step` | *(dropped)* | Controlled via `ge_kwargs`/`cal_kwargs` |
| `warn_unconverged` | *(dropped)* | |

---

## Testing checklist

- [ ] `model.fit()` stores `_fit_tol`, `_loss_fn`, `_loss_kwargs`
- [ ] `model.converged` returns `True` after convergence, `False` before fit or after 1 iteration
- [ ] `model.conditional_loss` returns dict with one key per condition
- [ ] `model.get_df_loss(df, conditional=True)` returns per-condition loss dict
- [ ] `model.get_df_loss(df, conditional=False)` returns sum of per-condition losses
- [ ] `model.get_df_loss(training_data)` matches `model.conditional_loss`
- [ ] `model.get_mutations_df(times_seen_threshold=0)` matches current behavior
- [ ] `model.get_mutations_df(times_seen_threshold=N)` filters correctly
- [ ] `model.get_ge_curve()` returns DataFrame with `latent` and `observed` columns
- [ ] `model.get_ge_curve()` with Identity GE returns `observed == latent`
- [ ] `model.wildtype_latent` returns dict with one key per condition
- [ ] `fit_one_model(data, ge_type='Sigmoid', fusionreg=0.01)` returns pd.Series with fitted model
- [ ] `fit_models({'dataset': [data], 'fusionreg': [0.0, 0.01]})` returns `(2, 0, DataFrame)`
- [ ] `ModelCollection(models_df)` initializes without error
- [ ] `mc.split_apply_combine_muts()` returns DataFrame with `beta_{condition}` and `shift_{condition}` columns
- [ ] `mc.mut_param_heatmap(mut_param="beta")` melts `beta_{condition}` columns correctly
- [ ] `mc.mut_param_heatmap(mut_param="shift")` works as before
- [ ] `mc.add_validation_loss(test_data)` populates validation loss columns
- [ ] `mc.get_conditional_loss_df()` returns tidy DataFrame with training/validation rows
- [ ] `mc.shift_sparsity()` works unchanged
