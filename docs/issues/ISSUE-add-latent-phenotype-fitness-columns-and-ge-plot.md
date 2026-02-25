# Add latent phenotype and fitness columns to variant DataFrames, plus global epistasis landscape plot

## Summary

Extend `Model.get_variants_df()` and `Model.add_phenotypes_to_df()` to return the intermediate latent phenotype (`predicted_latent`) and back-transformed fitness score (`predicted_fitness`, `measured_fitness`) for each variant, in addition to the existing `predicted_func_score`. Also add a new function `multidms.plot.ge_landscape()` that produces an Altair chart of the global epistasis curve overlaid on per-variant fitness scores — a plot central to interpreting DMS global epistasis models.

## Background

- **Prior behavior (v0.4.0):** `Model.get_variants_df()` returned both latent phenotypes and predicted phenotypes. Users could directly inspect the latent space and plot the global epistasis function against variant fitness.
- **Current behavior (v2.0):** After the jaxmodels refactor, `get_variants_df()` returns only `predicted_func_score` (the fully-transformed functional score: `α * (g(φ(X)) - g(φ(x_wt)))`). The latent phenotype `φ(X)` and back-transformed fitness are not exposed through the public API.
- **What is missing:** Users who want to visualize or analyze the global epistasis landscape must reach into private `_jax_model` internals (as the `jaxmodels_simulation_fits.ipynb` notebook currently does). There is no public method or tidy DataFrame that provides these intermediate quantities, and no built-in plot for the global epistasis landscape.

## Proposed Approach

Add three columns to the variant DataFrames produced by `get_variants_df()` and `add_phenotypes_to_df()`, and add a new Altair-based plotting function to `multidms.plot`.

**Key design decisions:**

1. **Compute latent phenotype per-variant per-condition** using the existing `self._jax_model.φ[d](X)` — this is the natural intermediate that the model already computes internally.
2. **Provide both measured and predicted fitness** where possible. Predicted fitness is always available; measured fitness requires observed functional scores in the DataFrame.
3. **Fitness formula:** `f = y / α + g(φ(x_wt))`, where `y` is the functional score (measured or predicted), `α` is `model._jax_model.α[d]`, and `g(φ(x_wt))` is the global epistasis function evaluated at the wildtype latent phenotype. This back-transforms functional scores into the `g(φ)` space so they can be overlaid on the global epistasis curve.
4. **Plot as a standalone function in `multidms.plot`** — keeps the `Model` class lean and is consistent with the existing `plot.py` module pattern.
5. **Tidy DataFrame method** (`Model.get_ge_landscape_df()`) that returns all data needed for plotting, so users can also build custom plots.

## User Interface / API

### Programmatic API

#### Enhanced `get_variants_df()`

```python
model = Model(data, ge_type="Sigmoid")
model.fit()

df = model.get_variants_df()
# Now includes columns:
#   - predicted_func_score  (existing)
#   - predicted_latent      (NEW: φ(X) for each variant)
#   - predicted_fitness     (NEW: predicted_func_score / α + g(φ(x_wt)))
#   - measured_fitness      (NEW: func_score / α + g(φ(x_wt)))
```

#### Enhanced `add_phenotypes_to_df()`

```python
new_df = model.add_phenotypes_to_df(my_variants_df)
# Always adds:
#   - predicted_func_score  (existing)
#   - predicted_latent      (NEW)
#   - predicted_fitness     (NEW)
# Conditionally adds (if 'func_score' column present):
#   - measured_fitness      (NEW)
```

#### New: `get_ge_landscape_df()`

```python
ge_df = model.get_ge_landscape_df()
# Returns a tidy DataFrame with columns:
#   condition, predicted_latent, predicted_fitness, measured_fitness,
#   plus the ge_curve as a separate key or merged DataFrame
```

#### New: `multidms.plot.ge_landscape()`

```python
import multidms.plot as mplt

chart = mplt.ge_landscape(model)
chart  # Altair chart: hexbin/point density of (predicted_latent, fitness)
       # with g(φ) curve overlay and WT vertical lines per condition
```

## Proposed Changes

### 1. `Model.get_variants_df()` — add latent and fitness columns

**File:** `multidms/model.py` (existing)

Inside the per-condition loop, compute and add three new columns:

```python
for condition in self._data.conditions:
    cond_data = self._data.variants_df[
        self._data.variants_df.condition == condition
    ].copy()

    pred_scores = predictions[condition]
    full_predictions = np.concatenate([[0.0], np.array(pred_scores)])
    cond_data["predicted_func_score"] = full_predictions[: len(cond_data)]

    # NEW: latent phenotype
    φ = self._jax_model.φ[condition]
    X = self._jax_data_sets[condition].X
    x_wt = self._jax_data_sets[condition].x_wt
    φ_X = np.array(φ(X))
    φ_wt = float(φ(x_wt))
    # WT is index 0, its latent phenotype is φ(x_wt)
    full_latent = np.concatenate([[φ_wt], φ_X])
    cond_data["predicted_latent"] = full_latent[: len(cond_data)]

    # NEW: fitness (back-transformed into g(φ) space)
    α = float(self._jax_model.α[condition])
    g_φ_wt = float(self._jax_model.global_epistasis(φ(x_wt)))

    cond_data["predicted_fitness"] = (
        cond_data["predicted_func_score"] / α + g_φ_wt
    )
    cond_data["measured_fitness"] = (
        cond_data["func_score"] / α + g_φ_wt
    )

    result_rows.append(cond_data)
```

### 2. `Model.add_phenotypes_to_df()` — add latent and fitness columns

**File:** `multidms/model.py` (existing)

After computing `predicted_func_score` via `predict_score`, also compute latent phenotype and fitness for each condition group:

```python
for condition, (temp_data, condition_df) in encoded.items():
    temp_data_sets = {condition: temp_data}
    predictions = self._jax_model.predict_score(temp_data_sets)
    phenotype_predictions = np.array(predictions[condition])

    ret.loc[condition_df.index.values, predicted_phenotype_col] = (
        phenotype_predictions
    )

    # NEW: latent phenotype
    φ = self._jax_model.φ[condition]
    φ_X = np.array(φ(temp_data.X))
    ret.loc[condition_df.index.values, "predicted_latent"] = φ_X

    # NEW: fitness
    α = float(self._jax_model.α[condition])
    g_φ_wt = float(
        self._jax_model.global_epistasis(φ(temp_data.x_wt))
    )
    ret.loc[condition_df.index.values, "predicted_fitness"] = (
        phenotype_predictions / α + g_φ_wt
    )

    # Measured fitness (only if func_score column exists)
    if "func_score" in ret.columns:
        measured_scores = ret.loc[condition_df.index.values, "func_score"]
        ret.loc[condition_df.index.values, "measured_fitness"] = (
            measured_scores / α + g_φ_wt
        )
```

### 3. New method: `Model.get_ge_landscape_df()`

**File:** `multidms/model.py` (existing)

A convenience method that returns a tidy DataFrame suitable for plotting:

```python
def get_ge_landscape_df(
    self,
    n_curve_points: int = 200,
) -> pd.DataFrame:
    """Get a tidy DataFrame for plotting the global epistasis landscape.

    Returns a DataFrame with per-variant rows containing latent phenotype
    and fitness values, plus rows for the global epistasis curve.

    Parameters
    ----------
    n_curve_points : int
        Number of points for the g(φ) curve grid.

    Returns
    -------
    pd.DataFrame
        Tidy DataFrame with columns: condition, predicted_latent,
        predicted_fitness, measured_fitness, source ('variant' or 'curve'),
        and wildtype_latent.
    """
    variants_df = self.get_variants_df()

    # Add source label
    variants_df["source"] = "variant"

    # Compute ge curve spanning the observed latent range
    φ_min = variants_df["predicted_latent"].min()
    φ_max = variants_df["predicted_latent"].max()
    margin = (φ_max - φ_min) * 0.05
    ge_curve = self.get_ge_curve(
        grid_min=float(φ_min - margin),
        grid_max=float(φ_max + margin),
        n_points=n_curve_points,
    )
    ge_curve = ge_curve.rename(columns={
        "latent": "predicted_latent",
        "observed": "ge_curve_value",
    })
    ge_curve["source"] = "curve"

    # Add wildtype latent to variants_df for reference lines
    wt_latent = self.wildtype_latent
    variants_df["wildtype_latent"] = variants_df["condition"].map(wt_latent)

    return variants_df, ge_curve
```

### 4. New function: `multidms.plot.ge_landscape()`

**File:** `multidms/plot.py` (existing)

An Altair-based function that creates a layered chart:

```python
def ge_landscape(
    model,
    fitness_col="measured_fitness",
    color_by="condition",
    point_size=5,
    point_opacity=0.3,
    curve_color="grey",
    curve_width=3,
    width=500,
    height=400,
):
    """Plot the global epistasis landscape.

    Overlays per-variant fitness scores on the global epistasis curve
    g(φ), showing how the nonlinear transformation maps latent
    phenotype to observed fitness.

    Parameters
    ----------
    model : multidms.Model
        A fitted Model object.
    fitness_col : str
        Which fitness column to plot: 'measured_fitness' or
        'predicted_fitness'.
    color_by : str
        Column to color points by (default: 'condition').
    ...

    Returns
    -------
    alt.LayerChart
        Altair layered chart.
    """
    variants_df, ge_curve = model.get_ge_landscape_df()

    # Scatter layer: variant fitness vs latent phenotype
    scatter = (
        alt.Chart(variants_df)
        .mark_circle(size=point_size, opacity=point_opacity)
        .encode(
            x=alt.X("predicted_latent:Q", title="Predicted latent phenotype (φ)"),
            y=alt.Y(f"{fitness_col}:Q", title="Fitness"),
            color=alt.Color(f"{color_by}:N"),
        )
    )

    # Curve layer: g(φ)
    curve = (
        alt.Chart(ge_curve)
        .mark_line(color=curve_color, strokeWidth=curve_width)
        .encode(
            x="predicted_latent:Q",
            y=alt.Y("ge_curve_value:Q"),
        )
    )

    # WT reference lines
    wt_data = variants_df[["condition", "wildtype_latent"]].drop_duplicates()
    wt_rules = (
        alt.Chart(wt_data)
        .mark_rule(strokeDash=[4, 4], opacity=0.6)
        .encode(
            x="wildtype_latent:Q",
            color=alt.Color("condition:N"),
        )
    )

    return (scatter + curve + wt_rules).properties(
        width=width, height=height
    )
```

## Implementation Details

### Mathematical Background

The model predicts functional scores as:

```
y_pred[d] = α[d] * (g(φ[d](X)) - g(φ[d](x_wt)))
```

Where:
- `φ[d](X) = β0[d] + X @ β[d]` is the latent phenotype
- `g(·)` is the global epistasis function (Identity or Sigmoid)
- `α[d]` is a per-condition scaling factor
- `x_wt` is the wildtype binary encoding

To back-transform observed functional scores into "fitness" (the `g(φ)` space):

```
fitness = y / α + g(φ(x_wt))
```

This places observed data points in the same coordinate system as `g(φ)`, enabling direct visual comparison of variants to the global epistasis curve.

### Column Naming Convention

| Column | Description | Formula |
|---|---|---|
| `predicted_func_score` | Model-predicted functional score (existing) | `α * (g(φ(X)) - g(φ(x_wt)))` |
| `predicted_latent` | Latent phenotype | `φ(X) = β0 + X @ β` |
| `predicted_fitness` | Predicted fitness in g(φ) space | `predicted_func_score / α + g(φ(x_wt))` |
| `measured_fitness` | Measured fitness in g(φ) space | `func_score / α + g(φ(x_wt))` |

## Testing Strategy

- **Doctests** in `get_variants_df()` and `add_phenotypes_to_df()` verifying the new columns are present and have expected values for a simple two-condition model
- **Unit test** checking that `predicted_fitness` == `g(predicted_latent)` for predicted data (verifying internal consistency)
- **Unit test** checking that wildtype rows have `predicted_latent` == `wildtype_latent[condition]`
- **Unit test** for `ge_landscape()` verifying it returns an `alt.LayerChart` without error
- **Edge cases:**
  - Identity global epistasis (g = identity): fitness should equal `func_score / α + φ(x_wt)`
  - Single-condition model
  - `add_phenotypes_to_df` with a DataFrame lacking `func_score` — `measured_fitness` should be absent or NaN

## Documentation Updates

- [ ] Docstrings for new columns in `get_variants_df()` and `add_phenotypes_to_df()`
- [ ] Docstring for `get_ge_landscape_df()`
- [ ] Docstring for `ge_landscape()` in `plot.py`
- [ ] Update notebook `jaxmodels_simulation_fits.ipynb` to use the new API instead of reaching into `_jax_model` internals

## Dependencies

No new dependencies. Altair is already a dependency. All computation uses existing JAX/numpy infrastructure.

## Alternatives Considered

### Alternative 1: Matplotlib-based plot

Use matplotlib with hexbin (matching the existing notebook pattern).

**Why not chosen:** The user prefers Altair for interactivity and consistency with the existing `plot.py` module. A tidy DataFrame method (`get_ge_landscape_df`) enables users to create custom matplotlib plots if desired.

### Alternative 2: Add plot as a Model method

Add `Model.plot_ge_landscape()` directly.

**Why not chosen:** Keeps the Model class focused on data/fitting; plotting functions belong in `multidms.plot` per established convention.

### Alternative 3: Return latent phenotype only, let users compute fitness

Only add `predicted_latent` and let users do the `y/α + g(φ_wt)` transformation themselves.

**Why not chosen:** The fitness transformation is non-obvious and error-prone. Providing it directly makes the common workflow (plotting the GE landscape) trivial.

## Success Criteria

- [ ] `model.get_variants_df()` returns `predicted_latent`, `predicted_fitness`, and `measured_fitness` columns
- [ ] `model.add_phenotypes_to_df(df)` adds `predicted_latent` and `predicted_fitness`; adds `measured_fitness` when `func_score` is present
- [ ] `model.get_ge_landscape_df()` returns tidy DataFrames suitable for custom plotting
- [ ] `multidms.plot.ge_landscape(model)` produces an Altair chart matching the notebook's visual pattern
- [ ] Existing doctests and unit tests continue to pass
- [ ] The `jaxmodels_simulation_fits.ipynb` notebook can be simplified to use the new public API

## Future Work

- Add density-based rendering (2D histogram / hexbin equivalent in Altair) for large variant counts
- Support for `ModelCollection.plot_ge_landscape()` showing multiple models
- Option to facet by condition in the Altair chart
- Color variants by number of mutations or other variant metadata
