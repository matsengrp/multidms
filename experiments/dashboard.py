"""Interactive marimo dashboard for exploring ModelCollection results.

Param Correlation and Replicate Scatter tabs support a ``times_seen_threshold``
slider that filters out mutations unseen in some conditions before the
correlation is computed.
"""

import marimo

app = marimo.App(width="medium")


# ── A: Setup ──────────────────────────────────────────────────────────────


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import pickle
    from pathlib import Path

    import pandas as pd

    return os, pickle, Path, pd


@app.cell
def _(mo):
    mo.md(
        """
        # multidms Dashboard

        Interactive exploration of `ModelCollection` results from
        simulation and spike pipelines.
        """
    )
    return


# ── B: Discovery + Loading ───────────────────────────────────────────────


@app.cell
def _(Path, os):
    # Resolve the experiments directory relative to this file
    _dashboard_dir = Path(os.path.abspath(__file__)).parent

    # Scan for fit_collection.pkl files
    _pkl_paths = sorted(_dashboard_dir.glob("*/results*/fit_collection.pkl"))

    # Build mapping: display label -> path
    discovered_collections = {}
    for p in _pkl_paths:
        label = f"{p.parent.parent.name}/{p.parent.name}"
        discovered_collections[label] = p

    return (discovered_collections,)


@app.cell
def _(mo, discovered_collections):
    if not discovered_collections:
        mo.stop(
            True,
            mo.md(
                "**No `fit_collection.pkl` found.** "
                "Run a pipeline first (`pixi run sim-test`)."
            ),
        )

    pipeline_dropdown = mo.ui.dropdown(
        options=list(discovered_collections.keys()),
        value=list(discovered_collections.keys())[0],
        label="Pipeline results",
    )
    pipeline_dropdown
    return (pipeline_dropdown,)


@app.cell
def _(mo, pickle, pipeline_dropdown, discovered_collections):
    from multidms.model_collection import ModelCollection

    mo.stop(not pipeline_dropdown.value, mo.md("Select a pipeline above."))

    _pkl_path = discovered_collections[pipeline_dropdown.value]
    with open(_pkl_path, "rb") as _f:
        _loaded = pickle.load(_f)

    if isinstance(_loaded, ModelCollection):
        mc = _loaded
    else:
        mc = ModelCollection(_loaded)

    _n_models = len(mc.fit_models)
    datasets = list(mc.fit_models["dataset_name"].unique())
    fusionregs = [str(f) for f in sorted(mc.fit_models["fusionreg"].unique())]

    mo.md(
        f"Loaded **{_n_models}** models from `{pipeline_dropdown.value}` "
        f"| datasets: {datasets} | fusionreg values: {fusionregs}"
    )
    return mc, datasets, fusionregs


@app.cell
def _():
    import multidms.plot as mplot

    return (mplot,)


# ── C: Tab controls (always available) ───────────────────────────────────


@app.cell
def _(mo, datasets, fusionregs):
    conv_dataset_select = mo.ui.multiselect(
        options=datasets,
        value=datasets[:2],
        label="Datasets",
    )
    conv_fusionreg_select = mo.ui.multiselect(
        options=fusionregs,
        value=fusionregs[:1],
        label="Fusion reg",
    )
    ge_dataset_dropdown = mo.ui.dropdown(
        options=datasets,
        value=datasets[0],
        label="Dataset",
    )
    ge_fusionreg_dropdown = mo.ui.dropdown(
        options=fusionregs,
        value=fusionregs[0],
        label="Fusion reg",
    )
    scatter_fusionreg_dropdown = mo.ui.dropdown(
        options=fusionregs,
        value=fusionregs[0],
        label="Fusion reg",
    )
    return (
        conv_dataset_select,
        conv_fusionreg_select,
        ge_dataset_dropdown,
        ge_fusionreg_dropdown,
        scatter_fusionreg_dropdown,
    )


# Scatter param dropdown depends on the muts columns, so we derive param
# options from the collection's conditions.
@app.cell
def _(mo, mc):
    _conditions = mc._conditions
    _param_options = []
    for c in _conditions:
        _param_options.append(f"beta_{c}")
    for c in _conditions:
        col = f"shift_{c}"
        # shift columns only exist for non-reference conditions
        _param_options.append(col)
    for c in _conditions:
        _param_options.append(f"predicted_func_score_{c}")

    scatter_param_dropdown = mo.ui.dropdown(
        options=_param_options,
        value=_param_options[0],
        label="Parameter",
    )
    return (scatter_param_dropdown,)


@app.cell
def _(mo):
    times_seen_threshold_slider = mo.ui.slider(
        start=0,
        stop=20,
        step=1,
        value=1,
        label="Min times_seen (per-condition) to include",
    )
    return (times_seen_threshold_slider,)


# ── D: Tab chart computations ────────────────────────────────────────────

# --- Convergence ---


@app.cell
def _(mo, mc, mplot, conv_dataset_select, conv_fusionreg_select):
    if not conv_dataset_select.value or not conv_fusionreg_select.value:
        convergence_chart = mo.md(
            "Select at least one dataset and one fusionreg value."
        )
    else:
        _ds = conv_dataset_select.value
        _fr = [float(x) for x in conv_fusionreg_select.value]
        _query = f"dataset_name.isin({_ds}) and fusionreg.isin({_fr})"
        _conv_df = mc.convergence_trajectory_df(query=_query)
        convergence_chart = mplot.convergence_trajectory(
            _conv_df, id_cols=["dataset_name", "fusionreg"]
        )
    return (convergence_chart,)


# --- GE Landscape ---


@app.cell
def _(mo, mc, mplot, ge_dataset_dropdown, ge_fusionreg_dropdown):
    if not ge_dataset_dropdown.value or not ge_fusionreg_dropdown.value:
        ge_chart = mo.md("Select a dataset and fusionreg.")
    else:
        ge_ds = ge_dataset_dropdown.value  # noqa: F841 (used in query)
        ge_fr = float(ge_fusionreg_dropdown.value)  # noqa: F841 (used in query)
        _row = mc.fit_models.query(
            "dataset_name == @ge_ds and fusionreg == @ge_fr"
        ).iloc[0]
        _model = _row["model"]
        _variants_df, _ge_curve_df = _model.get_ge_landscape_df()
        _max_points = 5000
        if len(_variants_df) > _max_points:
            _variants_df = _variants_df.sample(n=_max_points, random_state=0)
        ge_chart = mplot.ge_landscape(_variants_df, _ge_curve_df, point_size=20)
    return (ge_chart,)


# --- Correlation ---


@app.cell
def _(mo, mc, times_seen_threshold_slider):
    if len(mc.fit_models["dataset_name"].unique()) < 2:
        correlation_chart = mo.md("Need at least 2 datasets for correlation analysis.")
    else:
        try:
            correlation_chart = mc.mut_param_dataset_correlation(
                times_seen_threshold=times_seen_threshold_slider.value,
            )
        except Exception as _e:
            correlation_chart = mo.md(
                f"Correlation failed (threshold too high?): {_e}. "
                "Try lowering the threshold slider."
            )
    return (correlation_chart,)


# --- Scatter ---


@app.cell
def _(
    mo,
    mc,
    mplot,
    datasets,
    scatter_fusionreg_dropdown,
    scatter_param_dropdown,
    times_seen_threshold_slider,
):
    if len(datasets) < 2:
        scatter_chart = mo.md("Need at least 2 datasets for scatter comparison.")
    else:
        _fr = float(scatter_fusionreg_dropdown.value)
        _param_col = scatter_param_dropdown.value

        _muts_df = mc.split_apply_combine_muts(
            groupby=("dataset_name", "fusionreg"),
            query=f"fusionreg == {_fr}",
            times_seen_threshold=times_seen_threshold_slider.value,
        ).reset_index()

        d0, d1 = datasets[0], datasets[1]  # noqa: F841 (used in query)
        _df0 = _muts_df.query("dataset_name == @d0")[["mutation", _param_col]]
        _df1 = _muts_df.query("dataset_name == @d1")[["mutation", _param_col]]

        _x_label = f"{_param_col} ({d0})"
        _y_label = f"{_param_col} ({d1})"

        _merged = _df0.merge(
            _df1, on="mutation", suffixes=(f"_{d0}", f"_{d1}")
        ).dropna()
        _x_col = f"{_param_col}_{d0}"
        _y_col = f"{_param_col}_{d1}"

        scatter_chart = mplot.replicate_param_scatter(
            _merged,
            x_col=_x_col,
            y_col=_y_col,
            x_label=_x_label,
            y_label=_y_label,
        )
    return (scatter_chart,)


# --- Sparsity ---


@app.cell
def _(mo, mc):
    _fusionregs = sorted(mc.fit_models["fusionreg"].unique())
    if len(_fusionregs) <= 1:
        sparsity_chart = mo.md(
            "Sparsity plot requires multiple fusionreg values. "
            "Only one value found in this collection."
        )
    else:
        try:
            sparsity_chart = mc.shift_sparsity()
        except Exception as _e:
            sparsity_chart = mo.md(f"Sparsity chart failed: {_e}")
    return (sparsity_chart,)


# --- Summary ---


@app.cell
def _(mo, mc, pd):
    _rows = []
    for _, _fit in mc.fit_models.iterrows():
        _model = _fit["model"]
        _jm = _model._jax_model
        _row = {
            "dataset": _fit["dataset_name"],
            "fusionreg": _fit["fusionreg"],
            "converged": _fit.get("converged", "N/A"),
            "fit_time": _fit.get("fit_time", "N/A"),
            "ge_type": _fit.get("ge_type", "N/A"),
        }
        # Alpha: shared scalar or legacy per-condition dict
        if hasattr(_jm.α, "items"):
            for cond, val in _jm.α.items():
                _row[f"alpha_{cond}"] = round(float(val), 4)
        else:
            _row["alpha"] = round(float(_jm.α), 4)
        # Beta0: per-condition from Latent objects
        for cond, latent in _jm.φ.items():
            if hasattr(latent, "β0"):
                _row[f"beta0_{cond}"] = round(float(latent.β0), 4)
        _rows.append(_row)
    summary_table = mo.ui.table(pd.DataFrame(_rows))
    return (summary_table,)


# ── E: Layout Assembly ───────────────────────────────────────────────────


@app.cell
def _(
    mo,
    conv_dataset_select,
    conv_fusionreg_select,
    convergence_chart,
    ge_dataset_dropdown,
    ge_fusionreg_dropdown,
    ge_chart,
    correlation_chart,
    scatter_fusionreg_dropdown,
    scatter_param_dropdown,
    times_seen_threshold_slider,
    scatter_chart,
    sparsity_chart,
    summary_table,
):
    mo.ui.tabs(
        {
            "Convergence": mo.vstack(
                [
                    mo.hstack([conv_dataset_select, conv_fusionreg_select]),
                    convergence_chart,
                ]
            ),
            "GE Landscape": mo.hstack(
                [
                    mo.vstack([ge_dataset_dropdown, ge_fusionreg_dropdown]),
                    ge_chart,
                ],
                widths=[1, 3],
            ),
            "Param Correlation": mo.vstack(
                [times_seen_threshold_slider, correlation_chart]
            ),
            "Replicate Scatter": mo.hstack(
                [
                    scatter_chart,
                    mo.vstack(
                        [
                            scatter_fusionreg_dropdown,
                            scatter_param_dropdown,
                            times_seen_threshold_slider,
                        ]
                    ),
                ],
                widths=[3, 1],
            ),
            "Sparsity": sparsity_chart,
            "Summary": summary_table,
        },
        lazy=True,
    )
    return


if __name__ == "__main__":
    app.run()
