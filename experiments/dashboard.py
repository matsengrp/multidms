"""Interactive marimo dashboard for exploring ModelCollection results."""

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
    _repo_root = _dashboard_dir.parent

    # Scan for fit_collection.pkl files
    _pkl_paths = sorted(_dashboard_dir.glob("*/results*/fit_collection.pkl"))

    # Build mapping: display label -> path
    discovered_collections = {}
    for p in _pkl_paths:
        # e.g. "simulation/results" or "scv2-spike/results-test"
        label = f"{p.parent.parent.name}/{p.parent.name}"
        discovered_collections[label] = p

    return discovered_collections, _repo_root


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
    mo.stop(not pipeline_dropdown.value, mo.md("Select a pipeline above."))

    _pkl_path = discovered_collections[pipeline_dropdown.value]
    with open(_pkl_path, "rb") as _f:
        mc = pickle.load(_f)

    _n_models = len(mc.fit_models)
    _datasets = list(mc.fit_models["dataset_name"].unique())
    _fusionregs = sorted(mc.fit_models["fusionreg"].unique())

    mo.md(
        f"Loaded **{_n_models}** models from `{pipeline_dropdown.value}` "
        f"| datasets: {_datasets} | fusionreg values: {_fusionregs}"
    )
    return (mc,)


# ── C: Tab Controls + Charts ─────────────────────────────────────────────

# --- Tab 1: Convergence ---


@app.cell
def _(mo, mc):
    _datasets = list(mc.fit_models["dataset_name"].unique())
    _fusionregs = [str(f) for f in sorted(mc.fit_models["fusionreg"].unique())]

    conv_dataset_select = mo.ui.multiselect(
        options=_datasets,
        value=_datasets[:2],
        label="Datasets",
    )
    conv_fusionreg_select = mo.ui.multiselect(
        options=_fusionregs,
        value=_fusionregs[:1],
        label="Fusion reg",
    )
    return conv_dataset_select, conv_fusionreg_select


@app.cell
def _(mo, mc, conv_dataset_select, conv_fusionreg_select):
    import multidms.plot as mplot

    mo.stop(
        not conv_dataset_select.value or not conv_fusionreg_select.value,
        mo.md("Select at least one dataset and one fusionreg value."),
    )

    _ds = conv_dataset_select.value
    _fr = [float(x) for x in conv_fusionreg_select.value]
    _query = f"dataset_name.isin({_ds}) and fusionreg.isin({_fr})"

    conv_df = mc.convergence_trajectory_df(query=_query)
    convergence_chart = mplot.convergence_trajectory(
        conv_df, id_cols=["dataset_name", "fusionreg"]
    )
    convergence_chart
    return mplot, convergence_chart


# --- Tab 2: GE Landscape ---


@app.cell
def _(mo, mc):
    _datasets = list(mc.fit_models["dataset_name"].unique())
    _fusionregs = [str(f) for f in sorted(mc.fit_models["fusionreg"].unique())]

    ge_dataset_dropdown = mo.ui.dropdown(
        options=_datasets,
        value=_datasets[0],
        label="Dataset",
    )
    ge_fusionreg_dropdown = mo.ui.dropdown(
        options=_fusionregs,
        value=_fusionregs[0],
        label="Fusion reg",
    )
    return ge_dataset_dropdown, ge_fusionreg_dropdown


@app.cell
def _(mo, mc, mplot, ge_dataset_dropdown, ge_fusionreg_dropdown):
    mo.stop(
        not ge_dataset_dropdown.value or not ge_fusionreg_dropdown.value,
        mo.md("Select a dataset and fusionreg."),
    )

    _ds = ge_dataset_dropdown.value
    _fr = float(ge_fusionreg_dropdown.value)
    _row = mc.fit_models.query("dataset_name == @_ds and fusionreg == @_fr").iloc[0]
    _model = _row["model"]
    _variants_df, _ge_curve_df = _model.get_ge_landscape_df()
    ge_chart = mplot.ge_landscape(_variants_df, _ge_curve_df)
    ge_chart
    return (ge_chart,)


# --- Tab 3: Parameter Correlation ---


@app.cell
def _(mo, mc):
    _n_datasets = len(mc.fit_models["dataset_name"].unique())
    if _n_datasets < 2:
        correlation_chart = mo.md("Need at least 2 datasets for correlation analysis.")
    else:
        correlation_chart = mc.mut_param_dataset_correlation()
    correlation_chart
    return (correlation_chart,)


# --- Tab 4: Replicate Scatter ---


@app.cell
def _(mo, mc):
    _datasets = list(mc.fit_models["dataset_name"].unique())
    _fusionregs = [str(f) for f in sorted(mc.fit_models["fusionreg"].unique())]
    _param_types = ["beta", "shift", "predicted_func_score"]

    scatter_dataset_select = mo.ui.multiselect(
        options=_datasets,
        value=_datasets[:2] if len(_datasets) >= 2 else _datasets,
        label="Datasets (select exactly 2)",
    )
    scatter_fusionreg_dropdown = mo.ui.dropdown(
        options=_fusionregs,
        value=_fusionregs[0],
        label="Fusion reg",
    )
    scatter_param_dropdown = mo.ui.dropdown(
        options=_param_types,
        value="beta",
        label="Parameter",
    )
    return scatter_dataset_select, scatter_fusionreg_dropdown, scatter_param_dropdown


@app.cell
def _(
    mo,
    mc,
    mplot,
    scatter_dataset_select,
    scatter_fusionreg_dropdown,
    scatter_param_dropdown,
):
    mo.stop(
        not scatter_dataset_select.value,
        mo.md("Select at least one dataset."),
    )
    mo.stop(
        len(scatter_dataset_select.value) != 2,
        mo.md("Select exactly 2 datasets for scatter comparison."),
    )

    _ds = scatter_dataset_select.value
    _fr = float(scatter_fusionreg_dropdown.value)
    _param = scatter_param_dropdown.value

    _query = f"fusionreg == {_fr}"
    _muts_df = mc.split_apply_combine_muts(
        groupby=("dataset_name", "fusionreg"),
        query=_query,
    ).reset_index()

    # Pivot to get one column per dataset for the chosen parameter
    _conditions = mc.conditions
    _param_cols = [c for c in _muts_df.columns if c.startswith(_param)]
    _keep_cols = ["mutation", "dataset_name", "fusionreg"] + _param_cols

    # For each condition param column, pivot datasets side by side
    _d0, _d1 = _ds[0], _ds[1]
    _df0 = _muts_df.query("dataset_name == @_d0")[["mutation"] + _param_cols]
    _df1 = _muts_df.query("dataset_name == @_d1")[["mutation"] + _param_cols]

    # Use the first param column (reference condition beta, or first shift)
    _col = _param_cols[0]
    _x_label = f"{_col} ({_d0})"
    _y_label = f"{_col} ({_d1})"

    _merged = _df0[["mutation", _col]].merge(
        _df1[["mutation", _col]], on="mutation", suffixes=(f"_{_d0}", f"_{_d1}")
    )
    _x_col = f"{_col}_{_d0}"
    _y_col = f"{_col}_{_d1}"

    scatter_chart = mplot.replicate_param_scatter(
        _merged,
        x_col=_x_col,
        y_col=_y_col,
        x_label=_x_label,
        y_label=_y_label,
    )
    scatter_chart
    return (scatter_chart,)


# --- Tab 5: Sparsity ---


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
    sparsity_chart
    return (sparsity_chart,)


# ── D: Summary ───────────────────────────────────────────────────────────


@app.cell
def _(mo, mc, pd):
    _rows = []
    for _, _fit in mc.fit_models.iterrows():
        _rows.append(
            {
                "dataset": _fit["dataset_name"],
                "fusionreg": _fit["fusionreg"],
                "converged": _fit.get("converged", "N/A"),
                "fit_time": _fit.get("fit_time", "N/A"),
                "ge_type": _fit.get("ge_type", "N/A"),
            }
        )
    summary_df = pd.DataFrame(_rows)
    summary_table = mo.ui.table(summary_df)
    summary_table
    return summary_df, summary_table


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
    scatter_dataset_select,
    scatter_fusionreg_dropdown,
    scatter_param_dropdown,
    scatter_chart,
    sparsity_chart,
    summary_table,
):
    _tabs = mo.ui.tabs(
        {
            "Convergence": mo.hstack(
                [
                    mo.vstack([conv_dataset_select, conv_fusionreg_select]),
                    convergence_chart,
                ],
                widths=[1, 3],
            ),
            "GE Landscape": mo.hstack(
                [
                    mo.vstack([ge_dataset_dropdown, ge_fusionreg_dropdown]),
                    ge_chart,
                ],
                widths=[1, 3],
            ),
            "Param Correlation": correlation_chart,
            "Replicate Scatter": mo.hstack(
                [
                    mo.vstack(
                        [
                            scatter_dataset_select,
                            scatter_fusionreg_dropdown,
                            scatter_param_dropdown,
                        ]
                    ),
                    scatter_chart,
                ],
                widths=[1, 3],
            ),
            "Sparsity": sparsity_chart,
        }
    )

    mo.vstack([_tabs, mo.md("## Model Summary"), summary_table])
    return


if __name__ == "__main__":
    app.run()
