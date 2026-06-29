"""Interactive marimo dashboard for exploring ModelCollection results.

Discovers every ``*.pkl`` found below the directory the dashboard is launched
from, so it can explore any fitted ``ModelCollection`` regardless of how it was
produced (pipeline run or otherwise). Each pickle is validated lazily when
selected; non-collection pickles surface a friendly inline error rather than
crashing the app.

Each interactive tab carries its own selectable fit table so the user can
switch which model(s) are plotted across the full hyperparameter grid
(fusionreg, ge_type, l2reg, scale_fusion_by_n, …). The Param Correlation and
Replicate Scatter tabs additionally support a ``times_seen_threshold`` slider.
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
    from pathlib import Path

    import pandas as pd

    return Path, pd


@app.cell
def _():
    # marimo always puts the notebook's own directory (``experiments/``) on
    # ``sys.path``, so the sibling helper module is importable as a top-level
    # module regardless of the directory the dashboard was launched from.
    # Do NOT use ``from experiments.dashboard_helpers import ...`` — that only
    # resolves when the repo root happens to be on the path (i.e. launched
    # from the repo root), which breaks the core use case of launching from
    # an arbitrary results directory below cwd.
    from dashboard_helpers import (
        common_param_columns,
        constant_summary,
        discover_pickles,
        display_table_df,
        load_collection,
        merge_two_fits_on_mutation,
        synthesize_isin_query,
    )

    return (
        common_param_columns,
        constant_summary,
        discover_pickles,
        display_table_df,
        load_collection,
        merge_two_fits_on_mutation,
        synthesize_isin_query,
    )


@app.cell
def _(mo):
    mo.md(
        """
        # multidms Dashboard

        Interactive exploration of any `ModelCollection` result. Discovers
        every `*.pkl` below the directory the dashboard was launched from;
        each is validated when you select it.
        """
    )
    return


# ── B: Discovery + Loading ───────────────────────────────────────────────


@app.cell
def _(Path, discover_pickles, mo):
    discovered_collections = discover_pickles(Path.cwd())

    if not discovered_collections:
        _cwd = Path.cwd()
        mo.stop(
            True,
            mo.md(
                f"**No `*.pkl` found** below the current directory "
                f"(`{_cwd}`). Launch the dashboard from a directory that "
                f"contains a fitted `ModelCollection` pickle."
            ),
        )

    pipeline_dropdown = mo.ui.dropdown(
        options=list(discovered_collections.keys()),
        value=list(discovered_collections.keys())[0],
        label="Pickle",
    )
    pipeline_dropdown
    return discovered_collections, pipeline_dropdown


@app.cell
def _(discovered_collections, load_collection, mo, pipeline_dropdown):
    mo.stop(not pipeline_dropdown.value, mo.md("Select a pickle above."))

    _pkl_path = discovered_collections[pipeline_dropdown.value]
    mc = None
    try:
        mc = load_collection(_pkl_path)
    except Exception as _e:
        mo.stop(
            True,
            mo.md(
                f"**Could not load `{pipeline_dropdown.value}` as a "
                f"`ModelCollection`.** It loaded or coerced with this error:\n\n"
                f"```\n{_e}\n```\n\n"
                f"Pick a different pickle above."
            ),
        )

    _n_models = len(mc.fit_models)
    _datasets = list(mc.fit_models["dataset_name"].unique())
    mo.md(
        f"Loaded **{_n_models}** fits from `{pipeline_dropdown.value}` "
        f"| datasets: {_datasets}"
    )
    return (mc,)


@app.cell
def _():
    import multidms.plot as mplot

    return (mplot,)


# ── C: Per-tab fit tables + sliders ──────────────────────────────────────


@app.cell
def _(constant_summary, display_table_df, mc, mo):
    _const = constant_summary(mc.fit_models)
    _caption = (
        "constant across all fits: " + ", ".join(f"{k}={v}" for k, v in _const.items())
        if _const
        else "all displayed columns vary across fits"
    )

    _table_df = display_table_df(mc.fit_models)

    ge_table = mo.ui.table(_table_df, selection="single", label="GE Landscape fit")
    conv_table = mo.ui.table(_table_df, selection="multi", label="Convergence fits")
    sparsity_table = mo.ui.table(_table_df, selection="multi", label="Sparsity fits")
    corr_table = mo.ui.table(
        _table_df, selection="multi", label="Correlation fit subset"
    )
    scatter_table = mo.ui.table(
        _table_df, selection="multi", label="Scatter fits (pick exactly 2)"
    )
    table_caption = mo.md(f"*{_caption}*")
    return (
        conv_table,
        corr_table,
        ge_table,
        scatter_table,
        sparsity_table,
        table_caption,
    )


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
def _(conv_table, mc, mo, mplot, synthesize_isin_query):
    _sel = conv_table.value  # a DataFrame of selected rows (may be empty)
    if _sel is None or _sel.empty:
        convergence_chart = mo.md("Select at least one fit.")
    else:
        _idx = list(_sel["_fit_idx"])
        _query = synthesize_isin_query(mc.fit_models, _idx)
        _conv_df = mc.convergence_trajectory_df(query=_query)
        convergence_chart = mplot.convergence_trajectory(
            _conv_df, id_cols=["dataset_name", "fusionreg"]
        )
    return (convergence_chart,)


# --- GE Landscape ---


@app.cell
def _(ge_table, mc, mo, mplot):
    _sel = ge_table.value  # a DataFrame of selected rows (may be empty)
    if _sel is None or len(_sel) != 1:
        ge_chart = mo.md("Select a fit.")
    else:
        _row = mc.fit_models.reset_index(drop=True).iloc[int(_sel["_fit_idx"].iloc[0])]
        _model = _row["model"]
        _variants_df, _ge_curve_df = _model.get_ge_landscape_df()
        _max_points = 5000
        if len(_variants_df) > _max_points:
            _variants_df = _variants_df.sample(n=_max_points, random_state=0)
        ge_chart = mplot.ge_landscape(_variants_df, _ge_curve_df, point_size=20)
    return (ge_chart,)


# --- Correlation ---


@app.cell
def _(corr_table, mc, mo, synthesize_isin_query, times_seen_threshold_slider):
    _sel = corr_table.value  # a DataFrame of selected rows (may be empty)
    _idx = list(_sel["_fit_idx"]) if _sel is not None and not _sel.empty else []
    _src = mc.fit_models.reset_index(drop=True)
    _n_datasets = _src.iloc[_idx]["dataset_name"].nunique() if _idx else 0
    if _n_datasets < 2:
        correlation_chart = mo.md("Select fits spanning ≥2 datasets.")
    else:
        _query = synthesize_isin_query(mc.fit_models, _idx)
        try:
            correlation_chart = mc.mut_param_dataset_correlation(
                query=_query,
                times_seen_threshold=times_seen_threshold_slider.value,
            )
        except Exception as _e:
            correlation_chart = mo.md(
                f"Correlation failed: {_e}. Try lowering the threshold or "
                "selecting different fits."
            )
    return (correlation_chart,)


# --- Scatter ---


@app.cell
def _(common_param_columns, mc, mo, scatter_table, times_seen_threshold_slider):
    _sel = scatter_table.value  # a DataFrame of selected rows (may be empty)
    if _sel is None or len(_sel) != 2:
        scatter_param_dropdown = mo.ui.dropdown(options=[], label="Parameter")
    else:
        _src = mc.fit_models.reset_index(drop=True)
        _ia, _ib = int(_sel["_fit_idx"].iloc[0]), int(_sel["_fit_idx"].iloc[1])
        _muts_a = _src.iloc[_ia]["model"].get_mutations_df(
            times_seen_threshold=times_seen_threshold_slider.value
        )
        _muts_b = _src.iloc[_ib]["model"].get_mutations_df(
            times_seen_threshold=times_seen_threshold_slider.value
        )
        _opts = common_param_columns(_muts_a, _muts_b)
        scatter_param_dropdown = mo.ui.dropdown(
            options=_opts,
            value=_opts[0] if _opts else None,
            label="Parameter",
        )
    scatter_param_dropdown
    return (scatter_param_dropdown,)


@app.cell
def _(
    mc,
    merge_two_fits_on_mutation,
    mo,
    mplot,
    scatter_param_dropdown,
    scatter_table,
    times_seen_threshold_slider,
):
    _sel = scatter_table.value  # a DataFrame of selected rows (may be empty)
    if _sel is None or len(_sel) != 2:
        scatter_chart = mo.md("Select exactly 2 fits.")
    elif scatter_param_dropdown.value is None:
        scatter_chart = mo.md("Select a parameter to compare.")
    else:
        _ia, _ib = int(_sel["_fit_idx"].iloc[0]), int(_sel["_fit_idx"].iloc[1])
        _src = mc.fit_models.reset_index(drop=True)
        _muts_a = _src.iloc[_ia]["model"].get_mutations_df(
            times_seen_threshold=times_seen_threshold_slider.value
        )
        _muts_b = _src.iloc[_ib]["model"].get_mutations_df(
            times_seen_threshold=times_seen_threshold_slider.value
        )
        _param = scatter_param_dropdown.value
        _merged, _x_col, _y_col = merge_two_fits_on_mutation(
            _muts_a, _muts_b, _param, key_a=_ia, key_b=_ib
        )
        scatter_chart = mplot.replicate_param_scatter(
            _merged,
            x_col=_x_col,
            y_col=_y_col,
            x_label=f"{_param} (fit {_ia})",
            y_label=f"{_param} (fit {_ib})",
        )
    return (scatter_chart,)


# --- Sparsity ---


@app.cell
def _(mc, mo, sparsity_table, synthesize_isin_query):
    _sel = sparsity_table.value  # a DataFrame of selected rows (may be empty)
    _idx = list(_sel["_fit_idx"]) if _sel is not None and not _sel.empty else []
    _src = mc.fit_models.reset_index(drop=True)
    _n_fr = _src.iloc[_idx]["fusionreg"].nunique() if _idx else 0
    if _n_fr < 2:
        sparsity_chart = mo.md("Select fits spanning ≥2 fusionreg values.")
    else:
        _query = synthesize_isin_query(mc.fit_models, _idx)
        try:
            sparsity_chart = mc.shift_sparsity(query=_query)
        except Exception as _e:
            sparsity_chart = mo.md(f"Sparsity chart failed: {_e}")
    return (sparsity_chart,)


# --- Summary ---


@app.cell
def _(mc, mo, pd):
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
    conv_table,
    convergence_chart,
    corr_table,
    correlation_chart,
    ge_chart,
    ge_table,
    mo,
    scatter_chart,
    scatter_param_dropdown,
    scatter_table,
    sparsity_chart,
    sparsity_table,
    summary_table,
    table_caption,
    times_seen_threshold_slider,
):
    mo.ui.tabs(
        {
            "Convergence": mo.vstack([table_caption, conv_table, convergence_chart]),
            "GE Landscape": mo.vstack([table_caption, ge_table, ge_chart]),
            "Param Correlation": mo.vstack(
                [
                    table_caption,
                    corr_table,
                    times_seen_threshold_slider,
                    correlation_chart,
                ]
            ),
            "Replicate Scatter": mo.vstack(
                [
                    table_caption,
                    scatter_table,
                    mo.hstack([scatter_param_dropdown, times_seen_threshold_slider]),
                    scatter_chart,
                ]
            ),
            "Sparsity": mo.vstack([table_caption, sparsity_table, sparsity_chart]),
            "Summary": summary_table,
        },
        lazy=True,
    )
    return


if __name__ == "__main__":
    app.run()
