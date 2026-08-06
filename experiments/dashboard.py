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

    ge_table = mo.ui.table(
        _table_df,
        selection="single",
        label="GE Landscape fit",
        page_size=100,
        show_column_summaries=False,
    )
    conv_table = mo.ui.table(
        _table_df,
        selection="multi",
        label="Convergence fits",
        page_size=100,
        show_column_summaries=False,
    )
    sparsity_table = mo.ui.table(
        _table_df,
        selection="multi",
        label="Sparsity fits",
        page_size=100,
        show_column_summaries=False,
    )
    corr_table = mo.ui.table(
        _table_df,
        selection="multi",
        label="Correlation fit subset",
        page_size=100,
        show_column_summaries=False,
    )
    scatter_table = mo.ui.table(
        _table_df,
        selection="multi",
        label="Scatter fits (pick exactly 2)",
        page_size=100,
        show_column_summaries=False,
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


# --- Staged selections ---
#
# The multi-select tabs do not render on selection change; they render when
# their "Plot" button is pressed. The committed selection lives in mo.state
# so it survives unrelated cell re-runs, and chart cells reference ONLY the
# state getter -- never the table -- so checking rows recomputes nothing.
#
# A staged value of None means "never plotted"; that is distinct from an
# empty/short selection, which means "plotted, but nothing valid was picked".


@app.cell
def _(mo):
    get_conv_staged, set_conv_staged = mo.state(None)
    get_corr_staged, set_corr_staged = mo.state(None)
    get_scatter_staged, set_scatter_staged = mo.state(None)
    get_sparsity_staged, set_sparsity_staged = mo.state(None)
    get_scatter_param, set_scatter_param = mo.state(None)
    return (
        get_conv_staged,
        get_corr_staged,
        get_scatter_param,
        get_scatter_staged,
        get_sparsity_staged,
        set_conv_staged,
        set_corr_staged,
        set_scatter_param,
        set_scatter_staged,
        set_sparsity_staged,
    )


@app.cell
def _(
    conv_table,
    corr_table,
    mo,
    scatter_table,
    set_conv_staged,
    set_corr_staged,
    set_scatter_staged,
    set_sparsity_staged,
    sparsity_table,
    times_seen_threshold_slider,
):
    def _idx_list(table):
        """Positional fit indices of a table's selected rows."""
        _sel = table.value
        if _sel is None or _sel.empty:
            return []
        return list(_sel["_fit_idx"])

    # The setters MUST be called from on_change, not from a cell body.
    # A setter called inside a cell records that cell's id, and marimo's
    # resolve_state_updates then skips any cell that already ran after it --
    # which makes the rendered chart intermittently stale or empty. A widget
    # callback fires outside cell execution, so marimo records an
    # "__external__" sentinel that matches no cell, and every cell referencing
    # the getter re-runs deterministically.
    conv_plot_button = mo.ui.run_button(
        label="Plot",
        on_change=lambda _: set_conv_staged(_idx_list(conv_table)),
    )
    corr_plot_button = mo.ui.run_button(
        label="Plot",
        on_change=lambda _: set_corr_staged(
            (_idx_list(corr_table), times_seen_threshold_slider.value)
        ),
    )
    scatter_plot_button = mo.ui.run_button(
        label="Plot",
        on_change=lambda _: set_scatter_staged(
            (_idx_list(scatter_table), times_seen_threshold_slider.value)
        ),
    )
    sparsity_plot_button = mo.ui.run_button(
        label="Plot",
        on_change=lambda _: set_sparsity_staged(_idx_list(sparsity_table)),
    )
    return (
        conv_plot_button,
        corr_plot_button,
        scatter_plot_button,
        sparsity_plot_button,
    )


# ── D: Tab chart computations ────────────────────────────────────────────

# --- Convergence ---


@app.cell
def _(get_conv_staged, mc, mo, mplot):
    _idx = get_conv_staged()  # staged at press time; None until first press
    if _idx is None:
        convergence_chart = mo.md("Select fits, then press **Plot**.")
    elif not _idx:
        convergence_chart = mo.md(
            "No fits selected. Select at least one, then press **Plot**."
        )
    else:
        _conv_df = mc.convergence_trajectory_df(fit_indices=_idx)
        convergence_chart = mplot.convergence_trajectory(
            _conv_df,
            id_cols=["_fit_idx"],
            tooltip_cols=["dataset_name", "fusionreg"],
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
        _max_points = 5000

        def _sampled_variants(_vdf):
            if len(_vdf) > _max_points:
                return _vdf.sample(n=_max_points, random_state=0)
            return _vdf

        # Computed once and shared: the placement parameters do not depend on
        # `space`. Calling mplot.ge_landscape directly (rather than
        # _model.plot_ge_landscape) means params_df must be passed explicitly —
        # it defaults to None, which silently disables the annotation.
        _params = _model.get_ge_params_df()

        _v_fit, _curve_fit = _model.get_ge_landscape_df(space="fitness")
        _fit_chart = mplot.ge_landscape(
            _sampled_variants(_v_fit),
            _curve_fit,
            space="fitness",
            point_size=20,
            params_df=_params,
        )

        _v_fs, _curve_fs = _model.get_ge_landscape_df(space="func_score")
        _fs_chart = mplot.ge_landscape(
            _sampled_variants(_v_fs),
            _curve_fs,
            space="func_score",
            point_size=20,
            params_df=_params,
        )

        ge_chart = mo.vstack(
            [
                mo.md("**Fitness space** — g(φ)"),
                _fit_chart,
                mo.md(
                    "**Functional-score space** — " "α·(g(φ) − g(φ_wt)), per condition"
                ),
                _fs_chart,
            ]
        )
    return (ge_chart,)


# --- Correlation ---


@app.cell
def _(get_corr_staged, mc, mo, synthesize_isin_query):
    _staged = get_corr_staged()  # (fit indices, threshold), or None
    if _staged is None:
        correlation_chart = mo.md("Select fits and a threshold, then press **Plot**.")
    else:
        _idx, _threshold = _staged
        _src = mc.fit_models.reset_index(drop=True)
        _n_datasets = _src.iloc[_idx]["dataset_name"].nunique() if _idx else 0
        if _n_datasets < 2:
            correlation_chart = mo.md(
                "Select fits spanning ≥2 datasets, then press **Plot**."
            )
        else:
            _query = synthesize_isin_query(mc.fit_models, _idx)
            try:
                correlation_chart = mc.mut_param_dataset_correlation(
                    query=_query,
                    times_seen_threshold=_threshold,
                )
            except Exception as _e:
                correlation_chart = mo.md(
                    f"Correlation failed: {_e}. Try lowering the threshold or "
                    "selecting different fits."
                )
    return (correlation_chart,)


# --- Scatter ---


@app.cell
def _(
    common_param_columns,
    get_scatter_param,
    get_scatter_staged,
    mc,
    mo,
    set_scatter_param,
):
    _staged = get_scatter_staged()  # (fit indices, threshold), or None
    _pair = _staged[0] if _staged is not None else []

    # Names must NOT start with "_": marimo treats a single leading underscore
    # as cell-local, so an underscore-named value never reaches another cell.
    # Both frames are bound on every branch -- leaving one unbound would
    # unbind scatter_chart downstream and take the whole tab bar with it.
    scatter_muts_a = None
    scatter_muts_b = None

    if len(_pair) != 2:
        scatter_param_dropdown = mo.ui.dropdown(options=[], label="Parameter")
    else:
        _src = mc.fit_models.reset_index(drop=True)
        _ia, _ib = int(_pair[0]), int(_pair[1])
        _threshold = _staged[1]
        # Computed once here and reused by the chart cell below.
        scatter_muts_a = _src.iloc[_ia]["model"].get_mutations_df(
            times_seen_threshold=_threshold
        )
        scatter_muts_b = _src.iloc[_ib]["model"].get_mutations_df(
            times_seen_threshold=_threshold
        )
        _opts = common_param_columns(scatter_muts_a, scatter_muts_b)
        # This cell re-runs on every press, which would reset the dropdown to
        # its first option; keep the previous choice when it is still offered.
        _prev = get_scatter_param()
        _initial = _prev if _prev in _opts else (_opts[0] if _opts else None)
        scatter_param_dropdown = mo.ui.dropdown(
            options=_opts,
            value=_initial,
            label="Parameter",
            on_change=set_scatter_param,
        )
    scatter_param_dropdown
    return scatter_muts_a, scatter_muts_b, scatter_param_dropdown


@app.cell
def _(
    get_scatter_staged,
    merge_two_fits_on_mutation,
    mo,
    mplot,
    scatter_muts_a,
    scatter_muts_b,
    scatter_param_dropdown,
):
    _staged = get_scatter_staged()  # (fit indices, threshold), or None
    _pair = _staged[0] if _staged is not None else []

    if _staged is None:
        scatter_chart = mo.md("Select exactly 2 fits, then press **Plot**.")
    elif len(_pair) != 2:
        scatter_chart = mo.md(
            f"Selected {len(_pair)} fits; this comparison needs exactly 2. "
            "Adjust the selection, then press **Plot**."
        )
    elif scatter_param_dropdown.value is None:
        scatter_chart = mo.md("Select a parameter to compare.")
    else:
        # Reuses the frames from the dropdown cell -- no refetch, so changing
        # the parameter re-renders without another press.
        _ia, _ib = int(_pair[0]), int(_pair[1])
        _param = scatter_param_dropdown.value
        _merged, _x_col, _y_col = merge_two_fits_on_mutation(
            scatter_muts_a, scatter_muts_b, _param, key_a=_ia, key_b=_ib
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
def _(get_sparsity_staged, mc, mo, synthesize_isin_query):
    _idx = get_sparsity_staged()  # staged at press time; None until first press
    if _idx is None:
        sparsity_chart = mo.md("Select fits, then press **Plot**.")
    else:
        _src = mc.fit_models.reset_index(drop=True)
        _n_fr = _src.iloc[_idx]["fusionreg"].nunique() if _idx else 0
        if _n_fr < 2:
            sparsity_chart = mo.md(
                "Select fits spanning ≥2 fusionreg values, then press **Plot**."
            )
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
    conv_plot_button,
    conv_table,
    convergence_chart,
    corr_plot_button,
    corr_table,
    correlation_chart,
    ge_chart,
    ge_table,
    mo,
    scatter_chart,
    scatter_param_dropdown,
    scatter_plot_button,
    scatter_table,
    sparsity_chart,
    sparsity_plot_button,
    sparsity_table,
    summary_table,
    table_caption,
    times_seen_threshold_slider,
):
    # Each multi-select tab pairs its table with a Plot button. On the two
    # tabs that stage the threshold, the slider sits beside the button to
    # signal that it is an input to the press rather than a live control.
    mo.ui.tabs(
        {
            "Convergence": mo.vstack(
                [table_caption, conv_table, conv_plot_button, convergence_chart]
            ),
            "GE Landscape": mo.vstack([table_caption, ge_table, ge_chart]),
            "Param Correlation": mo.vstack(
                [
                    table_caption,
                    corr_table,
                    mo.hstack([times_seen_threshold_slider, corr_plot_button]),
                    correlation_chart,
                ]
            ),
            "Replicate Scatter": mo.vstack(
                [
                    table_caption,
                    scatter_table,
                    mo.hstack([times_seen_threshold_slider, scatter_plot_button]),
                    scatter_param_dropdown,
                    scatter_chart,
                ]
            ),
            "Sparsity": mo.vstack(
                [table_caption, sparsity_table, sparsity_plot_button, sparsity_chart]
            ),
            "Summary": summary_table,
        },
        lazy=True,
    )
    return


if __name__ == "__main__":
    app.run()
