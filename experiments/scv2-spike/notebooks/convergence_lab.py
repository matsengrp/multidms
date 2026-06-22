"""Convergence lab: fixed-scale + warmstart validation (#246).

A fast, local marimo notebook that reproduces the scv2-spike convergence
pathology on full real data and scores the recompute_scale fix on the metric
that matters — replicate-shift correlation — not just iteration counts.

Run:  pixi run marimo edit experiments/scv2-spike/notebooks/convergence_lab.py
"""

import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import os
    import pickle
    import warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore")

    import multidms
    from multidms.model_collection import fit_models, ModelCollection
    from multidms.utils import explode_params_dict

    # Resolve paths relative to this notebook (mirrors experiments/dashboard.py),
    # so the notebook runs regardless of the process working directory.
    NB_DIR = Path(os.path.abspath(__file__)).parent
    SPIKE_DIR = NB_DIR.parent  # experiments/scv2-spike

    return (
        NB_DIR,
        SPIKE_DIR,
        pickle,
        np,
        pd,
        multidms,
        fit_models,
        ModelCollection,
        explode_params_dict,
    )


@app.cell
def _(SPIKE_DIR, multidms, pd):
    # ── Cell 1: load full real spike data → one multidms.Data per replicate ──
    DATA_CSV = (
        SPIKE_DIR
        / "results-prod-235-times-seen-threshold"
        / "training_functional_scores.csv"
    )
    REF = "Omicron_BA1"
    raw = pd.read_csv(DATA_CSV)
    raw["aa_substitutions"] = raw["aa_substitutions"].fillna("")

    rep_data = {}
    for rep in sorted(raw["replicate"].unique()):
        sub = raw[raw["replicate"] == rep][
            ["condition", "aa_substitutions", "func_score"]
        ].copy()
        d = multidms.Data(
            sub,
            reference=REF,
            alphabet=multidms.AAS_WITHSTOP,
            name=f"rep_{rep}",
            verbose=False,
        )
        rep_data[f"rep_{rep}"] = d

    return DATA_CSV, REF, rep_data


@app.cell
def _(SPIKE_DIR, explode_params_dict, fit_models, pickle, rep_data):
    # ── Cell 2: 2×2×3×2 = 24-fit factorial via fit_models, cached to pickle ──
    CACHE = SPIKE_DIR / "results" / "convergence_lab" / "fit_collection.pkl"

    FUSIONREG = [0.0, 8e-5, 3.2e-4]  # spans prod grid; 8e-5 = worst regression
    BLOCK_ITERS = 50

    if CACHE.exists():
        with open(CACHE, "rb") as fh:
            fit_collection_df = pickle.load(fh)
    else:
        # recompute_scale + warmstart are the factorial axes; dataset (replicate)
        # and fusionreg explode within each arm. Everything else mirrors prod
        # (_common.build_fit_params).
        params = {
            "dataset": list(rep_data.values()),
            "recompute_scale": [True, False],
            "warmstart": [True, False],
            "fusionreg": FUSIONREG,
            "maxiter": [BLOCK_ITERS],
            "tol": [1e-6],
            "l2reg": [0.0],
            "beta0_ridge": [0.0],
            "scale_fusion_by_n": [False],
            "ge_type": ["Sigmoid"],
            "share_alpha": [True],
            "alpha_init": [None],
            "beta0_init": [None],
            "beta_clip_range": [None],
            "ge_kwargs": [dict(tol=1e-4, maxiter=50, maxls=40, jit=True)],
            "cal_kwargs": [dict(tol=1e-4, maxiter=50, maxls=40, jit=True)],
            "loss_kwargs": [dict(δ=1.0)],
        }
        assert len(explode_params_dict(params)) == 24  # 2*2*3*2 factorial
        # n_processes=1: fit_models spawns workers via get_context("spawn"),
        # which re-imports the caller as __main__. A marimo notebook (like any
        # non-__main__-guarded module) would recursively re-execute under spawn,
        # so run the 24 fits single-process here. Slower but correct in-notebook.
        n_fit, n_failed, fit_collection_df = fit_models(
            params, n_processes=1, failures="tolerate"
        )
        # dict-valued columns → str for groupby compatibility (prod convention).
        for col in fit_collection_df.columns:
            if fit_collection_df[col].apply(lambda x: isinstance(x, dict)).any():
                fit_collection_df[col] = fit_collection_df[col].apply(str)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE, "wb") as fh:
            pickle.dump(fit_collection_df, fh)

    return BLOCK_ITERS, CACHE, FUSIONREG, fit_collection_df


@app.cell
def _(ModelCollection, fit_collection_df, pd):
    # ── Cell 3: metric ② — replicate-shift correlation per (scale, warmstart) ──
    # Build one ModelCollection per (recompute_scale, warmstart) arm so the two
    # replicates inside it pair up via itertools.combinations, then reuse the
    # canonical mut_param_dataset_correlation (model_collection.py).
    corr_frames = []
    arm_cols = ["recompute_scale", "warmstart"]
    for (recompute, warm), arm_df in fit_collection_df.groupby(arm_cols):
        mc = ModelCollection(arm_df.reset_index(drop=True))
        _, rep_df = mc.mut_param_dataset_correlation(
            x="fusionreg", return_data=True, r=1
        )
        # Headline metric is replicate-SHIFT correlation only (Delta, BA2; BA1 is
        # the reference and has no shift param).
        rep_df = rep_df[rep_df["mut_param"].str.startswith("shift")].copy()
        rep_df["recompute_scale"] = recompute
        rep_df["warmstart"] = warm
        rep_df["cell"] = (
            f"scale={'recompute' if recompute else 'fixed'}, "
            f"warmstart={'on' if warm else 'off'}"
        )
        corr_frames.append(rep_df)

    corr_df = pd.concat(corr_frames, ignore_index=True)
    return (corr_df,)


@app.cell
def _(fit_collection_df, np, pd):
    # ── Cell 4: metric ① — convergence, derived per ablation.py:72–81 ──
    BLOCK_TOL = 1e-6
    conv_rows = []
    for _, fit in fit_collection_df.iterrows():
        traj = fit["model"].convergence_trajectory_df
        if traj is None or len(traj) == 0:
            continue
        obj = traj["objective_total_trajectory"].to_numpy()
        diffs = np.diff(obj)
        inc = int((diffs > 1e-9 * np.abs(obj[:-1])).sum()) if len(diffs) else 0
        final_err = float(traj["objective_error_trajectory"].iloc[-1])
        conv_rows.append(
            {
                "dataset_name": fit["dataset_name"],
                "recompute_scale": fit["recompute_scale"],
                "warmstart": fit["warmstart"],
                "fusionreg": fit["fusionreg"],
                "converged": bool(final_err < BLOCK_TOL),
                "final_obj_error": final_err,
                "iters": len(traj),
                "inc": inc,
                "alpha": float(fit["model"].params.α),
            }
        )
    conv_df = pd.DataFrame(conv_rows).sort_values(
        ["recompute_scale", "warmstart", "fusionreg", "dataset_name"]
    )
    return BLOCK_TOL, conv_df


@app.cell
def _(corr_df, mo):
    # ── Cell 5: DECISIVE plot — replicate-shift correlation vs fusionreg ──
    import altair as alt

    decisive_chart = (
        alt.Chart(corr_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("fusionreg:Q", scale=alt.Scale(type="symlog")),
            y=alt.Y(
                "correlation:Q",
                title="replicate-shift correlation (Pearson)",
            ),
            color=alt.Color("cell:N", title="scale × warmstart"),
            strokeDash=alt.StrokeDash("mut_param:N", title="shift param"),
            tooltip=["cell", "mut_param", "fusionreg", "correlation"],
        )
        .properties(
            width=500,
            height=400,
            title="Metric ②: does fixed-scale keep replicate-shift "
            "correlation high as fusionreg grows?",
        )
    )
    mo.ui.altair_chart(decisive_chart)
    return alt, decisive_chart


@app.cell
def _(conv_df, corr_df, mo):
    # ── Cell 5b: tables alongside the decisive plot ──
    mo.vstack(
        [
            mo.md("### Metric ① — convergence per cell"),
            mo.ui.table(conv_df, selection=None),
            mo.md("### Metric ② — replicate-shift correlation per cell"),
            mo.ui.table(corr_df, selection=None),
        ]
    )
    return


@app.cell
def _(fit_collection_df, mo):
    # ── Cell 6: reactive diagnostic — α / β0 drift / predicted floor ──
    diag_opts = {
        f"{r['dataset_name']} | scale="
        f"{'recompute' if r['recompute_scale'] else 'fixed'} | "
        f"warmstart={'on' if r['warmstart'] else 'off'} | "
        f"fr={r['fusionreg']:.1e}": i
        for i, r in fit_collection_df.reset_index(drop=True).iterrows()
    }
    fit_selector = mo.ui.dropdown(
        options=diag_opts, value=list(diag_opts)[0], label="fit"
    )
    fit_selector
    return diag_opts, fit_selector


@app.cell
def _(alt, fit_collection_df, fit_selector, mo, pd):
    sel = fit_collection_df.reset_index(drop=True).iloc[fit_selector.value]
    sel_model = sel["model"]
    sel_traj = sel_model.convergence_trajectory_df

    # α-trajectory.
    alpha_chart = (
        alt.Chart(sel_traj)
        .mark_line()
        .encode(x="iteration:Q", y=alt.Y("alpha:Q", title="α"))
        .properties(width=320, height=180, title="α trajectory")
    )
    # β0-drift (one β0 column per condition: beta0_<cond>).
    beta0_cols = [c for c in sel_traj.columns if c.startswith("beta0_")]
    beta0_long = sel_traj.melt(
        id_vars="iteration",
        value_vars=beta0_cols,
        var_name="condition",
        value_name="beta0",
    )
    beta0_chart = (
        alt.Chart(beta0_long)
        .mark_line()
        .encode(x="iteration:Q", y="beta0:Q", color="condition:N")
        .properties(width=320, height=180, title="β0 drift per condition")
    )

    # Per-condition predicted functional-score FLOOR (min predicted score).
    # NOTE get_variants_df() returns ALL conditions with a `condition` column —
    # it takes no `condition=` arg. This compares Delta's predicted floor to
    # BA1/BA2's predicted floors (NOT to the −3.5 input-target clip; #246).
    variants_df = sel_model.get_variants_df()
    floor_df = (
        variants_df.groupby("condition")["predicted_func_score"]
        .min()
        .reset_index()
        .rename(columns={"predicted_func_score": "predicted_floor"})
    )
    floor_chart = (
        alt.Chart(floor_df)
        .mark_bar()
        .encode(
            x="condition:N",
            y=alt.Y("predicted_floor:Q", title="min predicted func_score"),
            color="condition:N",
        )
        .properties(
            width=320,
            height=180,
            title="Predicted floor per condition (Delta vs BA1/BA2)",
        )
    )
    mo.hstack([alpha_chart, beta0_chart, floor_chart])
    return (
        alpha_chart,
        beta0_chart,
        beta0_cols,
        beta0_long,
        floor_chart,
        floor_df,
        sel,
        sel_model,
        sel_traj,
        variants_df,
    )


@app.cell
def _(mo):
    alpha_probe_on = mo.ui.switch(
        label="α-bound probe: re-fit best cell with α steered away from −5"
    )
    alpha_probe_on
    return (alpha_probe_on,)


@app.cell
def _(alpha_probe_on, conv_df, mo, multidms, pd, rep_data):
    # ── Cell 7: α-bound probe on the best-converging cell ──
    best = None
    probe_model = None
    probe_variants = None
    probe_floor_df = None
    if not alpha_probe_on.value:
        probe_out = mo.md("Toggle the switch above to run the α-bound probe.")
    else:
        # Pick the best-converging cell (lowest final_obj_error, converged).
        converged_cells = conv_df[conv_df["converged"]]
        if len(converged_cells) == 0:
            converged_cells = conv_df  # fall back to least-bad
        best = converged_cells.sort_values("final_obj_error").iloc[0]
        probe_data = rep_data[best["dataset_name"]]
        # fit() has no direct α clamp; steer via alpha_init + beta_clip_range,
        # the levers fit() exposes. If α still collapses, that localizes the
        # degeneracy to a place fit() cannot currently constrain (a result).
        probe_model = multidms.Model(
            probe_data,
            ge_type="Sigmoid",
            fusionreg=float(best["fusionreg"]),
        )
        probe_model.fit(
            maxiter=50,
            tol=1e-6,
            warmstart=True,
            recompute_scale=False,
            alpha_init=1.0,
            beta_clip_range=(-10.0, 10.0),
            verbose=False,
        )
        probe_variants = probe_model.get_variants_df()
        probe_floor_df = (
            probe_variants.groupby("condition")["predicted_func_score"]
            .min()
            .reset_index()
            .rename(columns={"predicted_func_score": "predicted_floor"})
        )
        probe_floor_df["alpha"] = float(probe_model.params.α)
        probe_out = mo.vstack(
            [
                mo.md(
                    f"**Probe** re-fit `{best['dataset_name']}` "
                    f"fr={best['fusionreg']:.1e} with α-steering — "
                    f"resulting α = **{float(probe_model.params.α):.2f}**"
                ),
                mo.ui.table(probe_floor_df, selection=None),
            ]
        )
    probe_out
    return best, probe_floor_df, probe_model, probe_out, probe_variants


if __name__ == "__main__":
    app.run()
