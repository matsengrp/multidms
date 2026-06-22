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
        n_models = len(explode_params_dict(params))  # 2*2*3*2 = 24
        n_fit, n_failed, fit_collection_df = fit_models(
            params, n_processes=min(6, n_models), failures="tolerate"
        )
        # dict-valued columns → str for groupby compatibility (prod convention).
        for col in fit_collection_df.columns:
            if fit_collection_df[col].apply(lambda x: isinstance(x, dict)).any():
                fit_collection_df[col] = fit_collection_df[col].apply(str)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE, "wb") as fh:
            pickle.dump(fit_collection_df, fh)

    return BLOCK_ITERS, CACHE, FUSIONREG, fit_collection_df


if __name__ == "__main__":
    app.run()
