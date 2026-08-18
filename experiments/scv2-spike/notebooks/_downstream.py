"""Shared utilities for the spike manuscript-figure notebook.

This module is a deliberate sibling of ``_common.py`` rather than an addition
to it. ``_common.py`` is declared ``input:`` on all four fit-tier rules, so a
helper added there would make an edit to figure code invalidate the cached
model fit -- a 2 h 20 min refit, as measured on the prod run logged in
``experiments/scv2-spike/README.md``. This module is declared ``input:`` on
``rule manuscript_figures`` only, which keeps figure work in the downstream
tier where it belongs. See ``experiments/scv2-spike/README.md`` ("Configuration
tiers") and issue #287.
"""

import os
import time
import warnings

import matplotlib.pyplot as plt
import multidms
import pandas as pd
import requests

# The legacy analysis repository, pinned to the commit the rest of EPIC #290
# references. Deliberately NOT `main`, which is where `spike.data_url` points:
# the validation data must not drift underneath a published figure.
VALIDATION_DATA_COMMIT = "6c98b7b607d7387b508cdaa192d659ee9fca7367"
VALIDATION_DATA_URL = (
    "https://raw.githubusercontent.com/matsengrp/SARS-CoV-2_spike_multidms/"
    f"{VALIDATION_DATA_COMMIT}/data"
)

#: Files fetched by :func:`fetch_validation_data`, in the order returned.
VALIDATION_FILES = ("viral_titers.csv", "spike_validation_data.csv")


def fetch_validation_data(results_dir):
    """Download the viral-titer validation data used by Figure 5.

    The two files are standalone repository-level CSVs, unlike everything
    ``_common.download_data`` handles: that helper is condition-scoped, walking
    ``experiment_conditions`` and reading each condition's
    ``functional_selections.csv`` manifest. It has no slot for a repo-level
    file, so bending it to carry these two would distort a helper that four
    rules depend on.

    Files are cached under ``{results_dir}/raw_data/validation/`` and skipped
    if already present, matching ``_common.download_data``'s behaviour.

    Parameters
    ----------
    results_dir : str
        The pipeline's results directory. The cache is created beneath it.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(titers, validation)``. ``titers`` has columns ``virus``,
        ``background``, ``plate``, ``RLUperuL``. ``validation`` is wide, with
        one column per validation mutation and one row per
        (background, replicate).
    """
    cache_dir = os.path.join(results_dir, "raw_data", "validation")
    os.makedirs(cache_dir, exist_ok=True)

    frames = []
    for fname in VALIDATION_FILES:
        path = os.path.join(cache_dir, fname)
        if not os.path.exists(path):
            url = f"{VALIDATION_DATA_URL}/{fname}"
            print(f"  Downloading {url}")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(path, "w") as f:
                f.write(resp.text)
        frames.append(pd.read_csv(path))

    return tuple(frames)


def savefig(fig, name, figures_dir, formats=("pdf", "png"), dpi=300):
    """Save a figure to the pipeline's figures directory in several formats.

    Every manuscript figure is written as both a vector PDF (what the
    manuscript build consumes) and a raster PNG (what the docs page shows).
    Both are declared as rule outputs, so a figure that fails to render fails
    the pipeline run rather than silently vanishing into notebook output.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    name : str
        Base filename without extension, e.g.
        ``"shift_by_site_heatmap_zoom"``.
    figures_dir : str
        Destination directory. Created if absent.
    formats : sequence of str
        Extensions to write. Defaults to PDF and PNG.
    dpi : int
        Raster resolution.

    Returns
    -------
    list of str
        The paths written, in ``formats`` order.
    """
    os.makedirs(figures_dir, exist_ok=True)

    paths = []
    for ext in formats:
        path = os.path.join(figures_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        paths.append(path)
        print(f"  wrote {path}")

    return paths


def lasso_slice(df, lasso_choice, column="fusionreg"):
    """Select the rows of a multi-lambda table at the chosen lasso weight.

    Guards the failure mode that broke the test profile in this pipeline once
    already: an off-ladder ``lasso_choice`` matches no rows, and the resulting
    empty frame fails much later with an error naming neither the config nor
    the lambda. See ``tests/test_config_tiers.py``.

    Parameters
    ----------
    df : pandas.DataFrame
        A table carrying one row group per regularization weight.
    lasso_choice : float
        The chosen weight, from ``spike.lasso_choice`` in the downstream
        config.
    column : str
        Name of the regularization-weight column.

    Returns
    -------
    pandas.DataFrame
        The matching rows.

    Raises
    ------
    ValueError
        If no row matches, listing the weights that are present.
    """
    out = df[df[column] == lasso_choice]
    if out.empty:
        raise ValueError(
            f"lasso_choice={lasso_choice:g} matched no rows on '{column}'. "
            f"Present values: {sorted(df[column].unique())}"
        )
    return out


def set_plot_style(rc=None):
    """Apply the manuscript's matplotlib defaults.

    Parameters
    ----------
    rc : dict or None
        Extra ``matplotlib.rcParams`` entries, merged last so a caller can
        override any default here.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,  # embed TrueType, not Type 3 -- journals require it
            "ps.fonttype": 42,
        }
    )
    if rc:
        plt.rcParams.update(rc)


def fit_single_condition(
    func_score_df, replicate, condition, fit_config, verbose=False
):
    """Fit one single-condition model -- the naive arm's unit of work.

    A single-condition ``Data`` has no shift parameters, so ``fusionreg`` and
    ``beta0_ridge`` are inert: every fusion and ridge term in ``jaxmodels`` is
    guarded by ``d != reference_condition``, and with one condition that branch
    never runs. They are therefore not passed at all, and the naive arm needs
    no lambda ladder -- which is why it costs ~4 minutes against the joint
    arm's ~2 h 20 min.

    Uses the raw ``aa_substitutions`` column, which ``_common.load_raw_data``
    has already rewritten into reference (BA.1) numbering. The subset must
    retain its wildtype row: ``jaxmodels.Data.from_multidms`` takes
    ``x_wt = X[0]`` and asserts ``x_wt.sum() == 0``.

    Parameters
    ----------
    func_score_df : pandas.DataFrame
        Training functional scores already subset to one (replicate,
        condition). Needs ``aa_substitutions``, ``func_score`` and
        ``condition`` columns.
    replicate : int or str
        Replicate label, recorded in the returned diagnostics.
    condition : str
        The condition to fit. Becomes the ``Data`` object's own reference.
    fit_config : dict
        The ``spike.fitting`` section of the fit-tier config. Read, never
        written -- writing it would invalidate the cached joint fit.
    verbose : bool
        Forwarded to ``Model.fit``.

    Returns
    -------
    tuple
        ``(model, diagnostics)``. ``diagnostics`` is a dict carrying
        ``replicate``, ``condition``, ``converged``, ``n_outer_sweeps``,
        ``final_objective_error``, ``tol``, ``alpha``, ``beta0``,
        ``n_variants`` and ``seconds``.
    """
    agg = (
        func_score_df.groupby(["condition", "aa_substitutions"], dropna=False)
        .agg({"func_score": "mean"})
        .reset_index()
    )

    data = multidms.Data(
        agg,
        alphabet=multidms.AAS_WITHSTOP_WITHGAP,
        reference=condition,
        assert_site_integrity=False,
        name=f"rep_{replicate}_{condition}",
    )

    # fusionreg and beta0_ridge are forwarded rather than left at their 0.0
    # defaults. They are inert here -- with one condition the guarded branches
    # never run -- but passing them makes that inertness a property the test
    # suite can actually falsify, instead of one hidden behind a default.
    model = multidms.Model(
        data,
        ge_type=fit_config["ge_type"],
        l2reg=fit_config["l2reg"],
        fusionreg=fit_config.get("fusionreg", 0.0),
        beta0_ridge=fit_config.get("beta0_ridge", 0.0),
    )

    tol = fit_config["tol"]
    start = time.time()
    model.fit(
        maxiter=fit_config["maxiter"],
        tol=tol,
        warmstart=fit_config.get("warmstart", False),
        recompute_scale=fit_config.get("recompute_scale", False),
        share_alpha=fit_config.get("share_alpha", True),
        alpha_init=fit_config["alpha_init"],
        # beta0_init must be a dict keyed by condition; passing the scalar that
        # reads naturally from the config raises TypeError at jaxmodels.py:647,
        # where `d in beta0_init` runs `in` against a float.
        beta0_init={condition: fit_config["beta0_init"][condition]},
        beta_clip_range=tuple(fit_config["beta_clip_range"]),
        ge_kwargs=fit_config["ge_kwargs"],
        cal_kwargs=fit_config["cal_kwargs"],
        loss_kwargs=fit_config["loss_kwargs"],
        verbose=verbose,
    )
    elapsed = time.time() - start

    trajectory = model.convergence_trajectory_df
    # Use the public per-condition parameter frame rather than reaching into
    # the jax model's attributes. With one condition it is a single row.
    ge_row = model.get_ge_params_df().iloc[0]

    diagnostics = {
        "replicate": replicate,
        "condition": condition,
        "converged": bool(model.converged),
        "n_outer_sweeps": int(len(trajectory)),
        "final_objective_error": float(
            trajectory["objective_error_trajectory"].iloc[-1]
        ),
        "tol": float(tol),
        "alpha": float(ge_row["alpha"]),
        "beta0": float(ge_row["beta0"]),
        "n_variants": int(len(agg)),
        "seconds": round(elapsed, 1),
    }
    return model, diagnostics


def fit_naive_arm(func_score_df, conditions, replicates, fit_config, verbose=False):
    """Fit the whole naive arm -- one model per (replicate, condition).

    A subset lacking the minimum for ``Data`` construction -- at least one
    wildtype row plus one variant -- is warned about and skipped rather than
    raising, so the subsampled test profile stays runnable. Its absence is
    recorded with ``converged=False``.

    Parameters
    ----------
    func_score_df : pandas.DataFrame
        The full training functional scores, with ``replicate`` and
        ``condition`` columns.
    conditions : sequence of str
        Every condition to fit, including the reference.
    replicates : sequence
        Replicate labels to fit.
    fit_config : dict
        The ``spike.fitting`` section of the fit-tier config.
    verbose : bool
        Forwarded to each fit.

    Returns
    -------
    tuple
        ``(models, convergence_df, ge_params_df)``, where ``models`` maps
        ``(replicate, condition)`` to a fitted :class:`multidms.Model`.
    """
    models = {}
    diagnostics = []

    for replicate in replicates:
        for condition in conditions:
            subset = func_score_df[
                (func_score_df["replicate"] == replicate)
                & (func_score_df["condition"] == condition)
            ]
            n_wt = int((subset["aa_substitutions"] == "").sum())
            if n_wt < 1 or len(subset) - n_wt < 1:
                warnings.warn(
                    f"rep {replicate} / {condition}: {len(subset)} variants "
                    f"({n_wt} wildtype) is below the minimum for Data "
                    "construction; skipping this fit. Expected when the "
                    "profile subsamples, unexpected on full data.",
                    stacklevel=2,
                )
                diagnostics.append(
                    {
                        "replicate": replicate,
                        "condition": condition,
                        "converged": False,
                        "n_outer_sweeps": 0,
                        "final_objective_error": float("nan"),
                        "tol": float(fit_config["tol"]),
                        "alpha": float("nan"),
                        "beta0": float("nan"),
                        "n_variants": int(len(subset)),
                        "seconds": 0.0,
                    }
                )
                continue

            model, diag = fit_single_condition(
                subset, replicate, condition, fit_config, verbose=verbose
            )
            models[(replicate, condition)] = model
            diagnostics.append(diag)
            print(
                f"  rep {replicate} / {condition}: "
                f"converged={diag['converged']} "
                f"sweeps={diag['n_outer_sweeps']} "
                f"alpha={diag['alpha']:.2f} ({diag['seconds']}s)"
            )

    diag_df = pd.DataFrame(diagnostics)
    convergence_cols = [
        "replicate",
        "condition",
        "converged",
        "n_outer_sweeps",
        "final_objective_error",
        "tol",
    ]
    ge_cols = ["replicate", "condition", "alpha", "beta0", "n_variants"]
    return models, diag_df[convergence_cols], diag_df[ge_cols]


def derive_naive_shifts(models, reference, times_seen_threshold=1):
    """Derive naive shifts as plain differences of independently fitted betas.

    This is the "naive approach" the manuscript contrasts against joint
    fitting: fit each condition alone, then subtract. Shifts are computed on
    the INNER join of the per-condition mutation indices -- a union would
    silently pair non-equivalent labels, because 32 spike sites carry
    different wildtype letters across conditions, so the same physical
    substitution takes a different label per condition.

    ``times_seen_threshold`` defaults to 1 to match ``evaluate.ipynb``'s
    joint-arm call. Filtering the two arms differently rigs the comparison:
    joint shift_Delta replicate R^2 reads 0.543 at threshold 3 but 0.410 at 1.

    Parameters
    ----------
    models : dict
        Maps ``(replicate, condition)`` to a fitted model, as returned by
        :func:`fit_naive_arm`.
    reference : str
        The condition betas are subtracted against, e.g. ``"Omicron_BA1"``.
        Its own shift is identically zero.
    times_seen_threshold : int
        Forwarded to each model's ``get_mutations_df``.

    Returns
    -------
    pandas.DataFrame
        Long form, with columns ``mutation``, ``wts``, ``sites``, ``muts``,
        ``replicate``, ``condition``, ``beta``, ``naive_shift`` and
        ``times_seen``.

    Raises
    ------
    ValueError
        If a replicate's mutation-index intersection is empty, which means
        the key spaces disagree rather than that data is missing.
    """
    replicates = sorted({rep for rep, _ in models})
    frames = []

    for replicate in replicates:
        per_condition = {
            cond: model.get_mutations_df(times_seen_threshold=times_seen_threshold)
            for (rep, cond), model in models.items()
            if rep == replicate
        }
        if reference not in per_condition:
            warnings.warn(
                f"rep {replicate}: reference condition {reference!r} was not "
                "fitted, so no naive shifts can be derived for it.",
                stacklevel=2,
            )
            continue

        shared = None
        for frame in per_condition.values():
            shared = frame.index if shared is None else shared.intersection(frame.index)

        if len(shared) == 0:
            raise ValueError(
                f"rep {replicate}: the naive mutation index intersection is "
                f"empty across conditions {sorted(per_condition)}. An empty "
                "join means the key spaces disagree, which is a bug rather "
                "than missing data."
            )

        ref_beta = per_condition[reference].loc[shared, f"beta_{reference}"]
        for condition, frame in per_condition.items():
            sub = frame.loc[shared]
            frames.append(
                pd.DataFrame(
                    {
                        "mutation": list(shared),
                        "wts": sub["wts"].to_numpy(),
                        "sites": sub["sites"].to_numpy(),
                        "muts": sub["muts"].to_numpy(),
                        "replicate": replicate,
                        "condition": condition,
                        "beta": sub[f"beta_{condition}"].to_numpy(),
                        "naive_shift": (
                            sub[f"beta_{condition}"].to_numpy() - ref_beta.to_numpy()
                        ),
                        "times_seen": sub[f"times_seen_{condition}"].to_numpy(),
                    }
                )
            )

    if not frames:
        raise ValueError(
            f"No naive shifts could be derived: reference condition "
            f"{reference!r} was not fitted for any of replicates {replicates}."
        )

    return pd.concat(frames, ignore_index=True)


def assert_wt_agreement(naive_muts, site_map, conditions):
    """Assert no mutation in the shared index sits on a disagreeing site.

    32 spike sites carry different wildtype letters across conditions (19,
    417, 484, 501, 655, 681, 796 and friends), so at those sites the same
    physical substitution carries a different label per condition. A label
    can therefore appear in every condition's index only when the wildtype
    letter agrees -- which is exactly what makes the plain intersection safe.
    Verify that rather than trusting it.

    Parameters
    ----------
    naive_muts : pandas.DataFrame
        The long-form naive mutation table, with ``mutation`` and ``sites``.
    site_map : pandas.DataFrame
        The pipeline's ``site_map.csv``: ``sites`` plus one column per
        condition giving that condition's wildtype letter.
    conditions : sequence of str
        Conditions to compare. Any absent from ``site_map`` are skipped.

    Returns
    -------
    int
        The number of disagreeing mutations, always 0 on success.

    Raises
    ------
    ValueError
        If any mutation's site has disagreeing wildtype letters, since every
        naive shift would then be suspect.
    """
    present = [c for c in conditions if c in site_map.columns]
    merged = (
        naive_muts[["mutation", "wts", "sites"]]
        .drop_duplicates()
        .merge(site_map[["sites"] + present], on="sites", how="left")
    )
    # A site absent from site_map merges to all-NaN. nunique skips NaN, so
    # such a row scores 0 distinct letters and would slip through the
    # disagreement test below without ever having been checked. Catch it
    # separately -- an unverifiable site is not an agreeing one.
    unmapped = merged[merged[present].isna().any(axis=1)]
    if len(unmapped):
        raise ValueError(
            f"{len(unmapped)} mutations in the shared naive index sit on sites "
            f"absent from site_map, e.g. {unmapped['mutation'].head(5).tolist()}. "
            "Their wildtype letters cannot be checked, so the "
            "plain-intersection premise is unverified rather than satisfied."
        )

    disagree = merged[merged[present].nunique(axis=1) > 1]
    if len(disagree):
        raise ValueError(
            f"{len(disagree)} mutations in the shared naive index sit on sites "
            "whose wildtype letter disagrees across conditions, e.g. "
            f"{disagree['mutation'].head(5).tolist()}. The plain-intersection "
            "premise has broken and every naive shift is suspect."
        )
    return 0
