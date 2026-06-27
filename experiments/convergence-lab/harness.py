r"""Convergence-lab harness (#253): config-driven sequential model fitting.

Reads a grid YAML (a ``sweep:`` Cartesian product + optional ``fixed:``
overrides + optional ``replicates:``), fits each cell sequentially via
``multidms.model_collection.fit_one_model`` (NOT ``fit_models`` — kept
sequential for simplicity), extracts per-fit basin diagnostics into
``df_fits``, computes replicate-shift correlations into ``df_corr``, and
pickles ``{"df_fits": ..., "df_corr": ...}`` to ``results/<cache>.pkl``.

The runner owns the constant scv2-spike data; configs carry only swept
knobs and fixed overrides. See ``README.md`` in this directory.

Usage::

    pixi run python experiments/convergence-lab/harness.py \\
        --config grids/smoke.yaml --cache smoke
"""

from __future__ import annotations

import argparse  # noqa: F401 — used in later tasks (CLI)
import os

# Pin per-process CPU threading BEFORE JAX/XLA imports (multidms imports JAX at
# module load; these env vars are only read at XLA init). Harmless for the
# sequential harness; required if a future config restores parallelism.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import pickle  # noqa: E402,F401 — used in later tasks (persist results)
import time  # noqa: E402,F401 — used in later tasks (timing)
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402,F401 — used in later tasks (corr)
import pandas as pd  # noqa: E402,F401 — used in later tasks (df_fits)
import yaml  # noqa: E402

warnings.filterwarnings("ignore")

import multidms  # noqa: E402,F401 — used in later tasks (Data)
from multidms.model_collection import (  # noqa: E402
    ModelCollection,  # noqa: F401 — used in later tasks
    fit_one_model,  # noqa: F401 — used in later tasks
    stack_fit_models,  # noqa: F401 — used in later tasks
)
from multidms.utils import explode_params_dict  # noqa: E402

# --- Constant data the runner owns (resolved relative to this file) ----------
HARNESS_DIR = Path(os.path.abspath(__file__)).parent
DATA_CSV = (
    HARNESS_DIR.parent
    / "scv2-spike"
    / "results-prod-235-times-seen-threshold"
    / "training_functional_scores.csv"
)
RESULTS_DIR = HARNESS_DIR / "results"
REF = "Omicron_BA1"

# Exact fit_one_model signature (model_collection.py:61) minus `dataset`
# (runner-supplied) and `verbose`. The allowlist for config validation.
VALID_KWARGS = frozenset(
    {
        "ge_type",
        "l2reg",
        "fusionreg",
        "beta0_ridge",
        "scale_fusion_by_n",
        "loss_type",
        "maxiter",
        "tol",
        "warmstart",
        "recompute_scale",
        "beta0_init",
        "beta_init",
        "alpha_init",
        "share_alpha",
        "beta_clip_range",
        "ge_kwargs",
        "cal_kwargs",
        "loss_kwargs",
    }
)


def load_config(path) -> dict:
    """Load and validate a grid YAML.

    The config has two kwargs sections — ``sweep`` (each value a list, crossed
    in a Cartesian product) and ``fixed`` (each value a scalar, applied to every
    fit) — plus an optional harness-level ``replicates`` list (default
    ``[1, 2]``). Every key in ``sweep``/``fixed`` must be a valid
    ``fit_one_model`` kwarg (``VALID_KWARGS``); ``fit_one_model``'s ``**kwargs``
    would otherwise swallow typos silently, so we reject them here.

    Args:
        path: Path to the YAML config.

    Returns:
        ``{"sweep": dict, "fixed": dict, "replicates": list[int]}``.

    Raises:
        ValueError: If ``sweep`` is missing/empty, or any ``sweep``/``fixed``
            key is not in ``VALID_KWARGS`` (the message names the bad key).
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    sweep = raw.get("sweep") or {}
    fixed = raw.get("fixed") or {}
    replicates = raw.get("replicates", [1, 2])

    if not sweep:
        raise ValueError(f"config {path}: must define a non-empty 'sweep:' section")

    for section_name, section in (("sweep", sweep), ("fixed", fixed)):
        for key in section:
            if key not in VALID_KWARGS:
                raise ValueError(
                    f"config {path}: '{key}' in '{section_name}:' is not a valid "
                    f"fit_one_model kwarg. Valid keys: {sorted(VALID_KWARGS)}"
                )

    return {"sweep": dict(sweep), "fixed": dict(fixed), "replicates": list(replicates)}


def explode_grid(config: dict) -> list[dict]:
    """Cartesian-product the sweep × replicates, merging fixed into each cell.

    Args:
        config: The dict returned by :func:`load_config`.

    Returns:
        A list of kwarg dicts, one per fit. Each carries every ``fixed`` key and
        an integer ``replicate`` key (the runner maps it to the rep's Data).
    """
    sweep = dict(config["sweep"])
    sweep["replicate"] = list(config["replicates"])
    exploded = explode_params_dict(sweep)
    for cell in exploded:
        cell.update(config["fixed"])
    return exploded


def load_rep_data() -> dict[str, multidms.Data]:
    """Build one ``multidms.Data`` per replicate from the prod spike CSV.

    Mirrors the pipeline ``fit_models.ipynb``: aggregate ``func_score`` by
    ``(condition, aa_substitutions)`` with ``.mean()`` within each replicate,
    then construct a ``multidms.Data`` with the gap-inclusive alphabet.

    Returns:
        Mapping ``"rep_<n>" -> multidms.Data`` (reference = ``Omicron_BA1``).
    """
    raw = pd.read_csv(DATA_CSV).fillna({"aa_substitutions": ""})
    rep_data: dict[str, multidms.Data] = {}
    for rep in sorted(raw["replicate"].unique()):
        df_rep = raw[raw["replicate"] == rep]
        df_agg = (
            df_rep.groupby(["condition", "aa_substitutions"], dropna=False)
            .agg({"func_score": "mean"})
            .reset_index()
        )
        rep_data[f"rep_{rep}"] = multidms.Data(
            df_agg,
            alphabet=multidms.AAS_WITHSTOP_WITHGAP,
            reference=REF,
            assert_site_integrity=False,
            name=f"rep_{rep}",
            verbose=False,
        )
    return rep_data


def basin_metrics(model) -> dict[str, float]:
    """Per-fit degeneracy diagnostics from a fitted ``multidms.Model``.

    Detects the α/β see-saw: the small-α basin collapses α while β explodes
    (Σβ² large, max|φ| large). All access is Greek-attribute (``_jax_model.α``,
    ``_jax_model.φ[c].β``); ``final_obj_err`` comes from the convergence
    trajectory (it is NOT on the fit Series). Any field that cannot be
    extracted is ``NaN`` (``converged`` defaults ``False``).

    Args:
        model: A fitted ``multidms.Model``.

    Returns:
        ``alpha_final``, ``beta_l2_norm`` (Σ_cond Σ_i β²), ``max_abs_phi``,
        ``final_obj_err``, ``converged``.
    """
    out: dict[str, float] = {
        "alpha_final": float("nan"),
        "beta_l2_norm": float("nan"),
        "max_abs_phi": float("nan"),
        "final_obj_err": float("nan"),
        "converged": False,
    }

    try:
        jm = model._jax_model
    except Exception:
        return out

    try:
        out["alpha_final"] = float(np.ravel(np.asarray(jm.α))[0])
    except Exception:
        pass

    try:
        out["beta_l2_norm"] = float(
            sum((np.asarray(jm.φ[c].β) ** 2).sum() for c in jm.φ)
        )
    except Exception:
        pass

    try:
        vdf = model.get_variants_df(phenotype_as_effect=False)
        out["max_abs_phi"] = float(
            np.nanmax(np.abs(vdf["predicted_latent"].to_numpy()))
        )
    except Exception:
        pass

    try:
        traj = model.convergence_trajectory_df["objective_error_trajectory"]
        out["final_obj_err"] = float(traj.iloc[-1])
    except Exception:
        pass

    try:
        out["converged"] = bool(model.converged)
    except Exception:
        pass

    return out


def run_fits(exploded: list[dict], rep_data: dict) -> pd.DataFrame:
    """Fit every grid cell sequentially and assemble ``df_fits``.

    Each cell's integer ``replicate`` selects the rep's Data; the rest are
    ``fit_one_model`` kwargs. Failures are tolerated: a failed cell is dropped
    here (``stack_fit_models`` does NOT skip ``None`` — it would raise — so the
    harness filters them out before stacking), and the grid continues. A failed
    fit therefore contributes no row rather than poisoning the frame; if every
    fit fails, a ``RuntimeError`` is raised. Sequential ``fit_one_model`` is used
    directly — NOT ``fit_models`` — for simplicity and deterministic ordering.

    Args:
        exploded: Output of :func:`explode_grid`.
        rep_data: Output of :func:`load_rep_data`.

    Returns:
        ``df_fits``: the ``stack_fit_models`` frame plus ``alpha_final``,
        ``beta_l2_norm``, ``max_abs_phi``, ``final_obj_err``, ``converged``,
        and ``replicate`` columns. Retains the ``model`` column for Task 4's
        correlation step.
    """
    n = len(exploded)
    print(f"[harness] fitting {n} models SEQUENTIALLY …", flush=True)
    results = []
    t0 = time.time()
    for i, cell in enumerate(exploded, 1):
        kw = dict(cell)
        rep = kw.pop("replicate")
        dataset = rep_data[f"rep_{rep}"]
        ft = time.time()
        try:
            results.append(fit_one_model(dataset=dataset, **kw))
            status = "ok"
        except Exception as exc:
            results.append(None)
            status = f"FAILED {type(exc).__name__}"
        swept = {k: kw.get(k) for k in ("l2reg", "warmstart", "fusionreg")}
        print(
            f"  [{i:>2}/{n}] rep_{rep} {swept} -> {time.time() - ft:5.1f}s {status}",
            flush=True,
        )
    print(f"[harness] wall {time.time() - t0:.1f}s", flush=True)

    # stack_fit_models does NOT tolerate None (it calls .to_frame() on each
    # element); drop failed fits here so one failure does not abort the grid.
    ok = [r for r in results if r is not None]
    n_failed = len(results) - len(ok)
    if n_failed:
        print(
            f"[harness] {n_failed}/{len(results)} fit(s) failed and were dropped",
            flush=True,
        )
    if not ok:
        raise RuntimeError("[harness] all fits failed — nothing to stack")
    fit_df = stack_fit_models(ok)
    metrics = fit_df["model"].apply(basin_metrics).apply(pd.Series)
    fit_df = pd.concat([fit_df.reset_index(drop=True), metrics], axis=1)
    fit_df["replicate"] = fit_df["model"].apply(
        lambda m: getattr(getattr(m, "data", None), "name", None)
    )
    return fit_df


def primary_axis(config: dict) -> str:
    """The x-axis for the replicate-correlation groupby.

    ``mut_param_dataset_correlation`` groups by ``("dataset_name", x)``; ``x``
    must be a column on the fits frame (every swept kwarg is). Prefer
    ``fusionreg`` (the method's natural axis); else the first swept key.

    Args:
        config: The dict from :func:`load_config` (only ``sweep`` is read).

    Returns:
        The axis column name.
    """
    sweep = config["sweep"]
    if "fusionreg" in sweep:
        return "fusionreg"
    return next(iter(sweep))


def compute_corr(fit_df: pd.DataFrame, x: str) -> pd.DataFrame:
    """Replicate-shift correlation across the primary swept axis.

    Builds a ``ModelCollection`` from the fits and calls
    ``mut_param_dataset_correlation(x=x, times_seen_threshold=1,
    return_data=True, r=1)``, then keeps only shift parameters (``mut_param``
    beginning ``shift_``). Pearson r is computed between the two replicates'
    per-mutation shift estimates at each value of ``x``. Returns an empty frame
    if the correlation cannot be computed (e.g. <2 surviving replicates).

    Args:
        fit_df: ``df_fits`` from :func:`run_fits` (must retain the ``model``
            column and have ≥2 distinct ``dataset_name`` values).
        x: The groupby axis from :func:`primary_axis`.

    Returns:
        ``df_corr`` with columns ``datasets, mut_param, correlation, <x>``
        (shift params only).
    """
    try:
        mc = ModelCollection(fit_df)
        _, corr = mc.mut_param_dataset_correlation(
            x=x, times_seen_threshold=1, return_data=True, r=1
        )
    except Exception as exc:
        print(f"[harness] df_corr unavailable: {type(exc).__name__}: {exc}", flush=True)
        return pd.DataFrame(columns=["datasets", "mut_param", "correlation", x])
    return corr[corr["mut_param"].str.startswith("shift_")].reset_index(drop=True)


def run(config_path, cache: str) -> Path:
    """Load config, fit the grid, compute correlations, and pickle results.

    Args:
        config_path: Path to the grid YAML.
        cache: Base name for the output pickle (``results/<cache>.pkl``).

    Returns:
        Path to the written pickle (a dict ``{"df_fits", "df_corr"}``).
    """
    config = load_config(config_path)
    rep_data = load_rep_data()
    exploded = explode_grid(config)
    fit_df = run_fits(exploded, rep_data)
    corr_df = compute_corr(fit_df, primary_axis(config))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{cache}.pkl"
    with open(out_path, "wb") as fh:
        pickle.dump({"df_fits": fit_df, "df_corr": corr_df}, fh)
    print(
        f"[harness] wrote {out_path}  "
        f"(df_fits={len(fit_df)} rows, df_corr={len(corr_df)} rows)",
        flush=True,
    )
    return out_path


def main() -> None:
    """CLI: ``--config <grid.yaml> --cache <name>``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to grid YAML")
    ap.add_argument(
        "--cache", required=True, help="output base name → results/<cache>.pkl"
    )
    args = ap.parse_args()
    # Resolve a relative --config against the harness dir (so `grids/smoke.yaml`
    # works regardless of cwd), matching how DATA_CSV is resolved.
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute() and not cfg_path.exists():
        cfg_path = HARNESS_DIR / args.config
    run(cfg_path, args.cache)


if __name__ == "__main__":
    main()
