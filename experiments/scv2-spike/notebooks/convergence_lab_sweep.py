"""Convergence-lab parallel sweep fitter (#246).

Standalone, parallel companion to the ``convergence_lab.py`` marimo notebook.
Fits a hyperparameter grid with ``fit_models(n_processes=N)`` (which the marimo
notebook cannot do — it is capped at ``n_processes=1`` because it cannot spawn
workers safely), extracts per-fit basin diagnostics, and writes one results
pickle. The notebook and dashboard then *load* that pickle; no fitting happens
in marimo.

Design + living findings: ``convergence_lab_SWEEP_PLAN.md`` (same directory).

Stage A (the β-explosion fix, data-anchored l2reg) — 40 fits::

    JAX_PLATFORM_NAME=cpu pixi run python \\
        experiments/scv2-spike/notebooks/convergence_lab_sweep.py --stage A

Output: ``results/convergence_lab/sweep_stage<A>.pkl`` — a DataFrame with the
full ``fit_models`` schema plus the basin-metric columns ``alpha_final``,
``beta_l2_norm``, ``max_abs_phi`` (and the input ``replicate``). The pickle is
regenerated on demand and is **not** committed (the first 24-fit cache was
~580 MB).
"""

from __future__ import annotations

import argparse
import os

# Pin per-process CPU threading BEFORE JAX/XLA is imported (multidms imports JAX
# at module load, and these env vars are only read at XLA init). Without this,
# every fit_models worker spins XLA's threadpool across ALL cores; running N
# workers then oversubscribes the machine N-fold and each fit thrashes (the
# n_processes=4 Stage A run burned >72 CPU-min/cell that finished in ~40 s
# single-process). One thread per worker → workers × 1 = cores, no contention.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import multidms
from multidms.model_collection import fit_one_model, stack_fit_models
from multidms.utils import explode_params_dict

# Resolve paths relative to this file (mirrors the notebook / dashboard), so the
# script runs regardless of the process working directory.
NB_DIR = Path(os.path.abspath(__file__)).parent
SPIKE_DIR = NB_DIR.parent  # experiments/scv2-spike
DATA_CSV = (
    SPIKE_DIR
    / "results-prod-235-times-seen-threshold"
    / "training_functional_scores.csv"
)
RESULTS_DIR = SPIKE_DIR / "results" / "convergence_lab"
REF = "Omicron_BA1"

# Reduced iteration caps for a fast local sweep. A *sweep* needs to locate the
# l2reg knee, not produce production-quality fits. Probe (block 50→25, inner
# 50→20) measured ~38 s/fit (vs ~87 s) with α and Σβ² stable vs the full fit, so
# the basin diagnostics stay trustworthy. block=15/inner=10 was rejected:
# final_obj_err blew up to ~2e-2 (not converged) and params drifted. Stage B
# sweeps the inner maxiter explicitly; Stage A holds it at this fast value.
_INNER = dict(tol=1e-4, maxiter=20, maxls=40, jit=True)
_BLOCK_ITERS = 25
_BLOCK_TOL = 1e-6


def load_rep_data() -> dict[str, "multidms.Data"]:
    """Build one ``multidms.Data`` per replicate from the prod spike CSV.

    Mirrors Cell 1 of ``convergence_lab.py`` exactly so the sweep fits the same
    data the first factorial did.

    Returns:
        Mapping ``"rep_<n>" -> multidms.Data`` (reference = ``Omicron_BA1``).
    """
    raw = pd.read_csv(DATA_CSV)
    raw["aa_substitutions"] = raw["aa_substitutions"].fillna("")
    rep_data: dict[str, multidms.Data] = {}
    for rep in sorted(raw["replicate"].unique()):
        sub = raw[raw["replicate"] == rep][
            ["condition", "aa_substitutions", "func_score"]
        ].copy()
        rep_data[f"rep_{rep}"] = multidms.Data(
            sub,
            reference=REF,
            alphabet=multidms.AAS_WITHSTOP,
            name=f"rep_{rep}",
            verbose=False,
        )
    return rep_data


def stage_a_params(rep_data: dict[str, "multidms.Data"]) -> dict:
    """Stage A grid — data-anchored l2reg × warmstart × fusionreg × replicate.

    40 fits. ``recompute_scale=False`` and ``share_alpha=True`` are locked (the
    first factorial proved the former convergence-correct; the latter is the
    degeneracy-prone axis we are deliberately not loosening yet).

    Args:
        rep_data: Per-replicate Data objects from :func:`load_rep_data`.

    Returns:
        A param dict suitable for ``fit_models`` / ``explode_params_dict``.
    """
    return {
        "dataset": list(rep_data.values()),  # rep_1, rep_2
        "l2reg": [0.0, 1e-4, 3e-4, 6e-4, 1e-3],  # knee-refocused (probe: knee ≈ 3e-4)
        "warmstart": [True, False],
        "fusionreg": [0.0, 8e-5],
        "recompute_scale": [False],  # locked (proven convergence-correct)
        "share_alpha": [True],  # locked (degeneracy-prone; not loosened in Stage A)
        "scale_fusion_by_n": [False],  # deferred to Stage B
        "maxiter": [_BLOCK_ITERS],
        "tol": [_BLOCK_TOL],
        "beta0_ridge": [0.0],
        "ge_type": ["Sigmoid"],
        "alpha_init": [None],
        "beta0_init": [None],
        "beta_clip_range": [None],
        "ge_kwargs": [dict(_INNER)],
        "cal_kwargs": [dict(_INNER)],
        "loss_kwargs": [dict(δ=1.0)],
    }


def smoke_params(rep_data: dict[str, "multidms.Data"]) -> dict:
    """An 8-cell subgrid for fast parallelism/threading validation.

    2 reps × {l2reg 0, 3e-4} × {warmstart T, F}, fusionreg fixed at 0. Spans the
    cheap and the previously-runaway corners so a timed ``n_processes=4`` run
    confirms thread-pinning fixed the oversubscription before the full Stage A.

    Args:
        rep_data: Per-replicate Data objects from :func:`load_rep_data`.

    Returns:
        A param dict suitable for ``fit_models`` / ``explode_params_dict``.
    """
    p = stage_a_params(rep_data)
    p["l2reg"] = [0.0, 3e-4]
    p["fusionreg"] = [0.0]
    return p


STAGES = {"A": (stage_a_params, 40), "smoke": (smoke_params, 8)}


def _basin_metrics(model) -> dict[str, float]:
    """Extract per-fit degeneracy diagnostics from a fitted ``multidms.Model``.

    Detects the α/β see-saw documented in ``convergence_lab_SWEEP_PLAN.md``:
    the small-α basin collapses α toward ~1.5 while β explodes (Σβ² up to ~75k,
    max|φ| up to ~4751). Access paths validated against the 24-fit cache.

    Args:
        model: A fitted ``multidms.Model``.

    Returns:
        ``alpha_final`` (shared scalar α), ``beta_l2_norm`` (Σ_cond Σ_i β²), and
        ``max_abs_phi`` (max |latent phenotype| over all variants/conditions).
        Any field that cannot be extracted is ``NaN``.
    """
    jm = model._jax_model
    out: dict[str, float] = {}

    # Shared scalar α (share_alpha=True throughout Stage A).
    try:
        out["alpha_final"] = float(np.ravel(np.asarray(jm.α))[0])
    except Exception:
        out["alpha_final"] = float("nan")

    # Σβ² over all conditions — the quantity the l2reg grid targets.
    try:
        out["beta_l2_norm"] = float(
            sum((np.asarray(jm.φ[c].β) ** 2).sum() for c in jm.φ)
        )
    except Exception:
        out["beta_l2_norm"] = float("nan")

    # max |φ| — the visible explosion (the GE-landscape x-axis blowout).
    try:
        vdf = model.get_variants_df(phenotype_as_effect=False)
        phi_col = next(
            (
                c
                for c in (
                    "predicted_latent",
                    "latent_phenotype",
                    "predicted_latent_phenotype",
                )
                if c in vdf.columns
            ),
            None,
        )
        out["max_abs_phi"] = (
            float(np.nanmax(np.abs(vdf[phi_col].to_numpy())))
            if phi_col
            else float("nan")
        )
    except Exception:
        out["max_abs_phi"] = float("nan")

    return out


def add_basin_metrics(fit_df: pd.DataFrame) -> pd.DataFrame:
    """Append basin-metric columns and a tidy ``replicate`` label to the fits.

    Args:
        fit_df: The DataFrame returned by ``fit_models``.

    Returns:
        ``fit_df`` with ``alpha_final``, ``beta_l2_norm``, ``max_abs_phi``, and
        ``replicate`` columns added.
    """
    metrics = fit_df["model"].apply(_basin_metrics).apply(pd.Series)
    fit_df = pd.concat([fit_df.reset_index(drop=True), metrics], axis=1)
    fit_df["replicate"] = fit_df["model"].apply(lambda m: getattr(m.data, "name", None))
    return fit_df


def run_stage(stage: str) -> Path:
    """Fit one stage's grid sequentially and write the results pickle.

    Args:
        stage: Stage key (``"A"`` or ``"smoke"``).

    Returns:
        Path to the written pickle.
    """
    build, expected = STAGES[stage]
    rep_data = load_rep_data()
    params = build(rep_data)

    exploded = explode_params_dict(params)
    n_combos = len(exploded)
    assert (
        n_combos == expected
    ), f"stage {stage}: expected {expected} fits, got {n_combos}"
    print(f"[stage {stage}] fitting {n_combos} models SEQUENTIALLY …", flush=True)

    # Sequential fit via a plain fit_one_model loop — NOT fit_models. fit_models
    # routes even n_processes=1 through a multiprocessing-spawn worker
    # (_fit_fun), which hangs on this grid's l2reg>0 cells (the first factorial
    # only ever ran l2reg=0). The direct call completes in ~40 s/fit. We rebuild
    # the same DataFrame fit_models would via stack_fit_models, so downstream
    # (add_basin_metrics, ModelCollection) is unchanged.
    import time

    t0 = time.time()
    results = []
    for i, kw in enumerate(exploded, 1):
        ft = time.time()
        try:
            results.append(fit_one_model(**kw))
            status = "ok"
        except Exception as exc:  # match fit_models failures="tolerate"
            results.append(None)
            status = f"FAILED {type(exc).__name__}"
        print(
            f"  [{i:>2}/{n_combos}] l2={kw['l2reg']:<7} warm={str(kw['warmstart']):<5} "
            f"fr={kw['fusionreg']:<7} -> {time.time() - ft:5.1f}s  {status}",
            flush=True,
        )
    n_failed = sum(r is None for r in results)
    n_fit = n_combos - n_failed
    dt = time.time() - t0
    print(
        f"[stage {stage}] fit: {n_fit}  failed: {n_failed}  "
        f"wall: {dt:.1f}s  ({dt / max(n_combos, 1):.1f}s/fit)",
        flush=True,
    )

    fit_df = stack_fit_models(results)
    fit_df = add_basin_metrics(fit_df)

    # dict-valued columns → str for groupby/pickle compatibility (prod convention).
    for col in fit_df.columns:
        if fit_df[col].apply(lambda x: isinstance(x, dict)).any():
            fit_df[col] = fit_df[col].apply(str)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"sweep_stage{stage}.pkl"
    with open(out_path, "wb") as fh:
        pickle.dump(fit_df, fh)
    print(f"[stage {stage}] wrote {out_path}  ({len(fit_df)} rows)")
    return out_path


def main() -> None:
    """CLI entrypoint: ``--stage`` selects the grid (fit sequentially)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="A", choices=sorted(STAGES), help="grid to fit")
    args = ap.parse_args()
    run_stage(args.stage)


if __name__ == "__main__":
    main()
