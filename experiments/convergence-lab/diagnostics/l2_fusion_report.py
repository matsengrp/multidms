r"""Downstream report for the l2-fusion β-explosion sweep (#256).

Loads a convergence-lab ``fit_collection.pkl`` into a ``ModelCollection``
and prints a per-cell table of basin diagnostics (Σβ², α, converged,
final_obj_err) plus replicate-shift Pearson r per ``l2reg`` slice. The
harness only fits; this computes the derived metrics on the fly, matching
the convergence-lab design (nothing derived is stored in the pickle).

The ``l2reg=0`` rows are the β-explosion control; success is at least one
``l2reg>0`` value showing bounded Σβ² + α across all three fusion
strengths, with replicate-r that does not collapse as fusionreg rises.

Run::

    pixi run python experiments/convergence-lab/diagnostics/l2_fusion_report.py \\
        --cache l2-fusion
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import warnings
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

# Reuse the harness's RESULTS_DIR / path constants.
sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))
import harness  # noqa: E402

from multidms.model_collection import ModelCollection  # noqa: E402


def basin_row(fit) -> dict:
    """Extract basin diagnostics from one fit-collection row.

    Args:
        fit: A row (``pd.Series``) of the fit-collection frame; ``fit.model``
            is a fitted ``multidms.Model``.

    Returns:
        Dict with ``l2reg``, ``fusionreg``, ``dataset_name``, ``sum_beta_sq``
        (Σβ² of the reference condition's mutation effects), ``alpha`` (shared
        scalar), ``converged`` (bool), and ``final_obj_err``.
    """
    model = fit.model
    ref = model.data.reference
    beta = model.params.φ[ref].β
    sum_beta_sq = float((beta**2).sum())
    alpha = float(model.params.α)
    try:
        final_obj_err = float(
            model.convergence_trajectory_df["objective_error_trajectory"].iloc[-1]
        )
    except (TypeError, KeyError, IndexError):
        final_obj_err = float("nan")
    return {
        "l2reg": fit.l2reg,
        "fusionreg": fit.fusionreg,
        "dataset_name": fit.dataset_name,
        "sum_beta_sq": sum_beta_sq,
        "alpha": alpha,
        "converged": bool(model.converged),
        "final_obj_err": final_obj_err,
    }


def basin_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-fit basin diagnostics for the whole collection.

    Args:
        frame: The raw fit-collection DataFrame.

    Returns:
        One row per fit, sorted by ``(l2reg, fusionreg, dataset_name)``.
    """
    rows = [basin_row(frame.iloc[i]) for i in range(len(frame))]
    return (
        pd.DataFrame(rows)
        .sort_values(["l2reg", "fusionreg", "dataset_name"])
        .reset_index(drop=True)
    )


def replicate_corr_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Replicate-shift Pearson r across fusionreg, per l2reg slice.

    ``mut_param_dataset_correlation`` groups correlation by ``(dataset_name,
    fusionreg)``, so it must be run once per ``l2reg`` level to avoid mixing
    l2 regimes. Returns the concatenated per-slice correlation frames, each
    tagged with its ``l2reg``. The underlying frame carries a ``mut_param``
    column (the shift, e.g. ``shift_Delta``), so each ``(l2reg, fusionreg)``
    cell yields one row per shift.

    Args:
        frame: The raw fit-collection DataFrame.

    Returns:
        Concatenated correlation frames (columns ``datasets``, ``mut_param``,
        ``correlation``, ``fusionreg`` plus an ``l2reg`` tag), or an empty
        frame if no slice has ≥2 datasets.
    """
    mc = ModelCollection(frame)
    out = []
    for l2 in sorted(frame["l2reg"].unique()):
        try:
            _, df = mc.mut_param_dataset_correlation(
                x="fusionreg",
                return_data=True,
                r=1,
                query=f"l2reg == {l2}",
            )
        except ValueError:
            # Fewer than 2 datasets in this slice — skip it honestly.
            continue
        df = df.copy()
        df["l2reg"] = l2
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> None:
    """CLI: ``--cache <name>`` → print basin + replicate-r tables."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache",
        default="l2-fusion",
        help="results subdir holding fit_collection.pkl (default: l2-fusion)",
    )
    args = ap.parse_args()

    pkl = harness.RESULTS_DIR / args.cache / "fit_collection.pkl"
    frame = pickle.load(open(pkl, "rb"))
    print(f"[report] loaded {len(frame)} fits from {pkl}\n")

    basin = basin_table(frame)
    with pd.option_context("display.width", 200, "display.max_rows", 50):
        print("=== basin diagnostics (per fit) ===")
        print(basin.to_string(index=False))
        print("\n=== replicate-shift Pearson r (per l2reg slice) ===")
        corr = replicate_corr_table(frame)
        print(corr.to_string(index=False) if len(corr) else "(no ≥2-dataset slice)")


if __name__ == "__main__":
    main()
