r"""Downstream report for the Phase 1 β-control head-to-head (#263).

Loads BOTH convergence-lab ``fit_collection.pkl``s (the clip arm and the
l2 arm), tags each fit with an ``arm`` label derived in Python, and prints
three tables: basin diagnostics (Σβ², α, converged, final_obj_err), a
maxiter-each-needs table (outer iteration where objective_error first fell
below 1e-6 and 1e-4), and replicate-shift Pearson r per arm.

An "arm" is the ``(beta_clip_range, l2reg)`` pair. ``beta_clip_range`` is a
list/None column that pandas ``.query()`` cannot match cleanly, so arms are
sliced by a Python-derived ``arm`` column, not a query string.

The convergence verdict is the maxiter-each-needs table, NOT the
``converged`` flag: at the strict ``tol=1e-6`` the flag is expected mostly
False (#256 logged all fits False even at looser truncation).

Run::

    pixi run python experiments/convergence-lab/diagnostics/beta_control_report.py \\
        --cache-clip beta-control-clip --cache-l2 beta-control-l2
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

# Reuse the harness's RESULTS_DIR and the l2-fusion report's basin_row.
sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent.parent))
import harness  # noqa: E402

sys.path.insert(0, str(Path(os.path.abspath(__file__)).parent))
import l2_fusion_report as l2rpt  # noqa: E402

from multidms.model_collection import ModelCollection  # noqa: E402


def arm_label(beta_clip_range) -> str:
    """Return the arm label for one fit's ``beta_clip_range`` cell.

    The clip arm stores a Python list ``[-10, 10]``; the l2 arm stores the
    absent bound, which round-trips through an object-dtype column as EITHER
    ``None`` OR ``float('nan')``. Both must map to ``"l2"``.

    Args:
        beta_clip_range: The cell value (a list, ``None``, or ``NaN``).

    Returns:
        ``"clip"`` if the value is a list, else ``"l2"``.
    """
    return "clip" if isinstance(beta_clip_range, (list, tuple)) else "l2"


def first_below(trajectory, threshold: float, sentinel: int = 101) -> int:
    """First 1-based outer iteration where ``trajectory`` drops below ``threshold``.

    Searched from the FRONT (unlike ``basin_row``'s ``.iloc[-1]``), so the
    1e-4 crossing lands earlier than the last row.

    Args:
        trajectory: Iterable of per-sweep ``objective_error`` values.
        threshold: The floor to cross.
        sentinel: Value returned when the threshold is never crossed
            (default 101, one above the maxiter=100 ceiling).

    Returns:
        The 1-based index of the first crossing, or ``sentinel``.
    """
    for i, val in enumerate(trajectory, start=1):
        if val < threshold:
            return i
    return sentinel


def tagged_frame(clip_frame: pd.DataFrame, l2_frame: pd.DataFrame) -> pd.DataFrame:
    """Concatenate both arm frames, adding an ``arm`` column derived per row.

    Args:
        clip_frame: Raw fit-collection frame from the clip-arm pickle.
        l2_frame: Raw fit-collection frame from the l2-arm pickle.

    Returns:
        The concatenation with an ``arm`` column (``"clip"`` / ``"l2"``).
    """
    out = pd.concat([clip_frame, l2_frame], ignore_index=True)
    out["arm"] = out["beta_clip_range"].map(arm_label)
    return out


def basin_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-fit basin diagnostics for the tagged collection.

    Args:
        frame: The tagged frame from :func:`tagged_frame` (carries ``arm``).

    Returns:
        One row per fit (``arm`` + the ``basin_row`` fields), sorted by
        ``(arm, fusionreg, dataset_name)``.
    """
    rows = []
    for i in range(len(frame)):
        fit = frame.iloc[i]
        row = l2rpt.basin_row(fit)
        row["arm"] = fit.arm
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["arm", "fusionreg", "dataset_name"])
        .reset_index(drop=True)
    )


def maxiter_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-fit outer-iteration counts to cross 1e-6 and 1e-4.

    Args:
        frame: The tagged frame from :func:`tagged_frame`.

    Returns:
        One row per fit with ``arm``, ``fusionreg``, ``dataset_name``,
        ``iters_to_1e6``, ``iters_to_1e4``, sorted by
        ``(arm, fusionreg, dataset_name)``.
    """
    rows = []
    for i in range(len(frame)):
        fit = frame.iloc[i]
        traj = fit.model.convergence_trajectory_df["objective_error_trajectory"]
        rows.append(
            {
                "arm": fit.arm,
                "fusionreg": fit.fusionreg,
                "dataset_name": fit.dataset_name,
                "iters_to_1e6": first_below(traj, 1e-6),
                "iters_to_1e4": first_below(traj, 1e-4),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["arm", "fusionreg", "dataset_name"])
        .reset_index(drop=True)
    )


def replicate_corr_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Replicate-shift Pearson r across fusionreg, per arm slice.

    ``mut_param_dataset_correlation`` groups by ``(dataset_name, fusionreg)``,
    so it is run once per ``arm`` level to avoid mixing arms. The arm slice
    is selected by an ``arm``-column query (a plain string column, unlike the
    list-valued ``beta_clip_range``).

    Args:
        frame: The tagged frame from :func:`tagged_frame`.

    Returns:
        Concatenated per-arm correlation frames (columns ``datasets``,
        ``mut_param``, ``correlation``, ``fusionreg`` plus an ``arm`` tag),
        or an empty frame if no arm slice has ≥2 datasets.
    """
    mc = ModelCollection(frame)
    out = []
    for arm in sorted(frame["arm"].unique()):
        try:
            _, df = mc.mut_param_dataset_correlation(
                x="fusionreg",
                return_data=True,
                r=1,
                query=f"arm == '{arm}'",
            )
        except ValueError:
            # Fewer than 2 datasets in this slice — skip it honestly.
            continue
        df = df.copy()
        df["arm"] = arm
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> None:
    """CLI: ``--cache-clip/--cache-l2`` → print basin + maxiter + replicate-r."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-clip", default="beta-control-clip")
    ap.add_argument("--cache-l2", default="beta-control-l2")
    args = ap.parse_args()

    clip_pkl = harness.RESULTS_DIR / args.cache_clip / "fit_collection.pkl"
    l2_pkl = harness.RESULTS_DIR / args.cache_l2 / "fit_collection.pkl"
    clip_frame = pickle.load(open(clip_pkl, "rb"))
    l2_frame = pickle.load(open(l2_pkl, "rb"))
    frame = tagged_frame(clip_frame, l2_frame)
    print(f"[report] {len(clip_frame)} clip + {len(l2_frame)} l2 = {len(frame)} fits\n")

    with pd.option_context("display.width", 200, "display.max_rows", 50):
        print("=== basin diagnostics (per fit) ===")
        print(basin_table(frame).to_string(index=False))
        print("\n=== maxiter each needs (iters to cross 1e-6 / 1e-4) ===")
        print(maxiter_table(frame).to_string(index=False))
        print("\n=== replicate-shift Pearson r (per arm) ===")
        corr = replicate_corr_table(frame)
        print(corr.to_string(index=False) if len(corr) else "(no ≥2-dataset slice)")


if __name__ == "__main__":
    main()
