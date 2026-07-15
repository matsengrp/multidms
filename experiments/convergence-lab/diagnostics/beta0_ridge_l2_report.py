r"""Downstream report for the beta0_ridge × sub-knee l2reg scan (#284).

Loads a convergence-lab ``fit_collection.pkl`` into a ``ModelCollection`` and
prints, per ``(beta0_ridge, l2reg)`` cell: the convergence rate, median
``final_obj_err``, basin diagnostics (Σβ², α), and replicate-shift Pearson r.
With ``--baseline-cache`` it prints the same tables for the ``(0,0)`` baseline
so the two are read side by side — the baseline's convergence rate has never
been computed and cannot be quoted from the README.

Reuses ``l2_fusion_report.basin_row`` UNCHANGED (its contract is pinned by
``test_l2_fusion_report.py``) and adds the ``beta0_ridge`` column this grid
needs, mirroring how ``beta_control_report`` bolts on its ``arm`` column.

Adjudication (pre-registered in #284) — an axis shows an effect if any of:

* convergence: ≥1/8 converged in a cell where the baseline is 0/8;
* fallback:    median ``final_obj_err`` differing ≥2× from baseline;
* repro:       median replicate-r shifting ≥0.05 vs baseline;
* basin:       Σβ² or α differing ≥2× at matched ``fusionreg``.

The primary is expected to be DEGENERATE (0/72 vs 0/8) — when it is, the
``final_obj_err`` fallback IS the adjudicator, not a footnote.

Run::

    pixi run python experiments/convergence-lab/diagnostics/beta0_ridge_l2_report.py \\
        --cache beta0-ridge-l2-scan --baseline-cache 277-softplus-floor-off
"""

from __future__ import annotations

import argparse
import gc
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


def basin_row_with_ridge(fit) -> dict:
    """``l2_fusion_report.basin_row`` plus the ``beta0_ridge`` cell value.

    ``basin_row`` predates this grid's ``beta0_ridge`` axis and does not carry
    it. Rather than edit it (its contract is pinned by
    ``test_l2_fusion_report.py``), bolt the column on here — the same move
    ``beta_control_report.basin_table`` makes for its ``arm`` column.

    Args:
        fit: A row (``pd.Series``) of the fit-collection frame.

    Returns:
        ``basin_row``'s dict plus ``beta0_ridge``.
    """
    row = l2rpt.basin_row(fit)
    row["beta0_ridge"] = fit.beta0_ridge
    return row


def basin_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-fit basin diagnostics, sorted with ``beta0_ridge`` outermost.

    ``l2_fusion_report.basin_table`` sorts by ``(l2reg, fusionreg,
    dataset_name)`` and knows nothing of ``beta0_ridge``, so this grid needs its
    own.

    Args:
        frame: The raw fit-collection DataFrame.

    Returns:
        One row per fit, sorted by
        ``(beta0_ridge, l2reg, fusionreg, dataset_name)``.
    """
    rows = [basin_row_with_ridge(frame.iloc[i]) for i in range(len(frame))]
    return (
        pd.DataFrame(rows)
        .sort_values(["beta0_ridge", "l2reg", "fusionreg", "dataset_name"])
        .reset_index(drop=True)
    )


def convergence_table(basin: pd.DataFrame) -> pd.DataFrame:
    """Convergence rate + median diagnostics per ``(beta0_ridge, l2reg)`` cell.

    The denominator is explicit: each cell holds ``n`` fits (4 ``fusionreg`` × 2
    replicates = 8 for the full grid). ``median_obj_err`` is the pre-registered
    fallback adjudicator for when ``conv_rate`` is degenerate (all-zero) — the
    lab has repeatedly found the binary flag untrustworthy while
    ``final_obj_err`` stays tiny and informative (#273: 0/16 converged, obj_err
    ≤3e-4, parameters trustworthy).

    Args:
        basin: Output of :func:`basin_table`.

    Returns:
        One row per ``(beta0_ridge, l2reg)``: ``n``, ``n_converged``,
        ``conv_rate``, ``median_obj_err``, ``median_alpha``,
        ``median_sum_beta_sq``.
    """
    out = (
        basin.groupby(["beta0_ridge", "l2reg"], dropna=False)
        .agg(
            n=("converged", "size"),
            n_converged=("converged", "sum"),
            median_obj_err=("final_obj_err", "median"),
            median_alpha=("alpha", "median"),
            median_sum_beta_sq=("sum_beta_sq", "median"),
        )
        .reset_index()
    )
    out["conv_rate"] = out["n_converged"] / out["n"]
    return out[
        [
            "beta0_ridge",
            "l2reg",
            "n",
            "n_converged",
            "conv_rate",
            "median_obj_err",
            "median_alpha",
            "median_sum_beta_sq",
        ]
    ]


def replicate_corr_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Replicate-shift Pearson r across fusionreg, per ``(beta0_ridge, l2reg)``.

    ``mut_param_dataset_correlation`` forwards to ``split_apply_combine_muts``
    with ``groupby=("dataset_name", x)`` (``model_collection.py:1443-1445``) and
    mean-collapses every other column (``aggregate_func="mean"``). On this
    72-fit frame each ``(rep, fusionreg)`` group holds 9 fits (3 ``beta0_ridge``
    × 3 ``l2reg``), so the BARE call would average away the two axes under test
    and return a spuriously smooth r. Slicing with ``query=`` per cell is the
    fix — the same move ``l2_fusion_report.replicate_corr_table`` makes for its
    single ``l2reg`` loop, extended here to the nested pair.

    Each slice holds 2 datasets × 4 ``fusionreg`` = 8 fits, clearing the
    ``<2 datasets`` guard at ``model_collection.py:1440``. The correlated
    population is ``rep_1`` vs ``rep_2``; ``inner_merge_dataset_muts`` (default
    True) restricts to mutations shared across both.

    Args:
        frame: The raw fit-collection DataFrame.

    Returns:
        Concatenated per-slice correlation frames (columns ``datasets``,
        ``mut_param``, ``correlation``, ``fusionreg``, plus ``beta0_ridge`` and
        ``l2reg`` tags), or an empty frame if no slice has ≥2 datasets.
    """
    mc = ModelCollection(frame)
    out = []
    for b in sorted(frame["beta0_ridge"].unique()):
        for l2 in sorted(frame["l2reg"].unique()):
            try:
                _, df = mc.mut_param_dataset_correlation(
                    x="fusionreg",
                    return_data=True,
                    r=1,
                    query=f"beta0_ridge == {b} and l2reg == {l2}",
                )
            except ValueError:
                # Fewer than 2 datasets in this slice — skip it honestly.
                continue
            df = df.copy()
            df["beta0_ridge"] = b
            df["l2reg"] = l2
            out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _load(cache: str) -> pd.DataFrame:
    """Load one cache's fit-collection frame.

    Args:
        cache: Subdir under ``results/`` holding ``fit_collection.pkl``.

    Returns:
        The raw fit-collection DataFrame.
    """
    pkl = harness.RESULTS_DIR / cache / "fit_collection.pkl"
    frame = pickle.load(open(pkl, "rb"))
    size_gb = pkl.stat().st_size / 1e9
    print(f"[report] loaded {len(frame)} fits from {pkl} ({size_gb:.1f} GB)")
    return frame


def _report(cache: str, label: str) -> None:
    """Load one cache, print its tables, and drop the frame before returning.

    Loads inside the call rather than taking a frame, so the caller never holds
    two collections at once. These pickles carry the fitted models themselves at
    a measured ~88 MB/fit — the 72-fit scan is ~6.3 GB and the 8-fit baseline
    ~0.7 GB, so holding both would peak near 7 GB for no reason.

    Args:
        cache: Subdir under ``results/`` holding ``fit_collection.pkl``.
        label: Human-readable name for the section headers.
    """
    frame = _load(cache)
    basin = basin_table(frame)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(f"\n=== [{label}] convergence + basin per (beta0_ridge, l2reg) ===")
        print(convergence_table(basin).to_string(index=False))
        print(f"\n=== [{label}] basin diagnostics (per fit) ===")
        print(basin.to_string(index=False))
        print(f"\n=== [{label}] replicate-shift Pearson r (per cell slice) ===")
        corr = replicate_corr_table(frame)
        print(corr.to_string(index=False) if len(corr) else "(no ≥2-dataset slice)")

    # Drop the collection (and its fitted models) before the caller loads the
    # next one — see the docstring's size note.
    del frame
    gc.collect()


def main() -> None:
    """CLI: ``--cache`` (+ optional ``--baseline-cache``) → print all tables."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cache",
        default="beta0-ridge-l2-scan",
        help="results subdir holding fit_collection.pkl "
        "(default: beta0-ridge-l2-scan)",
    )
    ap.add_argument(
        "--baseline-cache",
        default=None,
        help="optional (0,0) baseline cache to report alongside "
        "(e.g. 277-softplus-floor-off)",
    )
    args = ap.parse_args()

    _report(args.cache, args.cache)
    if args.baseline_cache:
        _report(args.baseline_cache, f"BASELINE {args.baseline_cache}")


if __name__ == "__main__":
    main()
