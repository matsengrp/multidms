"""Tests for the beta0_ridge × l2reg report's extraction and grouping (#284)."""

import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

DIAG = Path(__file__).resolve().parent
sys.path.insert(0, str(DIAG))


def _sample_pickle():
    """Locate a sample fit_collection.pkl to extract from.

    ``results/`` is gitignored and never materialized in a worktree, so honor a
    ``BETA0_RIDGE_TEST_PKL`` override (point it at the canonical clone's copy);
    otherwise fall back to the in-tree baseline cache. Returns ``None`` when no
    fixture is reachable, so the test skips rather than fails.
    """
    override = os.environ.get("BETA0_RIDGE_TEST_PKL")
    if override and Path(override).exists():
        return Path(override)
    in_tree = DIAG.parent / "results" / "277-softplus-floor-off" / "fit_collection.pkl"
    return in_tree if in_tree.exists() else None


def _synthetic_basin() -> pd.DataFrame:
    """A basin frame with a known answer: 4 cells × 4 fits, half converged.

    2 beta0_ridge × 2 l2reg × 2 fusionreg × 2 replicates = 16 rows. Exactly one
    replicate per cell converges, so every cell's rate must be 0.5.
    """
    return pd.DataFrame(
        [
            {
                "beta0_ridge": b,
                "l2reg": l2,
                "fusionreg": f,
                "dataset_name": d,
                "sum_beta_sq": 100.0,
                "alpha": 5.0,
                "converged": conv,
                "final_obj_err": 1e-4,
            }
            for b in (1e-4, 1e-3)
            for l2 in (1e-8, 1e-7)
            for f in (0.0, 6.4e-4)
            for d, conv in (("rep_1", True), ("rep_2", False))
        ]
    )


def test_convergence_table_groups_by_both_axes():
    """convergence_table yields one row per (beta0_ridge, l2reg) with a rate.

    Uses a synthetic basin frame — no pickle needed, so this always runs.
    """
    import beta0_ridge_l2_report as rpt

    out = rpt.convergence_table(_synthetic_basin())
    assert len(out) == 4, "2 beta0_ridge × 2 l2reg = 4 cells"
    assert set(out.columns) >= {
        "beta0_ridge",
        "l2reg",
        "n",
        "n_converged",
        "conv_rate",
        "median_obj_err",
    }
    assert (out["n"] == 4).all(), "each cell holds 2 fusionreg × 2 reps = 4 fits"
    assert (out["conv_rate"] == 0.5).all(), "1 of 2 reps converged per cell"


def test_convergence_table_reports_a_degenerate_rate_honestly():
    """An all-unconverged grid yields conv_rate 0.0 with a finite median obj_err.

    This is the expected real-world case (#284 pre-registers a likely 0/72), and
    it is exactly when the median_obj_err fallback becomes the adjudicator — so
    the fallback column must still carry a real number when the rate is
    degenerate.
    """
    import beta0_ridge_l2_report as rpt

    basin = _synthetic_basin()
    basin["converged"] = False
    out = rpt.convergence_table(basin)
    assert (out["conv_rate"] == 0.0).all(), "degenerate rate reported as 0.0"
    assert out["median_obj_err"].notna().all(), "fallback must survive a 0/n rate"


def test_basin_table_sort_order_without_a_pickle(monkeypatch):
    """basin_table sorts beta0_ridge-outermost — checked without a fit collection.

    The pickle-backed sort test skips on a fresh checkout (results/ is
    gitignored), so the sort contract would go unverified exactly where it is
    most likely to regress. Stub the per-row extraction to cover the ordering on
    synthetic rows, which is all the sort itself depends on.
    """
    import beta0_ridge_l2_report as rpt

    rows = [
        {
            "beta0_ridge": b,
            "l2reg": l2,
            "fusionreg": f,
            "dataset_name": d,
            "sum_beta_sq": 1.0,
            "alpha": 1.0,
            "converged": False,
            "final_obj_err": 1e-4,
        }
        # Deliberately shuffled relative to the expected sort order.
        for b in (1e-2, 1e-4)
        for l2 in (1e-6, 1e-8)
        for f in (6.4e-4, 0.0)
        for d in ("rep_2", "rep_1")
    ]
    frame = pd.DataFrame(rows)
    monkeypatch.setattr(
        rpt, "basin_row_with_ridge", lambda fit: dict(fit), raising=True
    )

    table = rpt.basin_table(frame)
    keys = ["beta0_ridge", "l2reg", "fusionreg", "dataset_name"]
    assert table[keys].values.tolist() == (
        frame.sort_values(keys).reset_index(drop=True)[keys].values.tolist()
    )
    # The outermost key really is beta0_ridge, not l2reg (the prior art's order).
    assert table["beta0_ridge"].is_monotonic_increasing


def test_shift_corr_pivot_keeps_only_shift_rows():
    """shift_corr_pivot drops beta_*/predicted_* and pivots shift_* by fusionreg.

    The real correlation frame carries a row per mutation-parameter type; #284's
    reproducibility criterion is the shift_* rows only. Synthetic — always runs.
    """
    import beta0_ridge_l2_report as rpt

    corr = pd.DataFrame(
        [
            {
                "datasets": "rep_1,rep_2",
                "mut_param": mp,
                "correlation": 0.5,
                "fusionreg": f,
                "beta0_ridge": 1e-4,
                "l2reg": 1e-8,
            }
            for mp in (
                "beta_Delta",
                "shift_Delta",
                "predicted_func_score_Delta",
                "shift_Omicron_BA2",
            )
            for f in (0.0, 6.4e-4)
        ]
    )
    out = rpt.shift_corr_pivot(corr)
    assert len(out) == 2, "only shift_Delta and shift_Omicron_BA2 survive"
    assert set(out["mut_param"]) == {"shift_Delta", "shift_Omicron_BA2"}
    for col in (0.0, 6.4e-4):
        assert col in out.columns, f"fusionreg {col} should be a pivoted column"


def test_shift_corr_pivot_handles_empty_input():
    """An empty or shift-less correlation frame yields an empty pivot, not a raise."""
    import beta0_ridge_l2_report as rpt

    assert not len(rpt.shift_corr_pivot(pd.DataFrame()))
    beta_only = pd.DataFrame(
        [
            {
                "datasets": "rep_1,rep_2",
                "mut_param": "beta_Delta",
                "correlation": 0.5,
                "fusionreg": 0.0,
                "beta0_ridge": 1e-4,
                "l2reg": 1e-8,
            }
        ]
    )
    assert not len(rpt.shift_corr_pivot(beta_only))


@pytest.mark.skipif(
    _sample_pickle() is None,
    reason="needs a sample fit_collection.pkl (set BETA0_RIDGE_TEST_PKL)",
)
def test_basin_row_with_ridge_adds_beta0_ridge():
    """basin_row_with_ridge returns basin_row's keys plus beta0_ridge."""
    import beta0_ridge_l2_report as rpt

    frame = pickle.load(open(_sample_pickle(), "rb"))
    out = rpt.basin_row_with_ridge(frame.iloc[0])
    assert "beta0_ridge" in out
    for key in (
        "l2reg",
        "fusionreg",
        "dataset_name",
        "sum_beta_sq",
        "alpha",
        "converged",
        "final_obj_err",
    ):
        assert key in out, f"lost {key} from basin_row's contract"
    assert out["sum_beta_sq"] >= 0.0
    assert isinstance(out["converged"], bool)


@pytest.mark.skipif(
    _sample_pickle() is None,
    reason="needs a sample fit_collection.pkl (set BETA0_RIDGE_TEST_PKL)",
)
def test_basin_table_sorts_by_beta0_ridge_first():
    """basin_table sorts by (beta0_ridge, l2reg, fusionreg, dataset_name)."""
    import beta0_ridge_l2_report as rpt

    frame = pickle.load(open(_sample_pickle(), "rb"))
    table = rpt.basin_table(frame)
    assert len(table) == len(frame)
    expected = table.sort_values(
        ["beta0_ridge", "l2reg", "fusionreg", "dataset_name"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(table, expected)
