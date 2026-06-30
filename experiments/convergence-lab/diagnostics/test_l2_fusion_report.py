"""Smoke test for the l2-fusion report's per-fit metric extraction."""

import os
import pickle
import sys
from pathlib import Path

import pytest

DIAG = Path(__file__).resolve().parent
sys.path.insert(0, str(DIAG))


def _sample_pickle() -> Path | None:
    """Locate a sample fit_collection.pkl to extract from.

    The simulation collection is gitignored, so it is absent from a fresh
    checkout (and from a worktree, where ``results/`` is never materialized).
    Honor a ``L2_FUSION_TEST_PKL`` env override (point it at the canonical
    clone's copy when running from a worktree); otherwise fall back to the
    in-tree path. Returns ``None`` when no fixture is reachable, so the test
    skips rather than fails.
    """
    override = os.environ.get("L2_FUSION_TEST_PKL")
    if override and Path(override).exists():
        return Path(override)
    in_tree = (
        Path(__file__).resolve().parents[3]
        / "experiments/simulation/results/fit_collection.pkl"
    )
    return in_tree if in_tree.exists() else None


@pytest.mark.skipif(
    _sample_pickle() is None,
    reason="needs a sample fit_collection.pkl (set L2_FUSION_TEST_PKL)",
)
def test_basin_row_extracts_finite_numbers():
    """basin_row returns finite Σβ², α, and a bool converged for a real fit."""
    import l2_fusion_report as rpt

    frame = pickle.load(open(_sample_pickle(), "rb"))
    row = frame.iloc[0]
    out = rpt.basin_row(row)
    assert out["sum_beta_sq"] >= 0.0
    assert isinstance(out["converged"], bool)
    assert out["alpha"] == out["alpha"]  # not NaN
