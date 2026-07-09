"""Unit tests for the Phase 1 β-control report (#263)."""

import os
import pickle
import sys
from pathlib import Path

import pandas as pd
import pytest

DIAG = Path(__file__).resolve().parent
sys.path.insert(0, str(DIAG))


def _sample_pickle() -> Path | None:
    """Locate a sample fit_collection.pkl for the end-to-end test.

    The convergence-lab pickles are gitignored (absent in a fresh worktree).
    Honor a ``BETA_CONTROL_TEST_PKL`` env override (point it at a canonical
    clone's copy); otherwise fall back to any beta-control pickle in-tree.
    Returns ``None`` when none is reachable, so the end-to-end test skips.
    """
    override = os.environ.get("BETA_CONTROL_TEST_PKL")
    if override and Path(override).exists():
        return Path(override)
    for cache in ("beta-control-clip", "beta-control-l2"):
        p = DIAG.parent / "results" / cache / "fit_collection.pkl"
        if p.exists():
            return p
    return None


def test_arm_label_treats_none_and_nan_as_l2():
    """[-10,10]→clip; both None and NaN→l2 (object-column coercion case)."""
    import beta_control_report as rpt

    frame = pd.DataFrame({"beta_clip_range": [[-10, 10], None, float("nan")]})
    labels = frame["beta_clip_range"].map(rpt.arm_label).tolist()
    assert labels == ["clip", "l2", "l2"]


def test_first_below_crossing_and_sentinel():
    """first_below finds the 1-based crossing index, else the 101 sentinel."""
    import beta_control_report as rpt

    traj = [1.0, 1e-3, 1e-5, 1e-7]
    assert rpt.first_below(traj, 1e-4) == 3  # index 2, 1-based
    assert rpt.first_below(traj, 1e-6) == 4  # index 3, 1-based
    never = [1.0, 1e-2, 1e-3]
    assert rpt.first_below(never, 1e-6) == 101


@pytest.mark.skipif(
    _sample_pickle() is None,
    reason="needs a beta-control fit_collection.pkl (set BETA_CONTROL_TEST_PKL)",
)
def test_end_to_end_tables_nonempty():
    """tagged_frame → basin/maxiter tables are non-empty on a real pickle."""
    import beta_control_report as rpt

    frame = pickle.load(open(_sample_pickle(), "rb"))
    # Reuse the same real frame for both arms; tagged_frame just concatenates
    # and labels, so the test exercises the table builders end-to-end.
    tagged = rpt.tagged_frame(frame, frame)
    assert "arm" in tagged.columns
    assert len(rpt.basin_table(tagged)) == len(tagged)
    assert len(rpt.maxiter_table(tagged)) == len(tagged)
