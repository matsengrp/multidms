"""Grid-config test: the beta0_ridge × l2reg scan expands to exactly 72 cells (#284).

Pure config validation — no fitting, no pickle, so this always runs (unlike the
report tests, which need a fit collection). Guards the two things the #284
acceptance criteria pin: the cell count and the inherited clip bound.
"""

import os
import sys
from pathlib import Path

DIAG = Path(__file__).resolve().parent
sys.path.insert(0, str(DIAG.parent))

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

GRID = DIAG.parent / "grids" / "beta0-ridge-l2-scan.yaml"


def test_grid_expands_to_72_cells():
    """4 fusionreg × 3 beta0_ridge × 3 l2reg × 2 replicates = 72 fits."""
    import harness

    config = harness.load_config(GRID)
    cells = harness.explode_grid(config)
    assert len(cells) == 72, f"expected 72 cells, got {len(cells)}"


def test_grid_sweeps_the_three_axes():
    """The three swept axes carry exactly the specified values."""
    import harness

    config = harness.load_config(GRID)
    assert config["sweep"]["fusionreg"] == [0.0, 4.0e-5, 1.6e-4, 6.4e-4]
    assert config["sweep"]["beta0_ridge"] == [1.0e-4, 1.0e-3, 1.0e-2]
    assert config["sweep"]["l2reg"] == [1.0e-8, 1.0e-7, 1.0e-6]
    assert config["replicates"] == [1, 2]


def test_grid_inherits_the_clip_bound():
    """The #263 clip bound and the baseline's regime flags are inherited verbatim."""
    import harness

    config = harness.load_config(GRID)
    assert config["fixed"]["beta_clip_range"] == [-10, 10]
    assert config["fixed"]["output_floor"] is None
    assert config["fixed"]["share_alpha"] is True
    assert config["fixed"]["recompute_scale"] is False


def test_grid_matches_the_baseline_fixed_block():
    """Every fixed key equals softplus-floor-off's, minus the two swept axes.

    This is the #284 AC1 check in executable form: the grid must be a
    one-knob-at-a-time delta from its baseline, so any drift in the inherited
    block is a bug, not a choice.
    """
    import harness

    baseline = harness.load_config(DIAG.parent / "grids" / "softplus-floor-off.yaml")
    scan = harness.load_config(GRID)

    moved = {"l2reg", "beta0_ridge"}
    assert moved <= set(baseline["fixed"]), "baseline should fix the axes we sweep"
    assert moved <= set(scan["sweep"]), "scan should sweep them"
    assert not (moved & set(scan["fixed"])), "scan must not also fix them"

    expected_fixed = {k: v for k, v in baseline["fixed"].items() if k not in moved}
    assert scan["fixed"] == expected_fixed, "inherited fixed block drifted"
    assert scan["sweep"]["fusionreg"] == baseline["sweep"]["fusionreg"]
    assert scan["replicates"] == baseline["replicates"]
