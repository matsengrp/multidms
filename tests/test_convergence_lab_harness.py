"""Unit tests for the convergence-lab harness (issue #253)."""

import math
import sys
from pathlib import Path

import pytest
import yaml

# The harness lives under experiments/, not on the package path; import by path.
HARNESS_DIR = Path(__file__).parent.parent / "experiments" / "convergence-lab"
sys.path.insert(0, str(HARNESS_DIR))

import harness  # noqa: E402


def _write_yaml(tmp_path, body):
    """Write ``body`` as a YAML grid file under ``tmp_path`` and return its path."""
    p = tmp_path / "grid.yaml"
    p.write_text(yaml.safe_dump(body))
    return p


def test_load_config_defaults_replicates(tmp_path):
    """load_config defaults replicates to [1, 2] and leaves sweep/fixed intact."""
    cfg = harness.load_config(_write_yaml(tmp_path, {"sweep": {"l2reg": [0.0, 3e-4]}}))
    assert cfg["replicates"] == [1, 2]
    assert cfg["sweep"] == {"l2reg": [0.0, 3e-4]}
    assert cfg["fixed"] == {}


def test_load_config_rejects_unknown_sweep_key(tmp_path):
    """An unknown key in the sweep section raises ValueError naming the key."""
    with pytest.raises(ValueError, match="not_a_kwarg"):
        harness.load_config(_write_yaml(tmp_path, {"sweep": {"not_a_kwarg": [1, 2]}}))


def test_load_config_rejects_unknown_fixed_key(tmp_path):
    """An unknown key in the fixed section raises ValueError naming the key."""
    with pytest.raises(ValueError, match="bogus"):
        harness.load_config(
            _write_yaml(
                tmp_path,
                {"sweep": {"l2reg": [0.0]}, "fixed": {"bogus": True}},
            )
        )


def test_load_config_rejects_dataset_key(tmp_path):
    """`dataset` is runner-supplied, not a config key, so it is rejected."""
    with pytest.raises(ValueError, match="dataset"):
        harness.load_config(_write_yaml(tmp_path, {"sweep": {"dataset": [1]}}))


def test_load_config_requires_sweep(tmp_path):
    """A config with no sweep section raises ValueError."""
    with pytest.raises(ValueError, match="sweep"):
        harness.load_config(_write_yaml(tmp_path, {"fixed": {"warmstart": True}}))


def test_explode_grid_cartesian_with_replicates(tmp_path):
    """explode_grid crosses sweep x replicates and merges fixed into each cell."""
    cfg = harness.load_config(
        _write_yaml(
            tmp_path,
            {
                "sweep": {"l2reg": [0.0, 3e-4]},
                "fixed": {"warmstart": True},
                "replicates": [1, 2],
            },
        )
    )
    exploded = harness.explode_grid(cfg)
    assert len(exploded) == 4  # 2 l2reg × 2 replicates
    assert all(d["warmstart"] is True for d in exploded)
    assert {d["replicate"] for d in exploded} == {1, 2}
    assert {d["l2reg"] for d in exploded} == {0.0, 3e-4}


@pytest.mark.slow
def test_load_rep_data_builds_two_reps():
    """load_rep_data builds rep_1/rep_2 Data with the expected conditions."""
    rep_data = harness.load_rep_data()
    assert set(rep_data) == {"rep_1", "rep_2"}
    for name, data in rep_data.items():
        assert data.name == name
        assert "Omicron_BA1" in data.conditions
        assert "Delta" in data.conditions


def test_basin_metrics_keys_on_failure():
    """On a non-model object every field fails: all keys present, NaN/False."""
    out = harness.basin_metrics(object())
    assert set(out) == {
        "alpha_final",
        "beta_l2_norm",
        "max_abs_phi",
        "final_obj_err",
        "converged",
    }
    assert out["converged"] is False
    assert math.isnan(out["alpha_final"])


def test_primary_axis_prefers_fusionreg():
    """primary_axis returns fusionreg when it is among the swept keys."""
    cfg = {"sweep": {"l2reg": [0.0], "fusionreg": [0.0, 8e-5]}}
    assert harness.primary_axis(cfg) == "fusionreg"


def test_primary_axis_falls_back_to_first_sweep_key():
    """primary_axis falls back to the first sweep key when fusionreg is absent."""
    cfg = {"sweep": {"l2reg": [0.0, 3e-4]}}
    assert harness.primary_axis(cfg) == "l2reg"
