"""Unit tests for the convergence-lab harness (issue #253)."""

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


@pytest.mark.needs_local_data
def test_load_rep_data_builds_two_reps():
    """load_rep_data builds rep_1/rep_2 Data with the expected conditions.

    Reads the gitignored prod spike CSV, so this runs only where the pipeline
    has been run locally.
    """
    rep_data = harness.load_rep_data()
    assert set(rep_data) == {"rep_1", "rep_2"}
    for name, data in rep_data.items():
        assert data.name == name
        assert "Omicron_BA1" in data.conditions
        assert "Delta" in data.conditions


def test_build_params_maps_replicates_and_collects_sweep():
    """build_params maps replicate→Data and collects each kwarg's distinct values.

    The result is a ``fit_models`` ``params`` dict: ``dataset`` is the list of
    rep Data objects, every other key the distinct values seen across cells.
    Crossing dataset × those lists must reproduce exactly the exploded cells.
    """
    rep_data = {"rep_1": "DATA1", "rep_2": "DATA2"}
    exploded = [
        {"l2reg": 0.0, "warmstart": True, "replicate": 1},
        {"l2reg": 3e-4, "warmstart": True, "replicate": 1},
        {"l2reg": 0.0, "warmstart": True, "replicate": 2},
        {"l2reg": 3e-4, "warmstart": True, "replicate": 2},
    ]
    params = harness.build_params(exploded, rep_data)
    assert params["dataset"] == ["DATA1", "DATA2"]
    assert params["l2reg"] == [0.0, 3e-4]
    assert params["warmstart"] == [True]
    assert "replicate" not in params


def test_run_fits_delegates_to_fit_models(monkeypatch):
    """run_fits calls fit_models with the params + n_processes and returns its frame."""
    import pandas as pd

    captured = {}

    def fake_fit_models(params, n_processes, failures):
        captured["params"] = params
        captured["n_processes"] = n_processes
        captured["failures"] = failures
        frame = pd.DataFrame({"l2reg": [0.0, 3e-4], "model": [object(), object()]})
        return (2, 0, frame)

    monkeypatch.setattr(harness, "fit_models", fake_fit_models)
    params = {"dataset": ["D1", "D2"], "l2reg": [0.0, 3e-4]}
    df = harness.run_fits(params, n_processes=3)
    assert captured["n_processes"] == 3
    assert captured["failures"] == "tolerate"
    assert len(df) == 2


def test_default_n_processes_caps_at_grid_size(monkeypatch):
    """default_n_processes never exceeds the grid size and is at least 1."""
    monkeypatch.setattr(harness.os, "cpu_count", lambda: 8)
    # Grid smaller than cores-1 → capped at grid size.
    assert harness.default_n_processes(3) == 3
    # Grid larger than cores-1 → capped at cores-1.
    assert harness.default_n_processes(100) == 7
    # Always at least 1.
    monkeypatch.setattr(harness.os, "cpu_count", lambda: 1)
    assert harness.default_n_processes(10) == 1
