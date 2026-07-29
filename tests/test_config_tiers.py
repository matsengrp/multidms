"""Tests for two-tier config loading in both experiment pipelines.

The pipeline configs are split into a fit tier and a downstream tier so a
downstream-only edit cannot invalidate the cached model fit. These tests pin
the loader contract that split depends on.
"""

import importlib.util
import os

import pytest
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_DIR = os.path.join(REPO_ROOT, "experiments", "simulation")
SPIKE_DIR = os.path.join(REPO_ROOT, "experiments", "scv2-spike")


def _load_common(pipeline_dir, module_name):
    """Import a pipeline's notebooks/_common.py under a unique module name."""
    path = os.path.join(pipeline_dir, "notebooks", "_common.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SIM_COMMON = _load_common(SIM_DIR, "sim_common_for_tests")
SPIKE_COMMON = _load_common(SPIKE_DIR, "spike_common_for_tests")

PIPELINES = [
    pytest.param(SIM_COMMON.load_config, "simulation", id="simulation"),
    pytest.param(SPIKE_COMMON.load_config, "spike", id="spike"),
]


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload))
    return str(path)


@pytest.mark.parametrize("load_config,section", PIPELINES)
def test_single_tier_unchanged(load_config, section, tmp_path):
    """Loading without a downstream path returns the file verbatim."""
    fit = _write(tmp_path, "fit.yaml", {"seed": 4, section: {"output_dir": "results"}})
    assert load_config(fit) == {"seed": 4, section: {"output_dir": "results"}}


@pytest.mark.parametrize("load_config,section", PIPELINES)
def test_merges_downstream_section(load_config, section, tmp_path):
    """Downstream keys land in the same section as fit keys."""
    fit = _write(tmp_path, "fit.yaml", {"seed": 4, section: {"output_dir": "results"}})
    down = _write(tmp_path, "down.yaml", {section: {"lasso_choice": 8.0e-5}})
    merged = load_config(fit, down)
    assert merged[section]["output_dir"] == "results"
    assert merged[section]["lasso_choice"] == 8.0e-5


@pytest.mark.parametrize("load_config,section", PIPELINES)
def test_collision_raises(load_config, section, tmp_path):
    """A key in both tiers is an error, not a last-wins resolution."""
    fit = _write(tmp_path, "fit.yaml", {section: {"lasso_choice": 1.0}})
    down = _write(tmp_path, "down.yaml", {section: {"lasso_choice": 2.0}})
    with pytest.raises(ValueError, match="lasso_choice"):
        load_config(fit, down)


@pytest.mark.parametrize("load_config,section", PIPELINES)
def test_top_level_collision_raises(load_config, section, tmp_path):
    """Collisions are detected at the top level too."""
    fit = _write(tmp_path, "fit.yaml", {"seed": 4})
    down = _write(tmp_path, "down.yaml", {"seed": 5})
    with pytest.raises(ValueError, match="seed"):
        load_config(fit, down)


@pytest.mark.parametrize("load_config,section", PIPELINES)
def test_missing_downstream_file_raises(load_config, section, tmp_path):
    """A missing downstream file fails loudly rather than silently skipping."""
    fit = _write(tmp_path, "fit.yaml", {section: {}})
    with pytest.raises(FileNotFoundError):
        load_config(fit, str(tmp_path / "nope.yaml"))
