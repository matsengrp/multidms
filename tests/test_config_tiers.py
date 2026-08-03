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


SIM_VARIANTS = ["config", "config_test"]
SPIKE_VARIANTS = [
    "config",
    "config_test",
    "config_experimental",
    "config_recompute_false",
    "config_recompute_false_test",
    "config_recompute_false_maxiter200",
]
SPIKE_DOWNSTREAM_KEYS = {
    "lasso_choice",
    "condition_colors",
    "condition_titles",
    "domain_dict",
}

VARIANTS = [("simulation", SIM_DIR, n) for n in SIM_VARIANTS] + [
    ("spike", SPIKE_DIR, n) for n in SPIKE_VARIANTS
]


def _section_keys(path, section):
    with open(path) as f:
        return set((yaml.safe_load(f) or {}).get(section, {}))


@pytest.mark.parametrize("section,pipeline_dir,name", VARIANTS)
def test_every_variant_has_a_downstream_sibling(section, pipeline_dir, name):
    """Every config variant in both pipelines has a downstream sibling."""
    config_dir = os.path.join(pipeline_dir, "config")
    assert os.path.exists(os.path.join(config_dir, f"{name}.yaml"))
    assert os.path.exists(os.path.join(config_dir, f"{name}_downstream.yaml"))


@pytest.mark.parametrize("section,pipeline_dir,name", VARIANTS)
def test_downstream_keys_left_the_fit_tier(section, pipeline_dir, name):
    """No downstream key remains in any fit-tier config."""
    config_dir = os.path.join(pipeline_dir, "config")
    expected = {"lasso_choice"} if section == "simulation" else SPIKE_DOWNSTREAM_KEYS
    fit_keys = _section_keys(os.path.join(config_dir, f"{name}.yaml"), section)
    leftover = fit_keys & expected
    assert not leftover, f"{name}.yaml still holds {sorted(leftover)}"


@pytest.mark.parametrize("section,pipeline_dir,name", VARIANTS)
def test_tiers_merge_without_collision(section, pipeline_dir, name):
    """Every variant's two tiers merge cleanly and expose lasso_choice."""
    load_config = (
        SIM_COMMON.load_config if section == "simulation" else SPIKE_COMMON.load_config
    )
    config_dir = os.path.join(pipeline_dir, "config")
    merged = load_config(
        os.path.join(config_dir, f"{name}.yaml"),
        os.path.join(config_dir, f"{name}_downstream.yaml"),
    )
    assert "lasso_choice" in merged[section]


def test_simulation_downstream_tier_holds_only_lasso_choice():
    """Simulation's downstream tier is lasso_choice alone — it has no cosmetics."""
    config_dir = os.path.join(SIM_DIR, "config")
    for name in SIM_VARIANTS:
        with open(os.path.join(config_dir, f"{name}_downstream.yaml")) as f:
            payload = yaml.safe_load(f)
        assert set(payload) == {"simulation"}
        assert set(payload["simulation"]) == {"lasso_choice"}


def test_fit_tier_retains_keys_the_notebook_scan_cannot_see():
    """Snakefile-read and helper-read keys must stay in the fit tier."""
    config_dir = os.path.join(SPIKE_DIR, "config")
    experimental = _section_keys(
        os.path.join(config_dir, "config_experimental.yaml"), "spike"
    )
    assert "skip_cross_validation" in experimental
    prod = _section_keys(os.path.join(config_dir, "config.yaml"), "spike")
    assert "data_url" in prod
    assert "output_dir" in prod
