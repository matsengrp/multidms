"""Unit tests for the dashboard's pure helper functions."""

from pathlib import Path

import pandas as pd
import pytest

from experiments.dashboard_helpers import (
    constant_summary,
    discover_pickles,
    display_table_df,
    load_collection,
    selectable_columns,
    varying_columns,
)


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_discover_pickles_finds_all_pkls_labeled_by_relative_path(tmp_path):
    """Every ``*.pkl`` is found and labeled by its path relative to root."""
    _touch(tmp_path / "a" / "fit_collection.pkl")
    _touch(tmp_path / "b" / "deep" / "other.pkl")
    found = discover_pickles(tmp_path)
    assert set(found.keys()) == {"a/fit_collection.pkl", "b/deep/other.pkl"}
    for label, path in found.items():
        assert path.is_absolute()
        assert str(path.relative_to(tmp_path)) == label


def test_discover_pickles_includes_dot_dirs_and_is_sorted(tmp_path):
    """Dot-directory matches are kept and entries are sorted by path."""
    _touch(tmp_path / "z.pkl")
    _touch(tmp_path / ".worktrees" / "x" / "m.pkl")
    _touch(tmp_path / "a.pkl")
    found = discover_pickles(tmp_path)
    assert list(found.keys()) == [".worktrees/x/m.pkl", "a.pkl", "z.pkl"]


def test_discover_pickles_empty_when_none(tmp_path):
    """An empty mapping is returned when no pkl exists below the root."""
    assert discover_pickles(tmp_path) == {}


def test_load_collection_wraps_fit_models_shaped_object(tmp_path, monkeypatch):
    """A non-collection pickle is coerced via ``ModelCollection(loaded)``."""
    import pickle

    import experiments.dashboard_helpers as helpers

    captured = {}

    class _StubMC:
        def __init__(self, loaded):
            captured["wrapped"] = loaded

    # Patch the name load_collection looks up so we don't build a real fit.
    monkeypatch.setattr(helpers, "ModelCollection", _StubMC)

    not_a_collection = {"fit_models": "stand-in"}
    p = tmp_path / "c.pkl"
    p.write_bytes(pickle.dumps(not_a_collection))

    result = load_collection(p)
    assert isinstance(result, _StubMC)
    assert captured["wrapped"] == not_a_collection


def test_load_collection_raises_on_garbage(tmp_path):
    """A non-pickle file raises so the caller can show a friendly error."""
    p = tmp_path / "bad.pkl"
    p.write_bytes(b"not a pickle at all")
    with pytest.raises(Exception):
        load_collection(p)


def _fit_models_fixture():
    """A small ``fit_models``-shaped DataFrame for table-builder tests."""
    return pd.DataFrame(
        {
            "dataset_name": ["A", "A", "B"],
            "fusionreg": [0.0, 0.123456, 0.0],
            "l2reg": [0.0, 0.0, 0.0],
            "converged": [True, True, False],
            "ge_kwargs": [{"k": 1}, {"k": 1}, {"k": 1}],  # excluded (object)
            "beta_init": [object(), object(), object()],  # excluded (_init)
            "total_loss_training": [[1, 2], [1, 2], [1, 2]],  # excluded
            "model": [object(), object(), object()],  # excluded (model)
        }
    )


def test_selectable_columns_drops_model_and_bookkeeping():
    """``model`` and ``*_init``/``*_kwargs``/``*_loss_training`` are dropped."""
    cols = selectable_columns(_fit_models_fixture())
    assert "model" not in cols
    assert "ge_kwargs" not in cols
    assert "beta_init" not in cols
    assert "total_loss_training" not in cols
    assert {"dataset_name", "fusionreg", "l2reg", "converged"} <= set(cols)


def test_varying_columns_only_nonconstant():
    """Only columns that differ across at least two fits are reported."""
    varying = varying_columns(_fit_models_fixture())
    assert "fusionreg" in varying  # 0.0 vs 0.123456
    assert "dataset_name" in varying  # A vs B
    assert "l2reg" not in varying  # all 0.0
    assert "model" not in varying


def test_constant_summary_reports_constants_only():
    """Constant selectable columns appear; varying columns do not."""
    summary = constant_summary(_fit_models_fixture())
    assert summary["l2reg"] == 0.0
    assert "fusionreg" not in summary
    assert "model" not in summary


def test_display_table_df_shape_and_rounding():
    """Table starts with dataset_name, hides constants, keeps ``_fit_idx``."""
    df = display_table_df(_fit_models_fixture(), round_to=4)
    # dataset_name first, converged present, _fit_idx retained
    assert list(df.columns)[0] == "dataset_name"
    assert "converged" in df.columns
    assert "_fit_idx" in df.columns
    assert "model" not in df.columns
    assert "l2reg" not in df.columns  # constant -> hidden
    # rounding: 0.123456 -> 0.1235
    assert 0.1235 in set(df["fusionreg"].round(4))
    # one row per fit, indices 0..2
    assert list(df["_fit_idx"]) == [0, 1, 2]
