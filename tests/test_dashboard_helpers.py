"""Unit tests for the dashboard's pure helper functions."""

from pathlib import Path

import pandas as pd
import pytest

from experiments.dashboard_helpers import (
    common_param_columns,
    constant_summary,
    discover_pickles,
    display_table_df,
    load_collection,
    merge_two_fits_on_mutation,
    selectable_columns,
    synthesize_isin_query,
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


def test_synthesize_query_uses_unrounded_values():
    """Query is built from full-precision floats and round-trips selection."""
    fm = _fit_models_fixture()
    # select rows 0 and 1 (dataset A, fusionreg 0.0 and 0.123456)
    q = synthesize_isin_query(fm, [0, 1])
    selected = fm.reset_index(drop=True).query(q)
    assert set(selected.index) == {0, 1}
    # un-rounded float present verbatim (not 0.1235)
    assert "0.123456" in q


def test_synthesize_query_isin_idiom_and_dataset_membership():
    """The query uses ``.isin(...)`` and constrains dataset membership."""
    fm = _fit_models_fixture()
    q = synthesize_isin_query(fm, [0, 2])  # datasets A and B
    selected = fm.reset_index(drop=True).query(q)
    assert set(selected["dataset_name"]) == {"A", "B"}
    assert ".isin(" in q


def test_synthesize_query_all_rows_returns_all():
    """Selecting every fit yields a query that returns every fit."""
    fm = _fit_models_fixture()
    q = synthesize_isin_query(fm, [0, 1, 2])
    assert len(fm.reset_index(drop=True).query(q)) == 3


def test_common_param_columns_intersection_numeric_only():
    """Only numeric columns present in both fits are returned."""
    a = pd.DataFrame(
        {"mutation": ["M1A"], "beta_x": [1.0], "shift_y": [2.0], "label": ["z"]}
    )
    b = pd.DataFrame(
        {"mutation": ["M1A"], "beta_x": [3.0], "predicted_func_score_x": [4.0]}
    )
    cols = common_param_columns(a, b)
    assert cols == ["beta_x"]  # only numeric column in both, mutation excluded


def test_merge_two_fits_suffixes_by_fit_key_not_dataset():
    """Same-named columns are suffixed by fit key, not by dataset name."""
    # Both fits are the SAME dataset/condition -> identical column name.
    a = pd.DataFrame({"mutation": ["M1A", "M2B"], "beta_x": [1.0, 2.0]})
    b = pd.DataFrame({"mutation": ["M1A", "M2B"], "beta_x": [10.0, 20.0]})
    merged, x_col, y_col = merge_two_fits_on_mutation(a, b, "beta_x", key_a=0, key_b=1)
    assert x_col == "beta_x_0"
    assert y_col == "beta_x_1"
    assert list(merged[x_col]) == [1.0, 2.0]
    assert list(merged[y_col]) == [10.0, 20.0]


def test_merge_drops_nan_rows():
    """Rows with a NaN in either fit are dropped from the merge."""
    import numpy as np

    a = pd.DataFrame({"mutation": ["M1A", "M2B"], "beta_x": [1.0, np.nan]})
    b = pd.DataFrame({"mutation": ["M1A", "M2B"], "beta_x": [10.0, 20.0]})
    merged, _, _ = merge_two_fits_on_mutation(a, b, "beta_x", key_a=0, key_b=1)
    assert len(merged) == 1


def _muts_indexed_by_mutation():
    """Mimic ``Model.get_mutations_df()``: mutation is the index, not a col."""
    df = pd.DataFrame(
        {"mutation": ["M1A", "M2B"], "beta_x": [1.0, 2.0], "shift_y": [3.0, 4.0]}
    )
    return df.set_index("mutation")


def test_merge_handles_mutation_as_index():
    """``get_mutations_df`` returns mutation as the index; merge must cope."""
    a = _muts_indexed_by_mutation()
    b = _muts_indexed_by_mutation()
    assert "mutation" not in a.columns  # guard: index, not column
    merged, x_col, y_col = merge_two_fits_on_mutation(a, b, "beta_x", key_a=7, key_b=9)
    assert x_col == "beta_x_7"
    assert y_col == "beta_x_9"
    assert list(merged[x_col]) == [1.0, 2.0]


def test_common_param_columns_handles_mutation_as_index():
    """Common-param detection also works when mutation is the index."""
    a = _muts_indexed_by_mutation()
    b = _muts_indexed_by_mutation()
    assert common_param_columns(a, b) == ["beta_x", "shift_y"]
