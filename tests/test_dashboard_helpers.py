"""Unit tests for the dashboard's pure helper functions."""

from pathlib import Path

import pytest

from experiments.dashboard_helpers import (
    discover_pickles,
    load_collection,
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
