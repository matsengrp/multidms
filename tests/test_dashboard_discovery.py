"""Tests for the dashboard's fit_collection.pkl discovery helper."""

from pathlib import Path

from experiments.dashboard import _discover_collections


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_discovers_nested_pkls_labeled_by_relative_parent(tmp_path):
    _touch(tmp_path / "experiments" / "sim" / "results-test" / "fit_collection.pkl")
    _touch(tmp_path / "deep" / "a" / "b" / "run1" / "fit_collection.pkl")

    found = _discover_collections(tmp_path)

    assert set(found.keys()) == {
        "experiments/sim/results-test",
        "deep/a/b/run1",
    }
    for label, path in found.items():
        assert path.is_absolute()
        assert path.name == "fit_collection.pkl"
        assert str(path.parent.relative_to(tmp_path)) == label


def test_only_exact_filename_matches(tmp_path):
    _touch(tmp_path / "good" / "fit_collection.pkl")
    _touch(tmp_path / "bad" / "other.pkl")
    _touch(tmp_path / "bad" / "fit_collection.pickle")

    found = _discover_collections(tmp_path)

    assert list(found.keys()) == ["good"]


def test_no_pruning_includes_dot_dirs(tmp_path):
    _touch(tmp_path / ".worktrees" / "x" / "fit_collection.pkl")
    _touch(tmp_path / ".pixi" / "y" / "fit_collection.pkl")

    found = _discover_collections(tmp_path)

    assert ".worktrees/x" in found
    assert ".pixi/y" in found


def test_empty_when_none_present(tmp_path):
    assert _discover_collections(tmp_path) == {}


def test_results_are_sorted_by_path(tmp_path):
    _touch(tmp_path / "z" / "fit_collection.pkl")
    _touch(tmp_path / "a" / "fit_collection.pkl")
    _touch(tmp_path / "m" / "fit_collection.pkl")

    found = _discover_collections(tmp_path)

    assert list(found.keys()) == ["a", "m", "z"]
