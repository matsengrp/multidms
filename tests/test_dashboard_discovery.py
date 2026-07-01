"""Tests for the dashboard's recursive pickle discovery helper."""

import ast
from pathlib import Path

from experiments.dashboard_helpers import discover_pickles

#: The marimo notebook that imports the helper module.
_DASHBOARD_PY = Path(__file__).resolve().parents[1] / "experiments" / "dashboard.py"


def _touch(p: Path) -> None:
    """Create an empty file at ``p``, making parent directories as needed."""
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_discovers_nested_pkls_labeled_by_relative_path(tmp_path):
    """Pkls at any depth are found, labeled by path relative to root."""
    _touch(tmp_path / "experiments" / "sim" / "results-test" / "fit_collection.pkl")
    _touch(tmp_path / "deep" / "a" / "b" / "run1" / "other.pkl")

    found = discover_pickles(tmp_path)

    assert set(found.keys()) == {
        "experiments/sim/results-test/fit_collection.pkl",
        "deep/a/b/run1/other.pkl",
    }
    for label, path in found.items():
        assert path.is_absolute()
        assert str(path.relative_to(tmp_path)) == label


def test_matches_any_pkl_extension(tmp_path):
    """Every ``*.pkl`` is discovered; non-pkl extensions are ignored."""
    _touch(tmp_path / "good" / "fit_collection.pkl")
    _touch(tmp_path / "good" / "other.pkl")
    _touch(tmp_path / "bad" / "fit_collection.pickle")

    found = discover_pickles(tmp_path)

    assert set(found.keys()) == {
        "good/fit_collection.pkl",
        "good/other.pkl",
    }


def test_prunes_dot_hidden_dirs(tmp_path):
    """Matches under a dot-hidden directory component are pruned.

    Stray or duplicate collections buried in hidden trees (``.worktrees``,
    ``.pixi``, ``.git``, a nested ``.hidden``) must not pollute the picker,
    while visible directories such as ``results/`` are kept. A dot in the
    *filename* does not prune — only directory components count.
    """
    _touch(tmp_path / ".worktrees" / "x" / "fit_collection.pkl")  # pruned
    _touch(tmp_path / ".pixi" / "y" / "a.pkl")  # pruned
    _touch(tmp_path / ".git" / "g.pkl")  # pruned
    _touch(tmp_path / "sub" / ".hidden" / "c.pkl")  # pruned (nested)
    _touch(tmp_path / "results" / "run1" / "fit_collection.pkl")  # kept
    _touch(tmp_path / "top.pkl")  # kept

    found = discover_pickles(tmp_path)

    assert set(found.keys()) == {
        "results/run1/fit_collection.pkl",
        "top.pkl",
    }


def test_dot_hidden_root_keeps_visible_descendants(tmp_path):
    """Only components *below* root are tested, so a dot-hidden root works.

    When the dashboard is launched from inside a ``.worktrees`` checkout,
    ``root`` itself lives under a dot-hidden directory; pickles in its
    visible subdirectories must still be discovered.
    """
    root = tmp_path / ".worktrees" / "258-x"
    _touch(root / "results" / "f.pkl")

    found = discover_pickles(root)

    assert set(found.keys()) == {"results/f.pkl"}


def test_empty_when_none_present(tmp_path):
    """An empty mapping is returned when no pkl exists below the root."""
    assert discover_pickles(tmp_path) == {}


def test_results_are_sorted_by_path(tmp_path):
    """Discovered entries are ordered deterministically by path."""
    _touch(tmp_path / "z" / "a.pkl")
    _touch(tmp_path / "a" / "a.pkl")
    _touch(tmp_path / "m" / "a.pkl")

    found = discover_pickles(tmp_path)

    assert list(found.keys()) == ["a/a.pkl", "m/a.pkl", "z/a.pkl"]


def test_dashboard_imports_helper_as_top_level_module():
    """The notebook must import the helper as a top-level module.

    marimo always puts the notebook's own directory (``experiments/``) on
    ``sys.path`` but not the repo root, so ``from dashboard_helpers import
    ...`` resolves regardless of the directory the dashboard is launched
    from. ``from experiments.dashboard_helpers import ...`` only resolves
    when the repo root happens to be importable (i.e. launched from the
    repo root) and raises ``ModuleNotFoundError: No module named
    'experiments'`` when launched from any results directory below cwd —
    which is the dashboard's core use case. Guard the import form so that
    regression cannot slip back in.
    """
    tree = ast.parse(_DASHBOARD_PY.read_text())
    helper_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.endswith("dashboard_helpers")
    ]
    assert helper_imports, "dashboard.py never imports dashboard_helpers"
    for node in helper_imports:
        assert node.module == "dashboard_helpers", (
            f"dashboard.py imports the helper as {node.module!r}; it must be "
            "the top-level module 'dashboard_helpers' so the marimo launch "
            "works from any directory (see this test's docstring)."
        )
