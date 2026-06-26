"""Pure, testable helpers for the multidms marimo dashboard.

These functions hold all logic worth unit-testing (recursive pickle
discovery, collection loading, varying-column detection, pandas-query
synthesis from selected rows, and the two-fit merge). The marimo notebook
``dashboard.py`` imports them inside cells so they are visible to those
cells (marimo only exposes names a cell imports or that an ancestor cell
returns).
"""

from pathlib import Path

from multidms.model_collection import ModelCollection


def discover_pickles(root):
    """Find every ``*.pkl`` below ``root``.

    Args:
        root: Directory searched recursively (typically ``Path.cwd()``).

    Returns:
        Ordered mapping of ``label -> absolute Path``, one entry per
        ``*.pkl`` found anywhere below ``root``. ``label`` is the pickle's
        path relative to ``root`` *including the filename* (filenames now
        differ between matches). Entries are sorted by path. No directories
        are pruned: matches under ``.worktrees/``, ``.pixi/``, ``results/``
        etc. are intentionally included.
    """
    root = Path(root)
    discovered = {}
    for p in sorted(root.rglob("*.pkl")):
        label = str(p.relative_to(root))
        discovered[label] = p
    return discovered


def load_collection(path):
    """Load a pickle and coerce it to a :class:`ModelCollection`.

    Args:
        path: Filesystem path to a ``*.pkl`` file.

    Returns:
        A :class:`ModelCollection`. If the unpickled object is already a
        ``ModelCollection`` it is returned unchanged; otherwise it is wrapped
        via ``ModelCollection(loaded)``.

    Raises:
        Exception: Anything raised by ``pickle.load`` or the
            ``ModelCollection`` constructor (the caller renders a friendly
            inline error instead of crashing).
    """
    import pickle

    with open(path, "rb") as f:
        loaded = pickle.load(f)
    if isinstance(loaded, ModelCollection):
        return loaded
    return ModelCollection(loaded)
