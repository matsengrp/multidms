"""Pure, testable helpers for the multidms marimo dashboard.

These functions hold all logic worth unit-testing (recursive pickle
discovery, collection loading, varying-column detection, pandas-query
synthesis from selected rows, and the two-fit merge). The marimo notebook
``dashboard.py`` imports them inside cells so they are visible to those
cells (marimo only exposes names a cell imports or that an ancestor cell
returns).
"""

from pathlib import Path

import pandas as pd

from multidms.model_collection import ModelCollection

#: Columns excluded from any fit table because ``mo.ui.table`` cannot render
#: them (model objects, dict/object bookkeeping cells, per-row noise).
EXCLUDED_TABLE_COLUMNS = ("model",)
EXCLUDED_TABLE_SUFFIXES = ("_loss_training", "_init", "_kwargs")


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


def selectable_columns(fit_models):
    """Columns of ``fit_models`` eligible to appear in a fit table.

    Drops the ``model`` object column and every bookkeeping column whose name
    ends in one of :data:`EXCLUDED_TABLE_SUFFIXES`.

    Args:
        fit_models: A ``ModelCollection.fit_models`` DataFrame.

    Returns:
        List of column names, in the DataFrame's column order.
    """
    cols = []
    for c in fit_models.columns:
        if c in EXCLUDED_TABLE_COLUMNS:
            continue
        if any(c.endswith(suffix) for suffix in EXCLUDED_TABLE_SUFFIXES):
            continue
        cols.append(c)
    return cols


def varying_columns(fit_models):
    """Selectable columns whose value differs across at least two fits.

    Args:
        fit_models: A ``ModelCollection.fit_models`` DataFrame.

    Returns:
        List of column names (subset of :func:`selectable_columns`) that are
        non-constant across rows. Comparison is by stringified value so that
        unhashable cells do not raise.
    """
    varying = []
    for c in selectable_columns(fit_models):
        if fit_models[c].astype(str).nunique(dropna=False) > 1:
            varying.append(c)
    return varying


def constant_summary(fit_models):
    """Map of selectable columns that are constant across all fits.

    Args:
        fit_models: A ``ModelCollection.fit_models`` DataFrame.

    Returns:
        Dict ``{column: the single value}`` for every selectable column whose
        value is identical across all rows. Suitable for a caption above the
        table.
    """
    summary = {}
    for c in selectable_columns(fit_models):
        if fit_models[c].astype(str).nunique(dropna=False) <= 1:
            summary[c] = fit_models[c].iloc[0]
    return summary


def display_table_df(fit_models, *, round_to=4):
    """Build the per-tab fit table (display only; do not query off this).

    Column order: ``dataset_name``, then the varying columns, then
    ``converged``; floats rounded to ``round_to`` decimals **for display
    only**. The original positional row index is preserved as ``_fit_idx`` so
    a selection can be mapped back to un-rounded source rows.

    Args:
        fit_models: A ``ModelCollection.fit_models`` DataFrame.
        round_to: Decimal places for float display rounding.

    Returns:
        A DataFrame with one row per fit and columns
        ``[dataset_name, *varying-minus-dataset_name-and-converged,
        converged, _fit_idx]``.
    """
    varying = varying_columns(fit_models)
    ordered = ["dataset_name"]
    for c in varying:
        if c not in ("dataset_name", "converged"):
            ordered.append(c)
    if "converged" in fit_models.columns:
        ordered.append("converged")

    df = fit_models.reset_index(drop=True).copy()
    df["_fit_idx"] = range(len(df))
    out = df[ordered + ["_fit_idx"]].copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(round_to)
    return out
