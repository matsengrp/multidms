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


def synthesize_isin_query(fit_models, fit_indices, *, key_columns=None):
    """Build a pandas query string selecting the given fits by membership.

    The query is built from the **un-rounded** source values in
    ``fit_models`` (never the rounded display values), using ``.isin([...])``
    membership so it generalizes the existing
    ``dataset_name.isin([...]) and fusionreg.isin([...])`` idiom in the
    dashboard. The clause for each key column lists the distinct values the
    selected fits take on that column.

    Args:
        fit_models: A ``ModelCollection.fit_models`` DataFrame.
        fit_indices: Positional row indices (``_fit_idx`` values) of the
            selected fits.
        key_columns: Columns to constrain. Defaults to ``dataset_name`` plus
            the varying columns (excluding ``converged``, an outcome rather
            than a selector).

    Returns:
        A pandas query string. Caller passes it to ``fit_models.query(...)``
        or to a method that forwards ``query=`` to ``split_apply_combine``.
    """
    df = fit_models.reset_index(drop=True)
    selected = df.loc[list(fit_indices)]
    if key_columns is None:
        key_columns = ["dataset_name"] + [
            c for c in varying_columns(fit_models) if c != "converged"
        ]
        # de-dup while preserving order
        seen = set()
        key_columns = [c for c in key_columns if not (c in seen or seen.add(c))]

    clauses = []
    for col in key_columns:
        values = list(dict.fromkeys(selected[col].tolist()))  # unique, ordered
        clauses.append(f"{col}.isin({values!r})")
    return " and ".join(clauses)


def _with_mutation_column(muts):
    """Return ``muts`` with ``mutation`` as a column, not the index.

    ``Model.get_mutations_df()`` returns the mutation string as the DataFrame
    *index* (``index.name == "mutation"``), whereas some callers materialize it
    as a column. Normalize to a column so downstream merges are agnostic to
    which form the caller passed.

    Args:
        muts: A per-fit mutation DataFrame.

    Returns:
        A DataFrame guaranteed to have a ``mutation`` column. The input is not
        mutated.
    """
    if "mutation" in muts.columns:
        return muts
    if muts.index.name == "mutation":
        return muts.reset_index()
    return muts


def common_param_columns(muts_a, muts_b):
    """Numeric mutation-parameter columns present in both fits.

    Computes the intersection of the two fits' numeric columns (e.g.
    ``beta_<cond>``, ``shift_<cond>``, ``predicted_func_score_<cond>``),
    excluding ``mutation``. This is derived from the actual fit output rather
    than from the private ``ModelCollection._conditions`` attribute (which
    hand-rolls names and wrongly assumes ``shift_<cond>`` for every
    condition).

    Args:
        muts_a: Mutation DataFrame for the first fit.
        muts_b: Mutation DataFrame for the second fit.

    Returns:
        Sorted list of column names common to both and numeric in both.
    """
    muts_a = _with_mutation_column(muts_a)
    muts_b = _with_mutation_column(muts_b)
    num_a = {
        c
        for c in muts_a.columns
        if c != "mutation" and pd.api.types.is_numeric_dtype(muts_a[c])
    }
    num_b = {
        c
        for c in muts_b.columns
        if c != "mutation" and pd.api.types.is_numeric_dtype(muts_b[c])
    }
    return sorted(num_a & num_b)


def merge_two_fits_on_mutation(muts_a, muts_b, param_col, *, key_a, key_b):
    """Inner-merge two fits on ``mutation`` for one parameter column.

    Both fits expose the parameter under the *same* column name, so the merge
    suffixes by a fit-distinguishing key (``key_a``/``key_b``) rather than by
    dataset name — comparing two fits of the *same* dataset at different
    hyperparameters is an intended case, and dataset-name suffixes would
    collide.

    Args:
        muts_a: Mutation DataFrame for the first (x-axis) fit.
        muts_b: Mutation DataFrame for the second (y-axis) fit.
        param_col: The shared parameter column to compare.
        key_a: Distinguishing key for the first fit (e.g. its ``_fit_idx``).
        key_b: Distinguishing key for the second fit.

    Returns:
        Tuple ``(merged_df, x_col, y_col)`` with NaN rows dropped, where
        ``x_col = f"{param_col}_{key_a}"`` and ``y_col = f"{param_col}_{key_b}"``.
    """
    left = _with_mutation_column(muts_a)[["mutation", param_col]]
    right = _with_mutation_column(muts_b)[["mutation", param_col]]
    merged = left.merge(
        right, on="mutation", suffixes=(f"_{key_a}", f"_{key_b}")
    ).dropna()
    return merged, f"{param_col}_{key_a}", f"{param_col}_{key_b}"
