"""Shared utilities for the spike manuscript-figure notebook.

This module is a deliberate sibling of ``_common.py`` rather than an addition
to it. ``_common.py`` is declared ``input:`` on all four fit-tier rules, so a
helper added there would make an edit to figure code invalidate the cached
model fit -- a 2 h 20 min refit, as measured on the prod run logged in
``experiments/scv2-spike/README.md``. This module is declared ``input:`` on
``rule manuscript_figures`` only, which keeps figure work in the downstream
tier where it belongs. See ``experiments/scv2-spike/README.md`` ("Configuration
tiers") and issue #287.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import requests

# The legacy analysis repository, pinned to the commit the rest of EPIC #290
# references. Deliberately NOT `main`, which is where `spike.data_url` points:
# the validation data must not drift underneath a published figure.
VALIDATION_DATA_COMMIT = "6c98b7b607d7387b508cdaa192d659ee9fca7367"
VALIDATION_DATA_URL = (
    "https://raw.githubusercontent.com/matsengrp/SARS-CoV-2_spike_multidms/"
    f"{VALIDATION_DATA_COMMIT}/data"
)

#: Files fetched by :func:`fetch_validation_data`, in the order returned.
VALIDATION_FILES = ("viral_titers.csv", "spike_validation_data.csv")


def fetch_validation_data(results_dir):
    """Download the viral-titer validation data used by Figure 5.

    The two files are standalone repository-level CSVs, unlike everything
    ``_common.download_data`` handles: that helper is condition-scoped, walking
    ``experiment_conditions`` and reading each condition's
    ``functional_selections.csv`` manifest. It has no slot for a repo-level
    file, so bending it to carry these two would distort a helper that four
    rules depend on.

    Files are cached under ``{results_dir}/raw_data/validation/`` and skipped
    if already present, matching ``_common.download_data``'s behaviour.

    Parameters
    ----------
    results_dir : str
        The pipeline's results directory. The cache is created beneath it.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(titers, validation)``. ``titers`` has columns ``virus``,
        ``background``, ``plate``, ``RLUperuL``. ``validation`` is wide, with
        one column per validation mutation and one row per
        (background, replicate).
    """
    cache_dir = os.path.join(results_dir, "raw_data", "validation")
    os.makedirs(cache_dir, exist_ok=True)

    frames = []
    for fname in VALIDATION_FILES:
        path = os.path.join(cache_dir, fname)
        if not os.path.exists(path):
            url = f"{VALIDATION_DATA_URL}/{fname}"
            print(f"  Downloading {url}")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(path, "w") as f:
                f.write(resp.text)
        frames.append(pd.read_csv(path))

    return tuple(frames)


def savefig(fig, name, figures_dir, formats=("pdf", "png"), dpi=300):
    """Save a figure to the pipeline's figures directory in several formats.

    Every manuscript figure is written as both a vector PDF (what the
    manuscript build consumes) and a raster PNG (what the docs page shows).
    Both are declared as rule outputs, so a figure that fails to render fails
    the pipeline run rather than silently vanishing into notebook output.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    name : str
        Base filename without extension, e.g.
        ``"shift_by_site_heatmap_zoom"``.
    figures_dir : str
        Destination directory. Created if absent.
    formats : sequence of str
        Extensions to write. Defaults to PDF and PNG.
    dpi : int
        Raster resolution.

    Returns
    -------
    list of str
        The paths written, in ``formats`` order.
    """
    os.makedirs(figures_dir, exist_ok=True)

    paths = []
    for ext in formats:
        path = os.path.join(figures_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        paths.append(path)
        print(f"  wrote {path}")

    return paths


def lasso_slice(df, lasso_choice, column="fusionreg"):
    """Select the rows of a multi-lambda table at the chosen lasso weight.

    Guards the failure mode that broke the test profile in this pipeline once
    already: an off-ladder ``lasso_choice`` matches no rows, and the resulting
    empty frame fails much later with an error naming neither the config nor
    the lambda. See ``tests/test_config_tiers.py``.

    Parameters
    ----------
    df : pandas.DataFrame
        A table carrying one row group per regularization weight.
    lasso_choice : float
        The chosen weight, from ``spike.lasso_choice`` in the downstream
        config.
    column : str
        Name of the regularization-weight column.

    Returns
    -------
    pandas.DataFrame
        The matching rows.

    Raises
    ------
    ValueError
        If no row matches, listing the weights that are present.
    """
    out = df[df[column] == lasso_choice]
    if out.empty:
        raise ValueError(
            f"lasso_choice={lasso_choice:g} matched no rows on '{column}'. "
            f"Present values: {sorted(df[column].unique())}"
        )
    return out


def set_plot_style(rc=None):
    """Apply the manuscript's matplotlib defaults.

    Parameters
    ----------
    rc : dict or None
        Extra ``matplotlib.rcParams`` entries, merged last so a caller can
        override any default here.
    """
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,  # embed TrueType, not Type 3 -- journals require it
            "ps.fonttype": 42,
        }
    )
    if rc:
        plt.rcParams.update(rc)
