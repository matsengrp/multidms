r"""Convergence-lab harness (#253): config-driven sequential model fitting.

Reads a grid YAML (a ``sweep:`` Cartesian product + optional ``fixed:``
overrides + optional ``replicates:``), fits each cell sequentially via
``multidms.model_collection.fit_one_model`` (NOT ``fit_models`` — kept
sequential for simplicity), extracts per-fit basin diagnostics into
``df_fits``, computes replicate-shift correlations into ``df_corr``, and
pickles ``{"df_fits": ..., "df_corr": ...}`` to ``results/<cache>.pkl``.

The runner owns the constant scv2-spike data; configs carry only swept
knobs and fixed overrides. See ``README.md`` in this directory.

Usage::

    pixi run python experiments/convergence-lab/harness.py \\
        --config grids/smoke.yaml --cache smoke
"""

from __future__ import annotations

import argparse  # noqa: F401 — used in later tasks (CLI)
import os

# Pin per-process CPU threading BEFORE JAX/XLA imports (multidms imports JAX at
# module load; these env vars are only read at XLA init). Harmless for the
# sequential harness; required if a future config restores parallelism.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import pickle  # noqa: E402,F401 — used in later tasks (persist results)
import time  # noqa: E402,F401 — used in later tasks (timing)
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402,F401 — used in later tasks (corr)
import pandas as pd  # noqa: E402,F401 — used in later tasks (df_fits)
import yaml  # noqa: E402

warnings.filterwarnings("ignore")

import multidms  # noqa: E402,F401 — used in later tasks (Data)
from multidms.model_collection import (  # noqa: E402
    ModelCollection,  # noqa: F401 — used in later tasks
    fit_one_model,  # noqa: F401 — used in later tasks
    stack_fit_models,  # noqa: F401 — used in later tasks
)
from multidms.utils import explode_params_dict  # noqa: E402

# --- Constant data the runner owns (resolved relative to this file) ----------
HARNESS_DIR = Path(os.path.abspath(__file__)).parent
DATA_CSV = (
    HARNESS_DIR.parent
    / "scv2-spike"
    / "results-prod-235-times-seen-threshold"
    / "training_functional_scores.csv"
)
RESULTS_DIR = HARNESS_DIR / "results"
REF = "Omicron_BA1"

# Exact fit_one_model signature (model_collection.py:61) minus `dataset`
# (runner-supplied) and `verbose`. The allowlist for config validation.
VALID_KWARGS = frozenset(
    {
        "ge_type",
        "l2reg",
        "fusionreg",
        "beta0_ridge",
        "scale_fusion_by_n",
        "loss_type",
        "maxiter",
        "tol",
        "warmstart",
        "recompute_scale",
        "beta0_init",
        "beta_init",
        "alpha_init",
        "share_alpha",
        "beta_clip_range",
        "ge_kwargs",
        "cal_kwargs",
        "loss_kwargs",
    }
)


def load_config(path) -> dict:
    """Load and validate a grid YAML.

    The config has two kwargs sections — ``sweep`` (each value a list, crossed
    in a Cartesian product) and ``fixed`` (each value a scalar, applied to every
    fit) — plus an optional harness-level ``replicates`` list (default
    ``[1, 2]``). Every key in ``sweep``/``fixed`` must be a valid
    ``fit_one_model`` kwarg (``VALID_KWARGS``); ``fit_one_model``'s ``**kwargs``
    would otherwise swallow typos silently, so we reject them here.

    Args:
        path: Path to the YAML config.

    Returns:
        ``{"sweep": dict, "fixed": dict, "replicates": list[int]}``.

    Raises:
        ValueError: If ``sweep`` is missing/empty, or any ``sweep``/``fixed``
            key is not in ``VALID_KWARGS`` (the message names the bad key).
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    sweep = raw.get("sweep") or {}
    fixed = raw.get("fixed") or {}
    replicates = raw.get("replicates", [1, 2])

    if not sweep:
        raise ValueError(f"config {path}: must define a non-empty 'sweep:' section")

    for section_name, section in (("sweep", sweep), ("fixed", fixed)):
        for key in section:
            if key not in VALID_KWARGS:
                raise ValueError(
                    f"config {path}: '{key}' in '{section_name}:' is not a valid "
                    f"fit_one_model kwarg. Valid keys: {sorted(VALID_KWARGS)}"
                )

    return {"sweep": dict(sweep), "fixed": dict(fixed), "replicates": list(replicates)}


def explode_grid(config: dict) -> list[dict]:
    """Cartesian-product the sweep × replicates, merging fixed into each cell.

    Args:
        config: The dict returned by :func:`load_config`.

    Returns:
        A list of kwarg dicts, one per fit. Each carries every ``fixed`` key and
        an integer ``replicate`` key (the runner maps it to the rep's Data).
    """
    sweep = dict(config["sweep"])
    sweep["replicate"] = list(config["replicates"])
    exploded = explode_params_dict(sweep)
    for cell in exploded:
        cell.update(config["fixed"])
    return exploded


def load_rep_data() -> dict[str, multidms.Data]:
    """Build one ``multidms.Data`` per replicate from the prod spike CSV.

    Mirrors the pipeline ``fit_models.ipynb``: aggregate ``func_score`` by
    ``(condition, aa_substitutions)`` with ``.mean()`` within each replicate,
    then construct a ``multidms.Data`` with the gap-inclusive alphabet.

    Returns:
        Mapping ``"rep_<n>" -> multidms.Data`` (reference = ``Omicron_BA1``).
    """
    raw = pd.read_csv(DATA_CSV).fillna({"aa_substitutions": ""})
    rep_data: dict[str, multidms.Data] = {}
    for rep in sorted(raw["replicate"].unique()):
        df_rep = raw[raw["replicate"] == rep]
        df_agg = (
            df_rep.groupby(["condition", "aa_substitutions"], dropna=False)
            .agg({"func_score": "mean"})
            .reset_index()
        )
        rep_data[f"rep_{rep}"] = multidms.Data(
            df_agg,
            alphabet=multidms.AAS_WITHSTOP_WITHGAP,
            reference=REF,
            assert_site_integrity=False,
            name=f"rep_{rep}",
            verbose=False,
        )
    return rep_data
