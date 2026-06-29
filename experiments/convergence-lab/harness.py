r"""Convergence-lab harness (#253): config-driven parallel model fitting.

Reads a grid YAML (a ``sweep:`` Cartesian product + optional ``fixed:``
overrides + optional ``replicates:``), fits every cell in parallel via
``multidms.model_collection.fit_models`` (``n_processes`` spawn workers),
and pickles the raw fit-collection DataFrame to
``results/<cache>/fit_collection.pkl``.

The output is a *true* ``fit_collection.pkl`` — the same
``stack_fit_models`` frame the scv2-spike pipeline writes — so the marimo
dashboard discovers it directly (it ``rglob``s ``fit_collection.pkl`` below
cwd) and all downstream analysis, including replicate-shift correlation, is
done on the fly from a ``ModelCollection`` built over the frame. The harness
only fits; it computes no derived metrics.

CPU parallelism uses ``multiprocessing`` ``spawn``, which re-imports this
module in every worker. That is safe here because the fitting is reachable
only under the ``if __name__ == "__main__":`` guard at the bottom — see the
warning on ``multidms.model_collection.fit_models``. ``--n-processes``
selects worker count (default: all but one core, capped at the grid size).

The runner owns the constant scv2-spike data; configs carry only swept
knobs and fixed overrides. See ``README.md`` in this directory.

Usage::

    pixi run python experiments/convergence-lab/harness.py \\
        --config grids/smoke.yaml --cache smoke
    pixi run python experiments/convergence-lab/harness.py \\
        --config grids/smoke.yaml --cache smoke --n-processes 4
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

import pickle  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

warnings.filterwarnings("ignore")

import multidms  # noqa: E402
from multidms.model_collection import fit_models  # noqa: E402
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


def build_params(exploded: list[dict], rep_data: dict) -> dict:
    """Turn exploded grid cells into a ``fit_models`` ``params`` dict.

    ``fit_models`` takes a ``params`` dict whose every value is a list and
    crosses them internally. Each exploded cell already carries an integer
    ``replicate`` (which selects a Data) plus its fit kwargs; this maps the
    replicate to the rep's Data object and collects each kwarg's distinct
    values into a list, so ``fit_models`` reproduces exactly the cells in
    ``exploded``.

    Args:
        exploded: Output of :func:`explode_grid` (sweep × replicates × fixed).
        rep_data: Output of :func:`load_rep_data` (``"rep_<n>" -> Data``).

    Returns:
        A ``params`` dict for :func:`multidms.model_collection.fit_models`:
        ``dataset`` is the list of Data objects, every other key the sorted
        distinct values seen across cells (singletons stay singleton lists).
    """
    reps = sorted({c["replicate"] for c in exploded})
    datasets = [rep_data[f"rep_{rep}"] for rep in reps]
    params: dict = {"dataset": datasets}
    for cell in exploded:
        for key, val in cell.items():
            if key == "replicate":
                continue
            params.setdefault(key, [])
            if val not in params[key]:
                params[key].append(val)
    return params


def run_fits(params: dict, n_processes: int) -> pd.DataFrame:
    """Fit the whole grid in parallel via ``fit_models`` and return the frame.

    Delegates to :func:`multidms.model_collection.fit_models`, which explodes
    ``params`` (``dataset`` × swept kwargs), fits each combination in a
    ``spawn`` worker pool (``n_processes`` workers), and stacks the results.
    Failures are tolerated (``failures="tolerate"``): a failed fit is dropped
    and the grid continues, so one bad cell does not poison the run; if every
    fit fails, ``fit_models`` raises ``ModelCollectionFitError``.

    This MUST be reached only under the module's ``if __name__ ==
    "__main__":`` guard — ``spawn`` re-imports this module in every worker.

    Args:
        params: Output of :func:`build_params`.
        n_processes: Worker count passed to ``fit_models`` (>= 1; 1 runs
            in-process with no pool).

    Returns:
        The raw ``stack_fit_models`` fit-collection DataFrame (one row per
        fit, the ``model`` column plus the fit kwargs and ``dataset_name``).
        This is a *true* ``fit_collection.pkl`` — no derived metrics.
    """
    from math import prod

    n = len(params["dataset"]) * prod(
        len(v) for k, v in params.items() if k != "dataset"
    )
    print(f"[harness] fitting {n} models with n_processes={n_processes} …", flush=True)
    t0 = time.time()
    n_fit, n_failed, fit_df = fit_models(
        params, n_processes=n_processes, failures="tolerate"
    )
    print(
        f"[harness] wall {time.time() - t0:.1f}s — {n_fit} fit, {n_failed} failed",
        flush=True,
    )
    return fit_df


def default_n_processes(grid_size: int) -> int:
    """Worker count to use when ``--n-processes`` is not given.

    All but one CPU core (leave one for the OS / parent), never more than the
    number of fits (extra workers would idle), and at least 1.

    Args:
        grid_size: Number of fits in the exploded grid.

    Returns:
        The default worker count.
    """
    cores = os.cpu_count() or 1
    return max(1, min(grid_size, cores - 1))


def run(config_path, cache: str, n_processes: int | None = None) -> Path:
    """Load config, fit the grid in parallel, and pickle the fit collection.

    Writes ``results/<cache>/fit_collection.pkl`` — the raw fit-collection
    DataFrame, the same schema the scv2-spike pipeline writes. The marimo
    dashboard discovers it by name; build a ``ModelCollection`` over it for
    any downstream analysis (correlation, basin diagnostics, plots).

    Args:
        config_path: Path to the grid YAML.
        cache: Subdirectory name under ``results/`` to hold the pickle.
        n_processes: Worker count; ``None`` → :func:`default_n_processes`.

    Returns:
        Path to the written ``fit_collection.pkl``.
    """
    config = load_config(config_path)
    rep_data = load_rep_data()
    exploded = explode_grid(config)
    if n_processes is None:
        n_processes = default_n_processes(len(exploded))
    params = build_params(exploded, rep_data)
    fit_df = run_fits(params, n_processes)

    out_dir = RESULTS_DIR / cache
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fit_collection.pkl"
    with open(out_path, "wb") as fh:
        pickle.dump(fit_df, fh)
    print(f"[harness] wrote {out_path}  ({len(fit_df)} fits)", flush=True)
    return out_path


def main() -> None:
    """CLI: ``--config <grid.yaml> --cache <name> [--n-processes N]``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to grid YAML")
    ap.add_argument(
        "--cache",
        required=True,
        help="output subdir → results/<cache>/fit_collection.pkl",
    )
    ap.add_argument(
        "--n-processes",
        type=int,
        default=None,
        help="parallel worker count (default: all but one core, capped at "
        "the grid size)",
    )
    args = ap.parse_args()
    # Resolve a relative --config against the harness dir (so `grids/smoke.yaml`
    # works regardless of cwd), matching how DATA_CSV is resolved.
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute() and not cfg_path.exists():
        cfg_path = HARNESS_DIR / args.config
    run(cfg_path, args.cache, args.n_processes)


if __name__ == "__main__":
    main()
