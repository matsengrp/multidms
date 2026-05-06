"""
================
model_collection
================

Contains the :class:`ModelCollection` class, which takes a collection of models
and merges the results for comparison and visualization.

Two fitting strategies are provided:

- :func:`fit_models` fits each ``(dataset, hyperparameter)`` combination
  independently in parallel (across CPU processes or GPU devices).
- :func:`fit_models_path` fits the same combinations sequentially along an
  ascending ``fusionreg`` axis, warm-starting each step from the previous
  fit's ``(β, β0, α)``. Use this when a strong shift lasso distorts the
  global-epistasis calibration for data-poor conditions under independent
  fitting — starting at the unregularized solution keeps each step in the
  basin of the previous one as the lasso is tightened.

Both return the same DataFrame schema, so either output can be passed to
:class:`ModelCollection` unchanged.
"""

import itertools as it
import time
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from multiprocessing import get_context
import multiprocessing
import pprint
import logging

import multidms
from multidms.utils import explode_params_dict, my_concat

import pandas as pd
import jax

import numpy as onp

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)

logging.getLogger("jax._src.xla_bridge").addFilter(
    logging.Filter(
        "An NVIDIA GPU may be present on this machine, "
        "but a CUDA-enabled jaxlib is not installed. Falling back to cpu."
    )
)


class ModelCollectionFitError(Exception):
    """Error fitting models."""

    pass


def fit_one_model(
    dataset,
    ge_type="Sigmoid",
    l2reg=0.0,
    fusionreg=0.0,
    beta0_ridge=0.0,
    scale_fusion_by_n=False,
    loss_type="functional_score_loss",
    maxiter=10,
    tol=1e-6,
    warmstart=True,
    beta0_init=None,
    beta_init=None,
    alpha_init=None,
    share_alpha=True,
    beta_clip_range=None,
    ge_kwargs=None,
    cal_kwargs=None,
    loss_kwargs=None,
    verbose=False,
    **kwargs,
):
    """
    Fit a single multidms model to a dataset.

    This is a wrapper around Model construction and fitting that saves
    all hyperparameters for bookkeeping. Used by :func:`fit_models` for
    parallel fitting across parameter sweeps.

    Parameters
    ----------
    dataset : :class:`multidms.Data`
        The dataset to fit to. ``dataset.name`` is saved for bookkeeping.
    ge_type : str
        Global epistasis type: ``'Identity'`` or ``'Sigmoid'``.
    l2reg : float
        L2 regularization strength for mutation effects.
    fusionreg : float
        Fusion (shift lasso) regularization strength.
    beta0_ridge : float
        Ridge penalty for beta0 differences from reference condition.
    scale_fusion_by_n : bool
        Weight each condition's fusion penalty by n_ref / n_d.
    loss_type : str
        Loss function: ``'functional_score_loss'`` or ``'count_loss'``.
    maxiter : int
        Maximum block coordinate descent iterations.
    tol : float
        Convergence tolerance.
    warmstart : bool
        Whether to use Ridge regression for initialization.
    beta0_init, beta_init : dict, optional
        Initial parameter values per condition.
    alpha_init : float or dict, optional
        Initial α scaling value (float for all, dict for per-condition).
    share_alpha : bool
        If True (default), single shared α; if False, per-condition α.
    beta_clip_range : tuple, optional
        ``(min, max)`` clipping for beta parameters.
    ge_kwargs, cal_kwargs, loss_kwargs : dict, optional
        Kwargs for sub-optimizers and loss function.
    verbose : bool
        Print progress during fitting.
    **kwargs : dict
        Additional keyword arguments saved for bookkeeping.

    Returns
    -------
    fit_series : :class:`pandas.Series`
        A series containing reference to the fit :class:`multidms.Model` object
        and the associated parameters used for the fit, including
        ``'dataset_name'`` and ``'fit_time'``.
    """
    fit_attributes = locals().copy()
    del fit_attributes["kwargs"]
    for key, value in kwargs.items():
        fit_attributes[key] = value

    model = multidms.Model(
        dataset,
        ge_type=ge_type,
        l2reg=l2reg,
        fusionreg=fusionreg,
        beta0_ridge=beta0_ridge,
        scale_fusion_by_n=scale_fusion_by_n,
        loss_type=loss_type,
    )

    del fit_attributes["dataset"]
    del fit_attributes["verbose"]
    fit_attributes["dataset_name"] = dataset.name
    fit_attributes["model"] = model

    if verbose:
        print("running:")
        pprint.pprint(fit_attributes)

    start = time.time()
    model.fit(
        warmstart=warmstart,
        maxiter=maxiter,
        tol=tol,
        beta0_init=beta0_init,
        beta_init=beta_init,
        alpha_init=alpha_init,
        share_alpha=share_alpha,
        beta_clip_range=beta_clip_range,
        ge_kwargs=ge_kwargs,
        cal_kwargs=cal_kwargs,
        loss_kwargs=loss_kwargs,
        verbose=verbose,
    )
    fit_attributes["fit_time"] = round(time.time() - start)

    return pd.Series(fit_attributes)


def _fit_fun(params):
    """Workaround for multiprocessing to fit models with sets of kwargs"""
    _, kwargs = params
    try:
        return fit_one_model(**kwargs)
    except Exception:
        return None


def stack_fit_models(fit_models_list):
    """
    Given a list of pd.Series objects returned by fit_one_model,
    stack them into a single pd.DataFrame
    """
    return pd.concat([f.to_frame().T for f in fit_models_list], ignore_index=True)


def _fit_models_gpu(exploded_params, gpu_ids):
    """Round-robin model fitting across GPUs using jax.default_device.

    Uses ThreadPoolExecutor with one thread per GPU. Each thread
    wraps its model fits in a ``jax.default_device`` context manager,
    ensuring all computation targets the assigned GPU. A semaphore
    per GPU ensures only one model runs per GPU at any time.

    Parameters
    ----------
    exploded_params : list of dict
        Each dict is a set of kwargs for :func:`fit_one_model`.
    gpu_ids : list of int
        GPU device IDs to distribute work across.

    Returns
    -------
    list
        List of :class:`pandas.Series` (or None for failures), one per model.
    """
    import os

    available_gpus = {d.id: d for d in jax.devices("gpu")}
    for gid in gpu_ids:
        if gid not in available_gpus:
            cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            hint = ""
            if cuda_env is not None:
                hint = (
                    f" Note: CUDA_VISIBLE_DEVICES={cuda_env!r} is set "
                    f"in your environment, which restricts which GPUs "
                    f"JAX can see. To use multiple GPUs, set "
                    f"CUDA_VISIBLE_DEVICES to a comma-separated list "
                    f"of physical GPU IDs (e.g., '0,1,2,3') before "
                    f"starting Python/Jupyter."
                )
            raise ValueError(
                f"GPU {gid} not found. "
                f"JAX can see {len(available_gpus)} GPU(s): "
                f"{list(available_gpus.keys())}.{hint}"
            )

    devices = [available_gpus[gid] for gid in gpu_ids]
    n_gpus = len(devices)
    gpu_semaphores = [threading.Semaphore(1) for _ in range(n_gpus)]

    logger.info(
        f"Distributing {len(exploded_params)} models "
        f"across {n_gpus} GPUs: {gpu_ids}"
    )

    def fit_on_gpu(task):
        idx, gpu_idx, kwargs = task
        device = devices[gpu_idx]
        semaphore = gpu_semaphores[gpu_idx]

        with semaphore:
            logger.info(f"Model {idx} starting on GPU {gpu_ids[gpu_idx]}")
            with jax.default_device(device):
                try:
                    result = fit_one_model(**kwargs)
                except Exception:
                    result = None
            logger.info(f"Model {idx} finished on GPU {gpu_ids[gpu_idx]}")
        return idx, result

    tasks = [(i, i % n_gpus, kw) for i, kw in enumerate(exploded_params)]

    results = [None] * len(exploded_params)
    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = {executor.submit(fit_on_gpu, task): task for task in tasks}
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx] = result

    return results


def fit_models(
    params,
    gpu_ids=None,
    n_processes=1,
    n_threads=None,
    failures="error",
):
    """Fit collection of :class:`multidms.model.Model` models.

    Enables fitting of multiple models simultaneously. Most commonly,
    this function is used to fit a set of models across combinations
    of replicate training datasets and regularization coefficients for
    model selection and evaluation. The returned dataframe is meant to
    be passed into the :class:`ModelCollection` class for comparison
    and visualization.

    There are two parallelism modes, controlled by mutually exclusive
    parameters:

    - **GPU mode** (``gpu_ids``): Round-robin models across the
      specified GPUs using ``jax.default_device`` and a
      ``ThreadPoolExecutor``, one model per GPU at a time.
    - **CPU mode** (``n_processes``): Spawn independent processes
      via ``multiprocessing.Pool`` with the ``spawn`` context.

    Parameters
    ----------
    params : dict
        Dictionary which defines the parameter space of all models you
        wish to run. Each value in the dictionary must be a list of
        values, even in the case of singletons.
        This function will compute all combinations of the parameter
        space and pass each combination to :func:`fit_one_model`
        to be run in parallel, thus only key-value pairs which
        match the kwargs are allowed.
        See the docstring of :func:`fit_one_model` for
        a description of the allowed parameters.
    gpu_ids : list of int, optional
        GPU device IDs to use for fitting. Models are round-robin
        assigned across GPUs, one model per GPU at a time. Uses
        ``jax.default_device`` to pin each fit to a specific GPU.
        Mutually exclusive with ``n_processes``.

        .. note::
            The IDs correspond to JAX device IDs from
            ``jax.devices("gpu")``, which are determined by the
            ``CUDA_VISIBLE_DEVICES`` environment variable at the
            time JAX is first imported. To use multiple GPUs, ensure
            ``CUDA_VISIBLE_DEVICES`` includes all desired GPU IDs
            (e.g., ``export CUDA_VISIBLE_DEVICES=0,1,2,3``) before
            starting Python or Jupyter.
    n_processes : int
        Number of parallel CPU processes for fitting. Default is 1
        (sequential, no multiprocessing overhead). Uses
        ``multiprocessing.Pool`` with the ``spawn`` context when > 1.
        Mutually exclusive with ``gpu_ids``.
    n_threads : int, optional
        .. deprecated::
            Use ``gpu_ids`` for GPU fitting or ``n_processes`` for
            CPU fitting.
    failures : {"error", "tolerate"}
        What if fitting fails for a model? If ``"error"`` then raise
        an error, if ``"tolerate"`` then just return ``None`` for
        models that failed optimization.

    Returns
    -------
    (n_fit, n_failed, fit_models)
        Number of models that fit successfully, number of models that
        failed, and a dataframe which contains a row for each of the
        ``multidms.Model`` object references along with the parameters
        each was fit with for convenience. The dataframe is ultimately
        meant to be passed into the :class:`ModelCollection` class for
        comparison and visualization.
    """
    # Handle deprecated n_threads parameter
    if n_threads is not None:
        warnings.warn(
            "n_threads is deprecated. Use gpu_ids for GPU fitting or "
            "n_processes for CPU fitting. n_threads will be removed in "
            "a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        if gpu_ids is None and n_processes == 1:
            n_processes = n_threads if n_threads != -1 else multiprocessing.cpu_count()

    if gpu_ids is not None and n_processes != 1:
        raise ValueError(
            "Cannot specify both gpu_ids and n_processes. "
            "Use gpu_ids for GPU fitting or n_processes for CPU fitting."
        )

    if gpu_ids is not None and len(gpu_ids) == 0:
        raise ValueError("gpu_ids must be a non-empty list of GPU device IDs.")

    if n_processes < 1:
        raise ValueError("n_processes must be >= 1.")

    exploded_params = explode_params_dict(params)

    if gpu_ids is not None:
        fit_results = _fit_models_gpu(exploded_params, gpu_ids)
    elif n_processes == 1:
        fit_results = [_fit_fun((None, kw)) for kw in exploded_params]
    else:
        # see https://pythonspeed.com/articles/python-multiprocessing/
        # for why we use spawn context (JAX is multithreaded internally)
        with get_context("spawn").Pool(n_processes) as p:
            fit_results = p.map(
                _fit_fun,
                [(None, kw) for kw in exploded_params],
            )

    assert len(fit_results) == len(exploded_params)

    # Check to see if any models failed optimization
    n_failed = sum(model is None for model in fit_results)
    if failures == "error":
        if n_failed:
            raise ModelCollectionFitError(
                f"Failed fitting {n_failed} of {len(exploded_params)} " "parameter sets"
            )
    elif failures != "tolerate":
        raise ValueError(f"invalid {failures=}")
    n_fit = len(fit_results) - n_failed
    if n_fit == 0:
        raise ModelCollectionFitError(
            f"Failed fitting all {len(exploded_params)} parameter sets"
        )

    return (n_fit, n_failed, stack_fit_models(fit_results))


def _extract_seed(prev_model):
    """Extract (β, β0, α) seed dicts from a fitted multidms.Model.

    Returns a dict suitable for passing as ``beta_init``, ``beta0_init``,
    and ``alpha_init`` kwargs to :func:`fit_one_model`. ``alpha_init`` is
    a scalar jax array when the previous fit used ``share_alpha=True``
    and a ``dict[str, jax_scalar]`` when it used ``share_alpha=False``.
    """
    conditions = prev_model.data.conditions
    p = prev_model.params
    beta_init = {d: p.φ[d].β for d in conditions}
    beta0_init = {d: p.φ[d].β0 for d in conditions}
    if isinstance(p.α, dict):
        alpha_init = {d: p.α[d] for d in conditions}
    else:
        alpha_init = p.α
    return beta_init, beta0_init, alpha_init


def _assert_no_nan(model):
    """Raise :class:`ModelCollectionFitError` if any fitted parameter is NaN.

    Checks ``β``, ``β0``, and ``α`` across all conditions. The guard runs
    between steps of a continuation path so that a NaN fit cannot silently
    seed the next step.
    """
    import jax.numpy as jnp

    p = model.params
    for d in model.data.conditions:
        if bool(jnp.isnan(p.φ[d].β).any()):
            raise ModelCollectionFitError(f"NaN in β for condition {d}")
        if bool(jnp.isnan(jnp.asarray(p.φ[d].β0)).any()):
            raise ModelCollectionFitError(f"NaN in β0 for condition {d}")
    α_vals = p.α.values() if isinstance(p.α, dict) else [p.α]
    for a in α_vals:
        if bool(jnp.isnan(jnp.asarray(a)).any()):
            raise ModelCollectionFitError("NaN in α")


def _fit_one_path_step(prev_model, **kwargs):
    """Fit one step of a continuation path, seeded from ``prev_model``.

    Overrides any ``warmstart`` / ``beta_init`` / ``beta0_init`` /
    ``alpha_init`` values in ``kwargs`` with the fitted parameters of
    ``prev_model``. The Ridge warmstart is always disabled on path steps
    — the previous fit's parameters are the warm-start.

    The ``beta_init`` / ``beta0_init`` / ``alpha_init`` cells of the
    returned row are cleared to ``None`` after fitting. The seeds are
    JAX arrays (or dicts of them), so leaving them in the row would
    break pandas operations downstream (``.apply(str)``,
    ``groupby``) and make the path DataFrame schema-incompatible with
    :func:`fit_models` output. The true seed is always recoverable
    from the previous step's ``model`` column.
    """
    beta_init, beta0_init, alpha_init = _extract_seed(prev_model)
    kwargs = {
        **kwargs,
        "warmstart": False,
        "beta_init": beta_init,
        "beta0_init": beta0_init,
        "alpha_init": alpha_init,
    }
    row = fit_one_model(**kwargs)
    row["beta_init"] = None
    row["beta0_init"] = None
    row["alpha_init"] = None
    return row


def fit_models_path(params, verbose=False):
    """Fit a continuation path of models along ascending ``fusionreg``.

    For every combination of the non-``fusionreg`` hyperparameters, fit a
    sequence of models in order of increasing ``fusionreg``, warm-starting
    each step from the previous step's fitted ``(β, β0, α)``. The first
    step of each path is a normal :func:`fit_one_model` call and honours
    ``warmstart`` as passed; subsequent steps set ``warmstart=False`` and
    seed explicitly from the previous fit.

    Parameters
    ----------
    params : dict
        Same shape as the ``params`` dict accepted by :func:`fit_models`.
        The value at key ``"fusionreg"`` is treated as the path axis and
        is sorted ascending internally; all other list-valued keys are
        Cartesian-producted over as in :func:`fit_models`.
    verbose : bool
        If True, print the hyperparameters of each step before fitting.

    Returns
    -------
    (n_fit, n_failed, fit_collection_df)
        A tuple matching :func:`fit_models`. ``fit_collection_df`` has one
        row per successfully fit step and the same column schema as
        :func:`fit_models`.

    Notes
    -----
    - Failure handling is step-local. If step *k* of a combo fails, steps
      0..*k*−1 are kept and steps *k*..end of that combo are skipped;
      other combos continue independently. A step is considered a failure
      if :func:`fit_one_model` raises, or if the fitted parameters contain
      any NaN (caught by :func:`_assert_no_nan` so that NaNs cannot seed a
      downstream step).
    - If the smallest ``fusionreg`` in the path is non-zero, a warning is
      emitted — the physical motivation for continuation is to start from
      the unregularized problem.
    """
    if "fusionreg" not in params:
        raise ValueError(
            "fit_models_path requires a 'fusionreg' key in params; it is "
            "the path axis."
        )
    fusionreg_path = sorted(params["fusionreg"])
    if len(fusionreg_path) == 0:
        raise ValueError("fit_models_path requires at least one fusionreg value.")
    if fusionreg_path[0] != 0.0:
        warnings.warn(
            f"fit_models_path starting at fusionreg={fusionreg_path[0]} "
            f"(!= 0.0). The continuation rationale assumes the path starts "
            f"from the unregularized problem; a non-zero first step is "
            f"almost never intended."
        )

    non_path_params = {k: v for k, v in params.items() if k != "fusionreg"}

    rows = []
    n_failed = 0
    for combo in explode_params_dict(non_path_params):
        prev_model = None
        for fr in fusionreg_path:
            step_kwargs = {**combo, "fusionreg": fr}
            step_kwargs.setdefault("verbose", verbose)
            try:
                if prev_model is None:
                    row = fit_one_model(**step_kwargs)
                else:
                    row = _fit_one_path_step(prev_model, **step_kwargs)
                _assert_no_nan(row["model"])
            except Exception as e:
                n_failed += 1
                if verbose:
                    print(f"path step fusionreg={fr} failed: {e!r}")
                prev_model = None
                # Remaining steps in this path are also counted as failures,
                # since the path cannot continue without a valid seed.
                remaining = [f for f in fusionreg_path if f > fr]
                n_failed += len(remaining)
                break
            rows.append(row)
            prev_model = row["model"]

    if len(rows) == 0:
        raise ModelCollectionFitError(
            f"Failed fitting all path steps across {n_failed} attempts"
        )

    return len(rows), n_failed, stack_fit_models(rows)


def concat_path_trajectories(fit_collection_df, groupby_cols=None):
    """Concatenate per-step convergence trajectories within each path.

    Each row of ``fit_collection_df`` (the output of :func:`fit_models_path`)
    carries a fitted :class:`multidms.Model` whose
    ``convergence_trajectory_df`` describes only its own step. This helper
    stitches the per-step trajectories within each path into a single long
    DataFrame, tagged with ``fusionreg``, ``step_index``, and a running
    global iteration counter, so a whole path can be plotted on one axis.

    Parameters
    ----------
    fit_collection_df : pandas.DataFrame
        Output of :func:`fit_models_path`. Must contain a ``model`` column
        and a ``fusionreg`` column.
    groupby_cols : list of str, optional
        Columns that identify a path. When ``None`` (default), every
        column is used except ``fusionreg``, ``model``, ``fit_time``,
        ``beta0_init``, ``beta_init``, and ``alpha_init`` — the latter
        three change between path steps by construction.

    Returns
    -------
    pandas.DataFrame
        Long-form trajectory with ``path_id``, ``step_index``,
        ``fusionreg``, ``iteration_within_step``, and
        ``iteration_global`` columns, followed by the columns carried
        from each step's ``convergence_trajectory_df``.
    """
    excluded = {
        "fusionreg",
        "model",
        "fit_time",
        "beta0_init",
        "beta_init",
        "alpha_init",
    }
    if groupby_cols is None:
        groupby_cols = [c for c in fit_collection_df.columns if c not in excluded]

    rows = []
    if len(groupby_cols) == 0:
        groups = [(None, fit_collection_df)]
    else:
        groups = list(fit_collection_df.groupby(groupby_cols, dropna=False))

    for path_id, (_, path_df) in enumerate(groups):
        path_df = path_df.sort_values("fusionreg").reset_index(drop=True)
        iteration_global_offset = 0
        for step_index, row in path_df.iterrows():
            traj = row["model"].convergence_trajectory_df
            if traj is None or len(traj) == 0:
                continue
            traj = traj.copy()
            traj = traj.rename(columns={"iteration": "iteration_within_step"})
            traj["step_index"] = step_index
            traj["fusionreg"] = row["fusionreg"]
            traj["path_id"] = path_id
            traj["iteration_global"] = (
                iteration_global_offset + traj["iteration_within_step"]
            )
            iteration_global_offset = int(traj["iteration_global"].iloc[-1]) + 1
            rows.append(traj)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    front = [
        "path_id",
        "step_index",
        "fusionreg",
        "iteration_within_step",
        "iteration_global",
    ]
    ordered = front + [c for c in out.columns if c not in front]
    return out[ordered]


class ModelCollection:
    """
    A class for the comparison and visualization of multiple
    `multidms.Model` fits. The respective collection of
    training datasets for each fit must
    share the same reference sequence and conditions. Additionally,
    the inferred site maps must agree upon condition wildtypes
    for all shared sites.

    The utility function `multidms.model_collection.fit_models` is used to fit
    the collection of models, and the resulting dataframe is passed to the
    constructor of this class.

    Parameters
    ----------
    fit_models : :class:`pandas.DataFrame`
        A dataframe containing the fit attributes and pickled model objects
        as returned by `multidms.model_collection.fit_models`.
    """

    def __init__(self, fit_models):
        """See class docstring."""
        # Check that all datasets share reference, and conditions, and site maps
        first_dataset = fit_models.iloc[0].model.data
        validated_datasets = [first_dataset.name]
        site_map_union = first_dataset.site_map.copy()
        shared_mutations = set(first_dataset.mutations)
        all_mutations = set(first_dataset.mutations)
        for fit in fit_models.model:
            if fit.data.name in validated_datasets:
                continue
            if fit.data.reference != first_dataset.reference:
                raise ValueError(
                    "All model training datasets must share the same reference sequence"
                )
            if not len(set(fit.data.conditions) - set(first_dataset.conditions)) == 0:
                raise ValueError(
                    "All model training datasets must share the same conditions"
                )
            shared_sites = list(
                set.intersection(
                    set(site_map_union.index), set(fit.data.site_map.index)
                )
            )

            if not site_map_union.loc[shared_sites].equals(
                fit.data.site_map.loc[shared_sites]
            ):
                raise ValueError(
                    "All model training datasets must share the same site map"
                )
            new_sites = list(set(fit.data.site_map.index) - set(site_map_union.index))
            if len(new_sites) > 0:
                site_map_union = pd.concat(
                    [site_map_union, fit.data.site_map.loc[new_sites]]
                ).sort_index()
            validated_datasets.append(fit.data.name)

            shared_mutations = set.intersection(
                shared_mutations, set(fit.data.mutations)
            )
            all_mutations = set.union(all_mutations, set(fit.data.mutations))

        # Add convergence flag.
        fit_models["converged"] = fit_models.model.apply(lambda x: x.converged).astype(
            bool
        )
        for idx, fit in fit_models.iterrows():
            for condition, loss in fit.model.training_loss.items():
                fit_models.loc[idx, f"{condition}_loss_training"] = loss

        self._site_map_union = site_map_union
        self._conditions = first_dataset.conditions
        self._reference = first_dataset.reference
        self.fit_models = fit_models
        self.condition_colors = first_dataset.condition_colors
        self._shared_mutations = tuple(shared_mutations)
        self._all_mutations = tuple(all_mutations)

    @property
    def site_map_union(self) -> pd.DataFrame:
        """The union of all site maps of all datasets used for fitting."""
        return self._site_map_union

    @property
    def conditions(self) -> list:
        """The conditions (shared by each fitting dataset) used for fitting."""
        return self._conditions

    @property
    def reference(self) -> str:
        """The reference conditions (shared by each fitting dataset) used for fitting."""
        return self._reference

    @property
    def shared_mutations(self) -> tuple:
        """The mutations shared by each fitting dataset."""
        return self._shared_mutations

    @property
    def all_mutations(self) -> tuple:
        """The mutations shared by each fitting dataset."""
        return self._all_mutations

    @lru_cache(maxsize=10)
    def split_apply_combine_muts(
        self,
        groupby=("dataset_name", "fusionreg"),
        aggregate_func="mean",
        inner_merge_dataset_muts=True,
        query=None,
        **kwargs,
    ):
        """
        Wrapper to split-apply-combine the set of mutational dataframes
        harbored by each of the fits in the collection.

        Here, we group the collection of fits using attributes
        (columns in :py:attr:`ModelCollection.fit_models`) specified using the
        ``groupby`` parameter.
        Each of the individual fits within a groups may then be filtered
        via ``**kwargs``, and aggregated via ``aggregate_func``, before
        the function stacks all the groups back together in a
        tall style dataframe. The resulting dataframe will have a multiindex
        with the mutation and the groupby attributes.

        Parameters
        ----------
        groupby : str or tuple of str or None, optional
            The attributes to group the fits by. If None, then group by all
            attributes except for the model, data, and step_loss attributes.
            The default is ("dataset_name", "fusionreg").
        aggregate_func : str or callable, optional
            The function to aggregate the mutational dataframes within each group.
            The default is "mean".
        inner_merge_dataset_muts : bool, optional
            Whether to toss mutations which are _not_ shared across all datasets
            before aggregation of group mutation parameter values.
            The default is True.
        query : str, optional
            The pandas query to apply to the `ModelCollection.fit_models`
            dataframe before splitting. The default is None.
        **kwargs : dict
            Keyword arguments to pass to the :func:`multidms.Model.get_mutations_df`
            method (``phenotype_as_effect`` and ``times_seen_threshold``). See the
            method docstring for details.

        Returns
        -------
        :class:`pandas.DataFrame`
            A dataframe containing the aggregated mutational parameter values
        """
        print("cache miss - this could take a moment")
        queried_fits = (
            self.fit_models.query(query) if query is not None else self.fit_models
        )
        if len(queried_fits) == 0:
            raise ValueError("invalid query, no fits returned")

        if groupby is None:
            ret = (
                pd.concat(
                    [
                        fit["model"].get_mutations_df(**kwargs)
                        for _, fit in queried_fits.iterrows()
                    ],
                    join="inner",  # the columns will always match based on class req.
                )
                .query(
                    f"mutation.isin({list(self.shared_mutations)})"
                    if inner_merge_dataset_muts
                    else "mutation.notna()"
                )
                .select_dtypes(include="number")
                .groupby("mutation")
                .aggregate(aggregate_func)
            )
            return ret

        elif isinstance(groupby, str):
            groupby = tuple([groupby])

        elif isinstance(groupby, tuple):
            if not all(feature in queried_fits.columns for feature in groupby):
                raise ValueError(
                    f"invalid groupby, values must be in {self.fit_models.columns}"
                )
        else:
            raise ValueError(
                "invalid groupby, must be tuple with values "
                f"in {queried_fits.columns}"
            )

        ret = pd.concat(
            [
                pd.concat(
                    [
                        fit["model"].get_mutations_df(**kwargs)
                        for _, fit in fit_group.iterrows()
                    ],
                    join="inner",  # the columns will always match based on class req.
                )
                .query(
                    f"mutation.isin({list(self.shared_mutations)})"
                    if inner_merge_dataset_muts
                    else "mutation.notna()"
                )
                .select_dtypes(include="number")
                .groupby("mutation")
                .aggregate(aggregate_func)
                .assign(**dict(zip(list(groupby), group)))
                .reset_index()
                .set_index(list(groupby))
                for group, fit_group in queried_fits.groupby(
                    list(groupby), observed=True
                )
            ],
            join="inner",
        )

        return ret

    def add_eval_loss(self, test_data, overwrite=False):
        """
        Add evaluation (validation) loss to the fit collection dataframe.

        Parameters
        ----------
        test_data : pd.DataFrame or dict(str, pd.DataFrame)
            The testing dataframe to compute validation loss with respect to,
            must have columns "aa_substitutions", "condition", and "func_score".
            If a dictionary is passed, there should be a key for
            each unique dataset_name factor in the self.fit_models dataframe
            - with the value being the respective testing dataframe.
        overwrite : bool, optional
            Whether to overwrite the validation_loss column if it already exists.
            The default is False.

        Returns
        -------
        None
        """
        if isinstance(test_data, pd.DataFrame):
            temp_test_data = test_data.copy()
            test_data = {}
            for name in self.fit_models["dataset_name"].unique():
                test_data[name] = temp_test_data

        # check there's a testing dataframe for each unique dataset_name
        assert set(test_data.keys()) == set(self.fit_models["dataset_name"].unique())

        all_loss_keys = list(self.conditions) + ["total"]
        validation_cols_exist = onp.any(
            [
                f"{key}_loss_validation" in self.fit_models.columns
                for key in all_loss_keys
            ]
        )
        if validation_cols_exist and not overwrite:
            raise ValueError(
                "validation_loss already exists in self.fit_models, set overwrite=True "
                "to overwrite"
            )

        self.fit_models = self.fit_models.assign(
            **{f"{key}_loss_validation": onp.nan for key in all_loss_keys},
        )

        for idx, fit in self.fit_models.iterrows():
            eval_loss = fit.model.eval_loss(test_data[fit["dataset_name"]])
            for key, loss in eval_loss.items():
                self.fit_models.loc[idx, f"{key}_loss_validation"] = loss

        return None

    def loss_df(self, query=None):
        """
        Return a long form dataframe with columns
        "dataset_name", "fusionreg",
        "split" ("training" or "validation"),
        "loss" (actual value), and "condition".

        The ``condition`` column includes ``"total"`` for the summed loss.

        Parameters
        ----------
        query : str, optional
            The query to apply to the fit_models dataframe
            before formatting the loss dataframe. The default is None.
        """
        if query is not None:
            queried_fits = self.fit_models.query(query)
        else:
            queried_fits = self.fit_models
        if len(queried_fits) == 0:
            raise ValueError("invalid query, no fits returned")

        id_vars = ["dataset_name", "fusionreg"]
        value_vars = [
            c
            for c in queried_fits.columns
            if c.endswith("_loss_training") or c.endswith("_loss_validation")
        ]
        loss_df = queried_fits.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="condition",
            value_name="loss",
        ).assign(
            split=lambda x: x.condition.str.split("_").str.get(-1),
            condition=lambda x: x.condition.str.split("_").str[:-2].str.join("_"),
        )
        return loss_df

    def convergence_trajectory_df(
        self,
        query=None,
        id_vars=("dataset_name", "fusionreg"),
    ):
        """
        Combine the converence trajectory dataframes of
        all fits in the queried collection.
        """
        queried_fits = (
            self.fit_models.query(query) if query is not None else self.fit_models
        )
        if len(queried_fits) == 0:
            raise ValueError("invalid query, no fits returned")

        if not all([var in queried_fits.columns for var in id_vars]):
            raise ValueError(f"invalid {id_vars=}")

        convergence_trajectory_data = my_concat(
            [
                (
                    fit.model.convergence_trajectory_df.assign(
                        **{key: fit[key] for key in id_vars}
                    )
                )
                for _, fit in queried_fits.iterrows()
            ]
        )

        return convergence_trajectory_data

    def plot_convergence_trajectory(
        self,
        query=None,
        id_vars=("dataset_name", "fusionreg"),
        **kwargs,
    ):
        """Plot convergence trajectories as an interactive Altair chart.

        Delegates to :func:`multidms.plot.convergence_trajectory` after
        extracting the convergence data from fitted models.

        Parameters
        ----------
        query : str, optional
            Query to filter ``fit_models`` before extracting trajectories.
        id_vars : tuple of str
            Columns identifying individual model runs.
        **kwargs
            Passed to :func:`multidms.plot.convergence_trajectory`.

        Returns
        -------
        alt.Chart
            Interactive Altair chart with group dropdown and legend toggle.
        """
        df = self.convergence_trajectory_df(query=query, id_vars=id_vars)
        return multidms.plot.convergence_trajectory(df, id_cols=list(id_vars), **kwargs)

    def mut_param_heatmap(
        self,
        query=None,
        mut_param="shift",
        aggregate_func="mean",
        inner_merge_dataset_muts=True,
        times_seen_threshold=0,
        phenotype_as_effect=True,
        **kwargs,
    ):
        """
        Create lineplot and heatmap altair chart
        across replicate datasets.
        This function optionally applies a given `pandas.query`
        on the fit_models dataframe that should result in a subset of
        fit's which make sense to aggregate mutational data across, e.g.
        replicate datasets.
        It then computes the mean or median mutational parameter value
        ("beta", "shift", or "predicted_func_score")
        between the remaining fits. and creates an interactive altair chart.


        Note that this will throw an error if the queried fits have more
        than one unique hyper-parameter besides "dataset_name".


        Parameters
        ----------
        query : str
            The query to apply to the fit_models dataframe. This should be
            used to subset the fits to only those which make sense to aggregate
            mutational data across, e.g. replicate datasets.
            For example, if you have a collection of
            fits with different epistatic models, you may want to query
            for only those fits with the same epistatic model. e.g.
            `query="ge_type == 'Sigmoid'"`. For more on the query
            syntax, see the
            `pandas.query <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html>`_
            documentation.
        mut_param : str, optional
            The mutational parameter to plot. The default is "shift".
            Must be one of "shift", "predicted_func_score", or "beta".
        aggregate_func : str, optional
            The function to aggregate the mutational parameter values
            between dataset fits. The default is "mean".
        inner_merge_dataset_muts : bool, optional
            Whether to toss mutations which are _not_ shared across all datasets
            before aggregation of group mutation parameter values.
            The default is True.
        times_seen_threshold : int, optional
            The minimum number of times a mutation must be seen across
            all conditions within a single fit to be included in the
            aggregation. The default is 0.
        phenotype_as_effect : bool, optional
            Passed to `Model.get_mutations_df()`,
            Only applies if `mut_param="predicted_func_score"`.
        **kwargs : dict
            Keyword arguments to pass to
            :func:`multidms.plot.lineplot_and_heatmap`.

        Returns
        -------
        altair.Chart
            A chart object which can be displayed in a jupyter notebook
            or saved to a file.
        """
        queried_fits = (
            self.fit_models.query(query) if query is not None else self.fit_models
        )
        if len(queried_fits) == 0:
            raise ValueError("invalid query, no fits returned")
        shouldbe_uniform = list(
            set(queried_fits.columns)
            - set(
                ["model", "dataset_name"]
                + [col for col in queried_fits.columns if "loss" in col]
            )
        )
        # print(shouldbe_uniform)
        groups_to_combine = queried_fits.groupby(shouldbe_uniform).ngroups
        if groups_to_combine > 1:
            warnings.warn(
                "the fits that will be aggregated appear to differ by features "
                "other than dataset_name, this may result in unexpected behavior"
            )
        if aggregate_func not in ["mean", "median"]:
            raise ValueError(f"invalid {aggregate_func=} must be mean or median")
        possible_mut_params = set(["shift", "predicted_func_score", "beta"])
        if mut_param not in possible_mut_params:
            raise ValueError(f"invalid {mut_param=}")

        # aggregate mutation values between dataset fits
        muts_df = (
            self.split_apply_combine_muts(
                groupby="dataset_name",
                aggregate_func=aggregate_func,
                inner_merge_dataset_muts=inner_merge_dataset_muts,
                times_seen_threshold=times_seen_threshold,
                phenotype_as_effect=phenotype_as_effect,
                query=query,
            )
            .groupby("mutation")
            .aggregate(aggregate_func)
        )

        # drop columns which are not the mutational parameter of interest
        drop_cols = [c for c in muts_df.columns if "times_seen" in c]
        for param in possible_mut_params - set([mut_param]):
            drop_cols.extend([c for c in muts_df.columns if c.startswith(param)])
        muts_df.drop(drop_cols, axis=1, inplace=True)

        # add in the mutation annotations
        parse_mut = self.fit_models.iloc[0].model.data.parse_mut
        muts_df["wildtype"], muts_df["site"], muts_df["mutant"] = zip(
            *muts_df.reset_index()["mutation"].map(parse_mut)
        )

        # no longer need mutation annotation
        muts_df.reset_index(drop=True, inplace=True)

        wt_dict = {
            "wildtype": self.site_map_union[self.reference].values,
            "mutant": self.site_map_union[self.reference].values,
            "site": self.site_map_union[self.reference].index.values,
        }

        for value_col in [c for c in muts_df.columns if c.startswith(mut_param)]:
            wt_dict[value_col] = 0

        # add reference wildtype values needed for lineplot and heatmap fx
        muts_df = pd.concat([muts_df, pd.DataFrame(wt_dict)])

        # add in wildtype values for each non-reference condition
        # these will be available in the tooltip
        addtl_tooltip_stats = []
        for condition in self.conditions:
            if condition == self.reference:
                continue
            addtl_tooltip_stats.append(f"wildtype_{condition}")
            muts_df[f"wildtype_{condition}"] = muts_df.site.apply(
                lambda site: self.site_map_union.loc[site, condition]
            )

        # melt conditions and stats cols
        # note that we must rename conditions with "." in the
        # name to "_" to avoid altair errors
        muts_df_tall = muts_df.melt(
            id_vars=["wildtype", "site", "mutant"] + addtl_tooltip_stats,
            value_vars=[c for c in muts_df.columns if c.startswith(mut_param)],
            var_name="condition",
            value_name=mut_param,
        ).replace(
            {
                f"{mut_param}_{condition}": condition.replace(".", "_")
                for condition in self.conditions
            },
        )

        # add in condition colors, rename for altair
        condition_colors = {
            con.replace(".", "_"): col for con, col in self.condition_colors.items()
        }

        # rename for altair
        addtl_tooltip_stats = [v.replace(".", "_") for v in addtl_tooltip_stats]
        muts_df_tall.rename(
            {c: c.replace(".", "_") for c in muts_df_tall.columns}, axis=1, inplace=True
        )

        return multidms.plot.mut_param_heatmap(
            muts_df_tall,
            mut_param=mut_param,
            addtl_tooltip_stats=addtl_tooltip_stats,
            category_colors=condition_colors,
            **kwargs,
        )

    def mut_param_traceplot(
        self,
        mutations,
        mut_param="shift",
        x="fusionreg",
        width_scalar=100,
        height_scalar=100,
        **kwargs,
    ):
        """
        Visualize mutation parameter values across the lasso penalty weights
        (by default) of a given subset of the mutations in the form of an
        `altair.FacetChart`. This is useful when you would like to confirm
        that a reported mutational parameter value carries through across the
        individual fits.


        Returns
        -------
        altair.Chart
            A chart object which can be displayed in a jupyter notebook
            or saved to a file.
        """
        if isinstance(mutations, str):
            mutations = [mutations]
        if len(mutations) == 0:
            raise ValueError("invalid mutations, must be non-empty list")
        if len(mutations) >= 500:
            raise ValueError("too many mutations, please subset to < 500")
        possible_mut_params = set(["shift", "predicted_func_score", "beta"])
        if mut_param not in possible_mut_params:
            raise ValueError(f"invalid {mut_param=}")

        # get mutation values, group by x axis variable and dataset
        muts_df = self.split_apply_combine_muts(
            groupby=("dataset_name", x), **kwargs
        ).reset_index()

        # drop columns which are not the mutational parameter of interest,
        # or mutational identifiers
        drop_cols = [c for c in muts_df.columns if "times_seen" in c]
        for param in possible_mut_params - set([mut_param]):
            drop_cols.extend([c for c in muts_df.columns if c.startswith(param)])
        muts_df.drop(drop_cols, axis=1, inplace=True)

        # subset to mutations of interest
        muts_df = muts_df.query("mutation.isin(@mutations)")

        # check that we have multiple regularization weights
        if len(muts_df[x].unique()) <= 1:
            raise ValueError(
                "invalid kwargs, must specify a subset of fits with "
                "multiple lasso penalty weights"
            )

        # add in mutation annotations for coloring
        def mut_type(mut):
            return "stop" if mut.endswith("*") else "nonsynonymous"

        muts_df = muts_df.assign(mut_type=muts_df.mutation.apply(mut_type))

        # melt conditions and stats cols
        id_cols = ["dataset_name", x, "mut_type", "mutation"]
        stat_cols_to_keep = [c for c in muts_df.columns if c.startswith(mut_param)]
        muts_df_tall = muts_df.melt(
            id_vars=id_cols,
            value_vars=stat_cols_to_keep,
            var_name="condition",
            value_name=mut_param,
        )
        muts_df_tall.condition = muts_df_tall.condition.str.replace(
            f"^{mut_param}_", "", regex=True
        )

        return multidms.plot.mut_param_traceplot(
            muts_df_tall,
            x=x,
            mut_param=mut_param,
            width_scalar=width_scalar,
            height_scalar=height_scalar,
        )

    def shift_sparsity(
        self,
        x="fusionreg",
        width_scalar=100,
        height_scalar=100,
        return_data=False,
        **kwargs,
    ):
        """
        Visualize shift parameter set sparsity across the lasso penalty weights
        (by default) in the form of an `altair.FacetChart`.
        We will group the mutations according to their status as either a
        a "stop" (e.g. A15*), or "nonsynonymous" (e.g. A15G) mutation before calculating
        the sparsity. This is because in a way, mutations to stop codons act as a
        False positive rate, as we expect their mutational effect to be equally
        deleterious in all experiments, and thus have a shift parameter value of zero.


        Returns
        -------
        altair.Chart or Tuple(pd.DataFrame, altair.Chart)
            A chart object which can be displayed in a jupyter notebook
            or saved to a file. If `return_data=True`, then a tuple
            containing the chart and the underlying data will be returned.
        """
        # get mutation values, group by x axis variable and dataset
        df = self.split_apply_combine_muts(groupby=("dataset_name", x), **kwargs)

        # no need to view parameters besides shifts
        to_throw = [
            col
            for col in df.columns
            if not col.startswith("shift") and col != "mutation"
        ]

        # feature columns for distinct sparsity measurements
        feature_cols = ["dataset_name", x, "mut_type"]

        def sparsity(x):
            return (x == 0).mean()

        def mut_type(mut):
            return "stop" if mut.endswith("*") else "nonsynonymous"

        # apply, drop, and melt
        sparsity_df = (
            df.drop(columns=to_throw)
            .assign(mut_type=lambda x: x.mutation.apply(mut_type))
            .reset_index()
            .groupby(by=feature_cols)
            .apply(sparsity, include_groups=False)
            .drop(columns=["mutation"])
            .reset_index(drop=False)
            .melt(id_vars=feature_cols, var_name="mut_param", value_name="sparsity")
        )

        chart = multidms.plot.shift_sparsity(
            sparsity_df,
            x=x,
            width_scalar=width_scalar,
            height_scalar=height_scalar,
        )

        if return_data:
            return chart, sparsity_df
        return chart

    def mut_param_dataset_correlation(
        self,
        x="fusionreg",
        width_scalar=400,
        height=400,
        return_data=False,
        r=1,
        **kwargs,
    ):
        """
        Visualize the correlation between replicate datasets across the lasso penalty
        weights (by default) in the form of an `altair.FacetChart`.
        We compute correlation of mutation parameters accross each pair of datasets
        in the collection.

        Parameters
        ----------
        x : str, optional
            The parameter to plot on the x-axis.
            The default is "fusionreg".
        width_scalar : int, optional
            The width of the chart. The default is 400.
        height : int, optional
            The height of the chart. The default is 400.
        return_data : bool, optional
            Whether to return the underlying data. The default is False.
        r : int, optional
            The exponential of the correlation coefficient reported.
            May be either 1 for pearson,
            2 for coefficient of determination (r-squared),
            The default is 1.
        **kwargs : dict
            The keyword arguments to pass to the
            :func:`multidms.model_collection.ModelCollection.split_apply_combine_muts`
            method. See the method docstring for details.

        Returns
        -------
        altair.Chart or Tuple(altair.Chart, pd.DataFrame)
            A chart object which can be displayed in a jupyter notebook
            or saved to a file. If `return_data=True`, then a tuple
            containing the chart and the underlying data will be returned.
        """
        if r not in [1, 2]:
            raise ValueError("invalid r, must be 1 or 2")

        query = "dataset_name.notna()" if "query" not in kwargs else kwargs["query"]
        if len(self.fit_models.query(query).dataset_name.unique()) < 2:
            raise ValueError("Must specify a subset of fits with multiple datasets")

        muts_df = self.split_apply_combine_muts(
            groupby=("dataset_name", x), **kwargs
        ).reset_index()

        replicate_series = []
        comparisons = list(it.combinations(muts_df.dataset_name.unique(), 2))
        for datasets in comparisons:
            wide_df = (
                muts_df.query(f"dataset_name.isin({datasets})")
                .drop(
                    [
                        c
                        for c in muts_df.columns
                        if "times_seen" in c or c in ("sites", "wts", "muts")
                    ],
                    axis=1,
                )
                .pivot(columns=["dataset_name", x], index="mutation")
            )
            wide_df.columns.names = ["mut_param"] + wide_df.columns.names[1:]
            for (mut_param, x_i), replicate_params_df in wide_df.T.groupby(
                ["mut_param", x]
            ):
                replicate_series.append(
                    pd.DataFrame(
                        {
                            "datasets": ",".join(datasets),
                            "mut_param": mut_param,
                            "correlation": replicate_params_df.T.corr().iloc[0, 1] ** r,
                            x: x_i,
                        },
                        index=[0],
                    ),
                )

        replicate_df = my_concat(replicate_series)

        chart = multidms.plot.mut_param_dataset_correlation(
            replicate_df,
            x=x,
            r=r,
            width_scalar=width_scalar,
            height=height,
        )

        if return_data:
            return chart, replicate_df
        return chart
