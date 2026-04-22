"""Shared utilities for the loss normalization experiment pipeline."""

import yaml


def load_config(config_path):
    """Load a pipeline configuration YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML config file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def _clip_range(value):
    """Normalize a beta_clip_range YAML value for jaxmodels.fit.

    The config value may be a 2-element list (box constraint), or null / None
    (no clipping). jaxmodels checks ``if beta_clip_range is not None`` to enable
    clipping, so None must propagate through unchanged; non-None must be a
    tuple of length 2.
    """
    if value is None:
        return None
    return tuple(value)


def build_fit_params(fit_config, datasets):
    """Build a fitting parameter dict that sweeps fusionreg AND l2reg.

    Unlike the standard simulation pipeline's ``build_fit_params`` which
    fixes ``l2reg`` to a single value, this version maps
    ``l2reg_values`` to a list for the ``l2reg`` sweep dimension.

    Parameters
    ----------
    fit_config : dict
        The ``fitting`` subsection of the experiment config.
    datasets : list
        List of ``multidms.Data`` objects to fit.

    Returns
    -------
    dict
        Ready to pass to ``multidms.model_collection.fit_models()``.
    """
    return {
        "maxiter": [fit_config["maxiter"]],
        "tol": [fit_config["tol"]],
        "fusionreg": fit_config["fusionreg_values"],
        "l2reg": fit_config["l2reg_values"],
        "beta0_ridge": [fit_config["beta0_ridge"]],
        "ge_type": [fit_config["ge_type"]],
        "ge_kwargs": [fit_config["ge_kwargs"]],
        "cal_kwargs": [fit_config["cal_kwargs"]],
        "loss_kwargs": [fit_config["loss_kwargs"]],
        "warmstart": [fit_config["warmstart"]],
        "beta0_init": [fit_config["beta0_init"]],
        "alpha_init": [fit_config["alpha_init"]],
        "share_alpha": [fit_config.get("share_alpha", True)],
        "beta_clip_range": [_clip_range(fit_config["beta_clip_range"])],
        "dataset": datasets,
    }
