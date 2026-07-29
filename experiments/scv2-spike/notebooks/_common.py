"""Shared utilities for the spike analysis pipeline notebooks."""

import os
from functools import reduce

import pandas as pd
import requests
import yaml


def load_config(config_path, downstream_config_path=None):
    """Load a pipeline configuration YAML, optionally merging a downstream tier.

    The spike config is split by dependency tier: the fit tier holds
    everything the expensive model fit depends on, and the downstream tier
    holds keys read only by evaluation and plotting. Splitting them keeps a
    downstream edit from invalidating the cached fit.

    Parameters
    ----------
    config_path : str
        Path to the fit-tier YAML config file.
    downstream_config_path : str or None
        Path to the downstream-tier YAML. When None, only the fit tier is
        loaded and the returned dict is exactly the file's contents.

    Returns
    -------
    dict
        Configuration dictionary. When both tiers are given, they are merged
        one level deep: top-level keys, and keys inside each shared top-level
        mapping.

    Raises
    ------
    ValueError
        If any key is present in both tiers. A collision is always a bug in
        the split, and resolving it silently would let a stale fit-tier value
        shadow the downstream one.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if downstream_config_path is None:
        return config

    with open(downstream_config_path) as f:
        downstream = yaml.safe_load(f) or {}

    for key, value in downstream.items():
        if key not in config:
            config[key] = value
            continue
        if isinstance(config[key], dict) and isinstance(value, dict):
            overlap = set(config[key]) & set(value)
            if overlap:
                raise ValueError(
                    f"Key(s) {sorted(overlap)} appear in both the fit tier "
                    f"({config_path}) and the downstream tier "
                    f"({downstream_config_path}) under '{key}'. Each key must "
                    "live in exactly one tier."
                )
            config[key].update(value)
        else:
            raise ValueError(
                f"Key '{key}' appears in both the fit tier ({config_path}) "
                f"and the downstream tier ({downstream_config_path}). Each "
                "key must live in exactly one tier."
            )

    return config


def download_data(config, cache_dir="results/raw_data"):
    """Download raw functional selection data from the public GitHub repository.

    Downloads ``functional_selections.csv`` and all referenced func_scores CSVs
    for each condition. Files are cached locally so subsequent runs skip the
    download.

    Parameters
    ----------
    config : dict
        The ``spike`` section of the pipeline config. Must contain
        ``data_url`` and ``experiment_conditions``.
    cache_dir : str
        Local directory for caching downloaded files.

    Returns
    -------
    dict
        Maps each condition name to a dict with ``"manifest"`` (DataFrame)
        and ``"paths"`` (list of local file paths).
    """
    data_url = config["data_url"]
    conditions = config["experiment_conditions"]
    condition_data = {}

    for condition in conditions:
        condition_dir = os.path.join(cache_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)

        # Download functional_selections.csv (experiment manifest)
        manifest_path = os.path.join(condition_dir, "functional_selections.csv")
        if not os.path.exists(manifest_path):
            url = f"{data_url}/{condition}/functional_selections.csv"
            print(f"  Downloading {url}")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(manifest_path, "w") as f:
                f.write(resp.text)

        manifest = pd.read_csv(manifest_path)

        # Download each func_scores CSV referenced in the manifest
        local_paths = []
        for _, row in manifest.iterrows():
            fname = (
                f"{row['library']}_{row['preselection_sample']}"
                f"_vs_{row['postselection_sample']}_func_scores.csv"
            )
            local_path = os.path.join(condition_dir, fname)
            if not os.path.exists(local_path):
                url = f"{data_url}/{condition}/{fname}"
                print(f"  Downloading {url}")
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                with open(local_path, "w") as f:
                    f.write(resp.text)
            local_paths.append(local_path)

        condition_data[condition] = {"manifest": manifest, "paths": local_paths}

    return condition_data


def load_raw_data(config, cache_dir="results/raw_data"):
    """Download (if needed) and load all raw functional score data.

    Parameters
    ----------
    config : dict
        The ``spike`` section of the pipeline config.
    cache_dir : str
        Local directory for caching downloaded files.

    Returns
    -------
    pandas.DataFrame
        All raw variant data with columns including ``pre_count``,
        ``post_count``, ``aa_substitutions``, ``condition`` (experiment
        name like ``"Delta-2"``), and ``homolog``.
    """
    condition_data = download_data(config, cache_dir)
    conditions = config["experiment_conditions"]

    all_dfs = []
    for condition in conditions:
        manifest = condition_data[condition]["manifest"]

        for local_path, (_, row) in zip(
            condition_data[condition]["paths"], manifest.iterrows()
        ):
            df = pd.read_csv(local_path)

            # Construct the experiment condition name: "{homolog}-{library_num}"
            # e.g. "Delta-2" from "Delta" + "Lib-2"
            lib_num = row["library"].replace("Lib-", "")
            experiment_name = f"{condition}-{lib_num}"

            df = df.assign(
                homolog=condition,
                condition=experiment_name,
            )
            all_dfs.append(df)

    raw_df = pd.concat(all_dfs, ignore_index=True)

    # Rename reference-numbering substitutions to standard name
    raw_df = raw_df.rename(
        {"aa_substitutions_reference": "aa_substitutions"}, axis=1
    ).fillna({"aa_substitutions": ""})

    return raw_df


def truncate_nonsense(row):
    """Truncate mutations in a variant at the first stop codon.

    For variants containing stop codons (``*``), keeps only mutations up to
    and including the first stop, then updates the substitution count.

    Parameters
    ----------
    row : pandas.Series
        Must have ``aa_substitutions`` (str) and ``n_subs`` (int) fields.

    Returns
    -------
    pandas.Series
        Copy of row with truncated substitutions and updated count.
    """
    if row.aa_substitutions:
        muts = row.aa_substitutions.split(" ")
        stop_idx = next((i for i, m in enumerate(muts) if "*" in m), None)
        if stop_idx is not None:
            new_muts = muts[: stop_idx + 1]
        else:
            new_muts = muts
        row = row.copy()
        row.aa_substitutions = " ".join(new_muts)
        row.n_subs = len(new_muts)
    return row


def build_fit_params(fit_config, datasets):
    """Build a standard fitting parameter dict from a config section.

    Parameters
    ----------
    fit_config : dict
        The ``fitting`` subsection of the pipeline config.
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
        "recompute_scale": [fit_config["recompute_scale"]],
        "fusionreg": fit_config["fusionreg_values"],
        "l2reg": [fit_config["l2reg"]],
        "beta0_ridge": [fit_config["beta0_ridge"]],
        "scale_fusion_by_n": [fit_config.get("scale_fusion_by_n", False)],
        "ge_type": [fit_config["ge_type"]],
        "ge_kwargs": [fit_config["ge_kwargs"]],
        "cal_kwargs": [fit_config["cal_kwargs"]],
        "loss_kwargs": [fit_config["loss_kwargs"]],
        "warmstart": [fit_config["warmstart"]],
        "beta0_init": [fit_config["beta0_init"]],
        "alpha_init": [fit_config["alpha_init"]],
        "share_alpha": [fit_config.get("share_alpha", True)],
        "beta_clip_range": [tuple(fit_config["beta_clip_range"])],
        "dataset": datasets,
    }


def combine_replicate_muts(
    fit_dict, predicted_func_scores=False, how="inner", **kwargs
):
    """Combine mutation DataFrames from replicate model fits.

    Parameters
    ----------
    fit_dict : dict
        Maps replicate name (str) to a fitted model object with a
        ``get_mutations_df(**kwargs)`` method.
    predicted_func_scores : bool
        If False (default), columns containing ``predicted_func_score``
        are excluded from the output.
    how : str
        Merge strategy passed to ``pd.merge`` (default ``"inner"``).
    **kwargs
        Forwarded to each model's ``get_mutations_df()``.

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame with per-replicate and ``avg_`` columns.
    """
    # Collect mutation DataFrames and track all parameter columns in one pass
    mutations_dfs = []
    all_cols = set()
    for replicate, fit in fit_dict.items():
        fit_mut_df = fit.get_mutations_df(**kwargs).reset_index()
        fit_mut_df = fit_mut_df.drop(
            [c for c in fit_mut_df.columns if "times_seen" in c], axis=1
        )
        all_cols.update(fit_mut_df.columns)
        new_column_name_map = {
            c: f"{replicate}_{c}" for c in fit_mut_df.columns if c != "mutation"
        }
        fit_mut_df = fit_mut_df.rename(new_column_name_map, axis=1)
        mutations_dfs.append(fit_mut_df)

    mut_df = reduce(
        lambda left, right: pd.merge(left, right, on="mutation", how=how),
        mutations_dfs,
    )

    meta_cols = ["mutation", "wts", "sites", "muts"]
    param_cols = sorted(c for c in all_cols if c not in meta_cols and c != "mutation")

    first_rep = list(fit_dict.keys())[0]
    for mc_col in meta_cols:
        col_name = f"{first_rep}_{mc_col}"
        if col_name in mut_df.columns:
            mut_df[mc_col] = mut_df[col_name]
    drop_meta = [
        f"{rep}_{mc_col}"
        for rep in fit_dict.keys()
        for mc_col in meta_cols
        if f"{rep}_{mc_col}" in mut_df.columns
    ]
    mut_df = mut_df.drop(drop_meta, axis=1)

    column_order = []
    for c in param_cols:
        if not predicted_func_scores and "predicted_func_score" in c:
            continue
        cols_to_combine = [
            f"{rep}_{c}" for rep in fit_dict.keys() if f"{rep}_{c}" in mut_df.columns
        ]
        if not cols_to_combine:
            continue
        mut_df[f"avg_{c}"] = mut_df[cols_to_combine].mean(axis=1)
        column_order += cols_to_combine + [f"avg_{c}"]

    return mut_df.loc[:, meta_cols + column_order]
