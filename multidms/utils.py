"""

==========
Utils
==========

This module contains utility functions for the multidms package.
"""

import copy
import pprint
import re
import time
from functools import reduce
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as onp
import pandas as pd
import multidms.biophysical

substitution_column = "aa_substitutions_reference"
experiment_column = "homolog_exp"
scaled_func_score_column = "log2e"


def is_wt(string):
    """Check if a string is a wildtype sequence"""
    return True if len(string.split()) == 0 else False


def split_sub(sub_string):
    """String match the wt, site, and sub aa
    in a given string denoting a single substitution
    """
    pattern = r"(?P<aawt>[A-Z])(?P<site>[\d\w]+)(?P<aamut>[A-Z\*])"
    match = re.search(pattern, sub_string)
    assert match is not None, sub_string
    return match.group("aawt"), str(match.group("site")), match.group("aamut")


def split_subs(subs_string, parser=split_sub):
    """Wrap the split_sub func to work for a
    string contining multiple substitutions
    """
    wts, sites, muts = [], [], []
    for sub in subs_string.split():
        wt, site, mut = parser(sub)
        wts.append(wt)
        sites.append(site)
        muts.append(mut)
    return wts, sites, muts


def fit_wrapper(
    dataset,
    huber_scale_huber=1,
    scale_coeff_lasso_shift=2e-5,
    scale_coeff_ridge_beta=0,
    scale_coeff_ridge_shift=0,
    scale_coeff_ridge_gamma=0,
    scale_coeff_ridge_ch=0,
    data_idx=0,
    epistatic_model="Identity",
    output_activation="Identity",
    lock_beta=False,
    lock_beta_naught=None,
    gamma_corrected=True,
    alpha_d=False,
    init_beta_naught=0.0,
    warmup_beta=False,
    tol=1e-3,
    num_training_steps=10,
    iterations_per_step=2000,
    save_model_at=[2000, 10000, 20000],
    PRNGKey=0,
):
    """
    Fit a multidms model to a dataset. This is a wrapper around the multidms
    fit method that allows for easy specification of the fit parameters.
    This method is helpful for comparing and organizing multiple fits.

    Parameters
    ----------
    dataset : :class:`multidms.Data`
        The dataset to fit to.
    huber_scale_huber : float, optional
        The scale of the huber loss function. The default is 1.
    scale_coeff_lasso_shift : float, optional
        The scale of the lasso penalty on the shift parameter. The default is 2e-5.
    scale_coeff_ridge_beta : float, optional
        The scale of the ridge penalty on the beta parameter. The default is 0.
    scale_coeff_ridge_shift : float, optional
        The scale of the ridge penalty on the shift parameter. The default is 0.
    scale_coeff_ridge_gamma : float, optional
        The scale of the ridge penalty on the gamma parameter. The default is 0.
    scale_coeff_ridge_ch : float, optional
        The scale of the ridge penalty on the ch parameter. The default is 0.
    data_idx : int, optional
        The index of the data to fit to. The default is 0.
    epistatic_model : str, optional
        The epistatic model to use. The default is "Identity".
    output_activation : str, optional
        The output activation function to use. The default is "Identity".
    lock_beta : bool, optional
        Whether to lock the beta parameter. The default is False.
    lock_beta_naught : float or None optional
        The value to lock the beta_naught parameter to. If None,
        the beta_naught parameter is free to vary. The default is None.
    gamma_corrected : bool, optional
        Whether to use the gamma corrected model. The default is True.
    alpha_d : bool, optional
        Whether to use the conditional c model. The default is False.
    init_beta_naught : float, optional
        The initial value of the beta_naught parameter. The default is 0.0.
    warmup_beta : bool, optional
        Whether to warmup the model by fitting beta parameters to the
        reference dataset before fitting the full model. The default is False.
    tol : float, optional
        The tolerance for the fit. The default is 1e-3.
    num_training_steps : int, optional
        The number of training steps to perform. The default is 10.
    iterations_per_step : int, optional
        The number of iterations to perform per training step. The default is 2000.
    save_model_at : list, optional
        The iterations at which to save the model. The default is [2000, 10000, 20000].
    PRNGKey : int, optional
        The PRNGKey to use to initialize model parameters. The default is 0.

    Returns
    -------
    fit_series : :class:`pandas.Series`
        A series containing the fit attributes and pickled model objects
        at the specified save_model_at steps.
    """
    if lock_beta and not warmup_beta:
        raise ValueError("Cannot lock beta without warming up beta")

    fit_attributes = locals().copy()
    biophysical_model = {
        "Identity": multidms.biophysical.identity_activation,
        "Sigmoid": multidms.biophysical.sigmoidal_global_epistasis,
        "NN": multidms.biophysical.nn_global_epistasis,
        "Softplus": multidms.biophysical.softplus_activation,
    }

    imodel = multidms.Model(
        dataset,
        epistatic_model=biophysical_model[fit_attributes["epistatic_model"]],
        output_activation=biophysical_model[fit_attributes["output_activation"]],
        alpha_d=fit_attributes["alpha_d"],
        gamma_corrected=fit_attributes["gamma_corrected"],
        init_beta_naught=fit_attributes["init_beta_naught"],
        PRNGKey=PRNGKey,
    )

    if fit_attributes["warmup_beta"]:
        imodel.fit_reference_beta()

    lock_params = {}
    if fit_attributes["lock_beta"]:
        lock_params["beta"] = imodel.params["beta"]

    if fit_attributes["lock_beta_naught"] is not None:
        lock_params["beta_naught"] = jnp.array([fit_attributes["lock_beta_naught"]])

    fit_attributes["step_loss"] = onp.zeros(num_training_steps)
    print("running:")
    pprint.pprint(fit_attributes)

    total_iterations = 0

    for training_step in range(num_training_steps):
        start = time.time()
        imodel.fit(
            lasso_shift=fit_attributes["scale_coeff_lasso_shift"],
            maxiter=iterations_per_step,
            tol=tol,
            huber_scale=fit_attributes["huber_scale_huber"],
            lock_params=lock_params,
            scale_coeff_ridge_shift=fit_attributes["scale_coeff_ridge_shift"],
            scale_coeff_ridge_beta=fit_attributes["scale_coeff_ridge_beta"],
            scale_coeff_ridge_gamma=fit_attributes["scale_coeff_ridge_gamma"],
            scale_coeff_ridge_ch=fit_attributes["scale_coeff_ridge_ch"],
        )
        end = time.time()

        fit_time = round(end - start)
        total_iterations += iterations_per_step

        if onp.isnan(float(imodel.loss)):
            break

        fit_attributes["step_loss"][training_step] = float(imodel.loss)

        print(
            f"training_step {training_step}/{num_training_steps},"
            f"Loss: {imodel.loss}, Time: {fit_time} Seconds",
            flush=True,
        )

        if total_iterations in save_model_at:
            fit_attributes[f"model_{total_iterations}"] = copy.copy(imodel)

    fit_series = pd.Series(fit_attributes).to_frame().T

    return fit_series


def plot_loss_simple(models):
    """
    Plot the loss of a set of models.
    Uses :func:`matplotlib.pyplot.show` to display the plot.


    Parameters
    ----------
    models : :class:`pandas.DataFrame`
        A dataframe where each row is the fit attributes of a model
        as output by the :func:`multidms.utils.fit_wrapper` function.


    Returns
    -------
    None.
    """
    fig, ax = plt.subplots(figsize=[7, 7])
    for model, model_row in models.iterrows():
        loss = model_row["epoch_loss"]
        iterations = [
            (i + 1) * model_row["step_size"]
            for i in range(model_row["num_training_steps"])
        ]

        ax.plot(iterations, loss[0], lw=3, linestyle="--", label=f"model: {model}")

    ax.set_ylabel("Loss")
    ax.set_xlabel("Iterations")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.show()


def combine_replicate_muts(
    fit_dict, times_seen_threshold=3, how="inner", phenotype_as_effect=True
):
    """
    Take a dictionary of fit objects, with key's as the prefix for individual
    replicate values, and merge then such that all individual and average mutation
    values are present in both.
    """
    # obtain and curate each of the replicate mutational dataframes
    mutations_dfs = []
    for replicate, fit in fit_dict.items():
        fit_mut_df = fit.get_mutations_df(phenotype_as_effect=phenotype_as_effect)
        new_column_name_map = {c: f"{replicate}_{c}" for c in fit_mut_df.columns}
        fit_mut_df = fit_mut_df.rename(new_column_name_map, axis=1)

        times_seen_cols = [c for c in fit_mut_df.columns if "times" in c]
        for c in times_seen_cols:
            fit_mut_df = fit_mut_df[fit_mut_df[c] >= times_seen_threshold]
        mutations_dfs.append(fit_mut_df)

    # merge each of the replicate mutational dataframes
    mut_df = reduce(
        lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how=how
        ),
        mutations_dfs,
    )

    column_order = []
    # now compute replicate averages
    for c in fit.mutations_df.columns:
        if "times_seen" in c:
            continue
        cols_to_combine = [f"{replicate}_{c}" for replicate in fit_dict.keys()]

        # just keep one replicate wt, site, mut .. as they are shared.
        if c in ["wts", "sites", "muts"]:
            mut_df[c] = mut_df[cols_to_combine[0]]
            mut_df.drop(cols_to_combine, axis=1, inplace=True)

        # take the average.
        else:
            mut_df[f"avg_{c}"] = mut_df[cols_to_combine].mean(axis=1)
            column_order += cols_to_combine + [f"avg_{c}"]

    return mut_df.loc[:, ["wts", "sites", "muts"] + column_order]
