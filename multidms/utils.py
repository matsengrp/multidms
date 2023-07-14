#!/usr/bin/env python

# TODO docstrings and add to autodoc

import pandas as pd
import numpy as np
import re
import json
import math
from functools import partial
from functools import reduce
from timeit import default_timer as timer
import binarymap as bmap
import jax

# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxopt import ProximalGradient
from jax.experimental import sparse
import jaxopt
import numpy as onp
from tqdm import tqdm

import pprint
import time
import copy
import multidms.biophysical

substitution_column = "aa_substitutions_reference"
experiment_column = "homolog_exp"
scaled_func_score_column = "log2e"


def is_wt(string):
    return True if len(string.split()) == 0 else False


def split_sub(sub_string):
    """String match the wt, site, and sub aa
    in a given string denoting a single substitution"""

    pattern = r"(?P<aawt>[A-Z])(?P<site>[\d\w]+)(?P<aamut>[A-Z\*])"
    match = re.search(pattern, sub_string)
    assert match != None, sub_string
    return match.group("aawt"), str(match.group("site")), match.group("aamut")


def split_subs(subs_string, parser=split_sub):
    """wrap the split_sub func to work for a
    string contining multiple substitutions"""

    wts, sites, muts = [], [], []
    for sub in subs_string.split():
        wt, site, mut = parser(sub)
        wts.append(wt)
        sites.append(site)
        muts.append(mut)
    return wts, sites, muts


# TODO cleanup and document
def fit_wrapper(
    dataset,
    δ_huber = 1,
    λ_lasso_shift = 2e-5,
    λ_ridge_beta = 0,
    λ_ridge_shift = 0,
    λ_ridge_gamma = 0,
    λ_ridge_ch = 0,
    data_idx = 0,
    epistatic_model = "Identity",
    output_activation = "Identity",
    lock_beta = False, 
    lock_C_ref = False,
    gamma_corrected = True,
    conditional_c = False,
    init_C_ref = 0.0,
    warmup_beta = False,
    tol=1e-3,
    num_training_steps = 10,
    iterations_per_step = 2000,
    save_model_at = [2000, 10000, 20000],
    PRNGKey=0
):
    """
    """
    
    fit_attributes = locals().copy()
    biophysical_model = {
        "Identity" : multidms.biophysical.identity_activation,
        "Sigmoid" : multidms.biophysical.sigmoidal_global_epistasis,
        "NN" : multidms.biophysical.nn_global_epistasis,
        "Softplus" : multidms.biophysical.softplus_activation
    }
    
    
    imodel = multidms.MultiDmsModel(
        dataset,
        epistatic_model=biophysical_model[fit_attributes['epistatic_model']],
        output_activation=biophysical_model[fit_attributes['output_activation']],
        conditional_c=fit_attributes['conditional_c'],
        gamma_corrected=fit_attributes['gamma_corrected'],
        init_C_ref=fit_attributes['init_C_ref'],
        PRNGKey=PRNGKey
    )

    if fit_attributes["warmup_beta"]:
        imodel.fit_reference_beta()

    lock_params = {}
    if fit_attributes["lock_beta"]:
        lock_params["β"] = imodel.params["β"]

    if fit_attributes["lock_C_ref"] != False:
        lock_params["C_ref"] = jnp.array([fit_attributes["lock_C_ref"]])

    fit_attributes['step_loss'] = onp.zeros(num_training_steps)
    print(f"running:")
    pprint.pprint(fit_attributes)

    total_iterations = 0

    for training_step in range(num_training_steps):

        start = time.time()
        imodel.fit(
            lasso_shift = fit_attributes['λ_lasso_shift'],
            maxiter=iterations_per_step, 
            tol=tol,
            δ=fit_attributes["δ_huber"],
            lock_params=lock_params,
            λ_ridge_shift = fit_attributes['λ_ridge_shift'],
            λ_ridge_beta = fit_attributes['λ_ridge_beta'],
            λ_ridge_gamma = fit_attributes['λ_ridge_gamma'],
            λ_ridge_ch = fit_attributes['λ_ridge_ch']
        )
        end = time.time()

        fit_time = round(end - start)
        total_iterations += iterations_per_step
        
        if onp.isnan(float(imodel.loss)):
            break
            
        fit_attributes['step_loss'][training_step] = float(imodel.loss)

        print(
            f"training_step {training_step}/{num_training_steps}, Loss: {imodel.loss}, Time: {fit_time} Seconds",
            flush=True
        )

        if total_iterations in save_model_at:
            fit_attributes[f"model_{total_iterations}"] = copy.copy(imodel)
              
    fit_series = pd.Series(fit_attributes).to_frame().T
            
    return fit_series

def plot_loss_simple(models):

    fig, ax = plt.subplots(figsize=[7,7])
    iterations = [(i+1)*2000 for i in range(10)]
    for model, model_row in models.iterrows():

        loss = model_row['epoch_loss']

        ax.plot(
            iterations, 
            loss[0],
            lw=3,
            linestyle = "--",
            label=f"model: {model}"
        )

    ax.set_ylabel(f"Loss")
    ax.set_xlabel(f"Iterations")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

    
def combine_replicate_muts(fit_dict, times_seen_threshold=3, how="inner"):
    """
    Take a dictionary of fit objects, with key's as the prefix for individual
    replicate values, and merge then such that all individual and average mutation
    values are present in both.
    """

    # obtain and curate each of the replicate mutational dataframes
    mutations_dfs = []
    for replicate, fit in fit_dict.items():

        fit_mut_df = fit.mutations_df.set_index("mutation")

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
        if c == "mutation" or "times_seen" in c:
            continue
        cols_to_combine = [f"{replicate}_{c}" for replicate in fit_dict.keys()]
        
        # just keep one replicate wt, site, mut .. as they are shared.
        if c in ["wts", "sites", "muts"]:
            mut_df[c] = mut_df[cols_to_combine[0]]
            mut_df.drop(cols_to_combine, axis=1, inplace=True)
            
        # take the average.
        else:
            mut_df[f"avg_{c}"] = mut_df[cols_to_combine].mean(axis=1)
            column_order += (cols_to_combine + [f"avg_{c}"])

    return mut_df.loc[:, ["wts", "sites", "muts"] + column_order]
