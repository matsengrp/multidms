#!/usr/bin/env python

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


def scale_func_score(func_score_df, bottleneck=1e5, pseudocount=0.1):

    ret = func_score_df.copy()
    for (h, hdf) in func_score_df.groupby("condition"):

        if "Delta" in h:
            bottleneck = 1e5
        elif "Omicron_BA.1" in h:
            bottleneck = 1.9e5
        elif "Omicron_BA.2" in h:
            bottleneck = 1.9e5
        else:
            raise ValueError(f"Could not parse homolog {h}")

        post_counts_sum = sum(hdf["post_count"])
        scaling_factor = bottleneck / post_counts_sum

        hdf["orig_post_count"] = hdf["post_count"]
        hdf["post_count"] *= scaling_factor
        hdf["post_count_wt"] *= scaling_factor

        hdf["pre_count_ps"] = hdf["pre_count"] + pseudocount
        hdf["post_count_ps"] = hdf["post_count"] + pseudocount
        hdf["pre_count_wt_ps"] = hdf["pre_count_wt"] + pseudocount
        hdf["post_count_wt_ps"] = hdf["post_count_wt"] + pseudocount

        total_pre_count = sum(hdf["pre_count_ps"])
        total_post_count = sum(hdf["post_count_ps"])

        hdf["pre_freq"] = hdf["pre_count_ps"] / total_pre_count
        hdf["post_freq"] = hdf["post_count_ps"] / total_post_count
        hdf["pre_freq_wt"] = hdf["pre_count_wt_ps"] / total_pre_count
        hdf["post_freq_wt"] = hdf["post_count_wt_ps"] / total_post_count

        hdf["wt_e"] = hdf["post_freq_wt"] / hdf["pre_freq_wt"]
        hdf["var_e"] = hdf["post_freq"] / hdf["pre_freq"]
        hdf["e"] = hdf["var_e"] / hdf["wt_e"]

        ret.loc[hdf.index, "func_score"] = hdf["e"].apply(lambda x: math.log(x, 2))

    return ret

def fit_wrapper(
    dataset,
    δ_huber = 1,
    λ_lasso_shift = 2e-5,
    λ_ridge_beta = 1e-6,
    λ_ridge_shift = 1e-6,
    data_idx = 0,
    epistatic_model = "identity",
    output_activation = "identity",
    lock_beta = False, 
    lock_C_ref = False,
    gamma_corrected = True,
    conditional_c = False,
    init_C_ref = 0.0,
    warmup_beta = False,
    num_training_steps = 10,
    iterations_per_step = 2000,
    save_model_at = [2000, 10000, 20000]
):
    
    fit_attributes = locals().copy()
    biophysical_model = {
        "identity" : multidms.model.identity_activation,
        "Sigmoid" : multidms.model.sigmoidal_global_epistasis,
        "Softplus" : multidms.model.softplus_activation
    }
    
    
    imodel = multidms.MultiDmsModel(
        dataset,
        epistatic_model=biophysical_model[fit_attributes['epistatic_model']],
        output_activation=biophysical_model[fit_attributes['output_activation']],
        conditional_c=fit_attributes['conditional_c'],
        gamma_corrected=fit_attributes['gamma_corrected'],
        init_C_ref=fit_attributes['init_C_ref']
    )

    if fit_attributes["warmup_beta"]:
        imodel.fit_reference_beta()

    lock_params = {}
    if fit_attributes["lock_beta"]:
        lock_params["β"] = imodel.params["β"]

    if fit_attributes["lock_C_ref"]:
        lock_params["C_ref"] = jnp.zeros(shape=(1,))

    fit_attributes['epoch_loss'] = []
    print(f"running:")
    pprint.pprint(fit_attributes)

    fit_attributes["total_iterations"] = 0
    ret = pd.DataFrame()

    for training_step in range(num_training_steps):

        start = time.time()
        imodel.fit(
            lasso_shift=fit_attributes['λ_lasso_shift'],
            λ_ridge_shift=fit_attributes['λ_ridge_shift'],
            λ_ridge_beta=fit_attributes['λ_ridge_beta'],
            maxiter=iterations_per_step, 
            tol=0.001,
            δ=fit_attributes["δ_huber"],
            lock_params=lock_params
        )
        end = time.time()

        fit_time = round(end - start)
        fit_attributes['total_iterations'] += iterations_per_step
        fit_attributes['epoch_loss'].append(float(imodel.loss))

        print(
            f"training_step {training_step}/{num_training_steps}, Loss: {imodel.loss}, Time: {fit_time} Seconds",
            flush=True
        )

        if fit_attributes['total_iterations'] in save_model_at:
            data_row = pd.Series(fit_attributes).to_frame().T
            data_row["model_object"] = copy.copy(imodel)
            ret = pd.concat([ret, data_row], ignore_index=True)
            
    return ret


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

    
def combine_replicate_muts(fit_dict, times_seen_threshold=3):
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
            left, right, left_index=True, right_index=True, how="inner"
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
