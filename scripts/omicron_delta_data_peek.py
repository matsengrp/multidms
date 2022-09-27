#!/usr/bin/env python

########################################
########################################
########################################
# NOTEBOOK

import os
import pickle
from itertools import combinations
import math
import pandas as pd
import re
import json
import binarymap as bmap
import matplotlib.pyplot as plt
import seaborn as sns
import plotnine
import jax
import jax.numpy as jnp
from jax.experimental import sparse
from jaxopt import ProximalGradient
import jaxopt
import numpy as onp
from scipy.stats import pearsonr
#from tqdm.notebook import tqdm
from tqdm import tqdm

import sys
sys.path.append("..")
from multidms.utils import create_homolog_modeling_data, initialize_model_params
from timeit import default_timer as timer
from multidms.model import ϕ, g, prox, cost_smooth


# ### Globals

substitution_column = 'aa_substitutions_reference'
experiment_column = 'homolog_exp'
scaled_func_score_column = 'log2e'

def run_fit(func_score_data, fit_params:dict):

    if fit_params["experiment_2"]:
        func_score_data = func_score_data.query(
            f"{experiment_column}.isin(['{fit_params['experiment_ref']}', '{fit_params['experiment_2']}'])"
        )

    func_score_df = pd.DataFrame()
    for idx, row in tqdm(func_score_data.iterrows(), total=len(func_score_data)):
        df = row.func_sel_scores_df.assign(homolog=row.homolog)
        df = df.assign(library = row.library)
        df = df.assign(replicate = row.replicate)
        exp_func_score_df = df.assign(homolog_exp=row.homolog_exp)
        func_score_df = pd.concat([func_score_df, exp_func_score_df])

    ##################################

    if fit_params["sample"]:
        func_score_df = func_score_df.sample(fit_params["sample"])

    func_score_df.aa_substitutions_reference.fillna("", inplace=True)
    gapped_sub_vars = []
    for idx, row in tqdm(func_score_df.iterrows(), total=len(func_score_df)):
        if "-" in row[substitution_column]:
            gapped_sub_vars.append(idx)

    stop_wt_vars = []
    for idx, row in tqdm(func_score_df.iterrows(), total=len(func_score_df)):
        for sub in row[substitution_column].split():
            if sub[0] == "*":
                stop_wt_vars.append(idx)

    to_drop = set.union(set(gapped_sub_vars), set(stop_wt_vars))
    func_score_df.drop(to_drop, inplace=True)


    # TODO re-write and make function
    # def normalize_by_freq()?
    dfs = []
    for (h, hdf) in func_score_df.groupby(fit_params["fs_scaling_group_column"]):
        n_post_counts = sum(hdf['post_count'])
        if 'Delta' in h:
            bottleneck = 1e5
            scaling_factor = bottleneck / n_post_counts # scaling_factor = 0.05
        else:
            bottleneck = 1e5
            scaling_factor = bottleneck / n_post_counts # scaling_factor = 0.05
        hdf['orig_post_count'] = hdf['post_count']
        hdf['post_count'] *= scaling_factor
        hdf['post_count_wt'] *= scaling_factor
        print(h, n_post_counts, round(scaling_factor, 2), round(sum(hdf['post_count']),2))

        # Recompute enrichment ratios with new counts
        hdf['pre_count_ps'] = hdf['pre_count'] + fit_params["pseudocount"]
        hdf['post_count_ps'] = hdf['post_count'] + fit_params["pseudocount"]
        hdf['pre_count_wt_ps'] = hdf['pre_count_wt'] + fit_params["pseudocount"]
        hdf['post_count_wt_ps'] = hdf['post_count_wt'] + fit_params["pseudocount"]

        total_pre_count = sum(hdf['pre_count_ps'])
        total_post_count = sum(hdf['post_count_ps'])

        hdf['pre_freq'] = hdf['pre_count_ps'] / total_pre_count
        hdf['post_freq'] = hdf['post_count_ps'] / total_post_count
        hdf['pre_freq_wt'] = hdf['pre_count_wt_ps'] / total_pre_count
        hdf['post_freq_wt'] = hdf['post_count_wt_ps'] / total_post_count

        hdf['wt_e'] = hdf['post_freq_wt'] / hdf['pre_freq_wt']
        hdf['var_e'] = hdf['post_freq'] / hdf['pre_freq']
        hdf['e'] = hdf['var_e'] / hdf['wt_e']
        #hdf.dropna(subset=['e'], inplace=True)
        hdf['log2e'] = hdf['e'].apply(lambda x: math.log(x, 2))
        dfs.append(hdf)
    func_score_df = pd.concat(dfs)
    # return


    if fit_params["agg_variants"]:
        func_score_df = func_score_df.groupby([substitution_column, experiment_column]).mean().reset_index()
        func_score_df["pre_count"] = func_score_df["pre_count"].astype(int)
        func_score_df["post_count"] = func_score_df["post_count"].astype(int)

    # TODO, what's the order of operations?
    if fit_params["shift_func_score_target_nonref"]:
        h2_idx = func_score_df.query(f"{experiment_column} == '{fit_params['experiment_2']}'").index
        func_score_df.loc[h2_idx, fit_params["func_score_target"]] += fit_params["shift_func_score_target_nonref"]

    if fit_params['clip_target']:
        tar = fit_params["func_score_target"]
        func_score_df[tar] = func_score_df[tar].clip(*fit_params["clip_target"])

    (X, y), df, all_subs, site_map = create_homolog_modeling_data(
                                func_score_df, 
                                experiment_column,
                                fit_params["experiment_ref"],
                                substitution_column,
                                fit_params["func_score_target"]
                            )

    sig_upper = df[fit_params["func_score_target"]].quantile(0.95)
    sig_lower = df[fit_params["func_score_target"]].quantile(0.05)
    sig_range = sig_upper - sig_lower

    # Initialize all params
    params = initialize_model_params(
        func_score_df[experiment_column].unique(), 
        n_beta_shift_params=X[fit_params["experiment_ref"]].shape[1],
        include_alpha=True,
        init_sig_range=sig_range,
        init_sig_min=sig_lower
    )

    print(f"\nPre-Optimization")
    print(f"----------------")
    print(f"cost = {cost_smooth(params, (X, y), λ_ridge=fit_params['λ_ridge']):.2e}")

    tol = 1e-6
    maxiter = fit_params['maxiter']
    start = timer()

    solver = ProximalGradient(cost_smooth, prox, tol=tol, maxiter=maxiter)

    # First, just fit data on reference homolog
    if fit_params["warmup_to_ref"]:
        print('Fitting model to just the reference homolog')
        params, state = solver.run(
            params, 
            hyperparams_prox = dict(
                lasso_params = None,
                lock_params = {
                    f"S_{fit_params['experiment_ref']}" : jnp.zeros(len(params['β'])),
                    f"S_{fit_params['experiment_2']}" : jnp.zeros(len(params['β'])),
                    f"C_{fit_params['experiment_ref']}" : jnp.zeros(shape=(1,)),
                    f"C_{fit_params['experiment_2']}" : jnp.zeros(shape=(1,))
                }
            ),
            data=(
                {fit_params['experiment_ref'] : X[fit_params['experiment_ref']]},
                {fit_params['experiment_ref'] : y[fit_params['experiment_ref']]}
            ),
            λ_ridge=0
        )

    # Next, jointly fit data on both homologs
    print('Fitting model to both homologs')
    params, state = solver.run(
        params, 
        hyperparams_prox = dict(
            lasso_params = {
                f"S_{fit_params['experiment_2']}" : fit_params['λ_lasso']
            },
            lock_params = {
                f"S_{fit_params['experiment_ref']}" : jnp.zeros(len(params['β'])),
                f"C_{fit_params['experiment_ref']}" : jnp.zeros(shape=(1,)),
                f"C_{fit_params['experiment_2']}" : jnp.zeros(shape=(1,))
            }
        ),
        data=(X, y),
        λ_ridge=fit_params['λ_ridge']
    )
    end = timer()

    print(f"\nPost-Optimization")
    print(f"-----------------")
    print(f"Full model optimization: {state.iter_num} iterations")
    print(f"error = {state.error:.2e}")
    print(f"cost = {cost_smooth(params, (X, y)):.2e}")
    print(f"Wall time for fit: {end - start}")

    for param in ["β", f"S_{fit_params['experiment_ref']}", f"S_{fit_params['experiment_2']}"]:
        print(f"\nFit {param} distribution\n===============")
        if param not in params:
            continue
        arr = onp.array(params[param])
        mean = onp.mean(arr)
        median = onp.median(arr)

        # measures of dispersion
        min = onp.amin(arr)
        max = onp.amax(arr)
        range = onp.ptp(arr)
        variance = onp.var(arr)
        sd = onp.std(arr)

        print("Descriptive analysis")
        print("Measures of Central Tendency")
        print(f"Mean = {mean:.2e}")
        print(f"Median = {median:.2e}")
        print("Measures of Dispersion")
        print(f"Minimum = {min:.2e}")
        print(f"Maximum = {max:.2e}")
        print(f"Range = {range:.2e}")
        print(f"Variance = {variance:.2e}")
        print(f"Standard Deviation = {sd:.2e}")

    # if f"C_{fit_params['experiment_2']}" in params:
    #     print(f"\nC_{fit_params['experiment_2']}: {params[f"C_{fit_params['experiment_2']}]}"

    print(f"\nFit Sigmoid Parameters, α\n================")
    for param, value in params['α'].items():
        print(f"{param}: {value}")

    df["predicted_latent_phenotype"] = onp.nan
    df[f"predicted_{fit_params['func_score_target']}"] = onp.nan

    print(f"\nRunning Predictions")
    print(f"-------------------")
    for homolog, hdf in df.groupby(experiment_column):
        h_params = {"β":params["β"], "S":params[f"S_{homolog}"], "C":params[f"C_{homolog}"]}
        z_h = ϕ(h_params, X[homolog])
        df.loc[hdf.index, "predicted_latent_phenotype"] = z_h
        y_h_pred = g(params["α"], z_h)
        df.loc[hdf.index, f"predicted_{fit_params['func_score_target']}"] = y_h_pred

    row = fit_params.copy()
    row["tuned_model_params_dict"] = params.copy()
    row["all_subs_list"] = all_subs.copy()
    row["variant_prediction_df"] = df.drop("index", axis=1)
    return pd.Series(row)

# ### Define Fit Params

# ## Read in metadata on homolog DMS experiments

func_score_data = pd.DataFrame()
sites = {}
wt_seqs = {}


for homolog in ["Delta", "Omicron_BA.1"]:
    
    # functional scores
    func_sel = pd.read_csv(f"../results/{homolog}/functional_selections.csv")
    func_sel = func_sel.assign(
        filename = f"../results/{homolog}/" + 
        func_sel.library + "_" + 
        func_sel.preselection_sample + 
        "_vs_" + func_sel.postselection_sample + 
        "_func_scores.csv"
    )
    func_sel = func_sel.assign(
        func_sel_scores_df = func_sel.filename.apply(lambda f: pd.read_csv(f))
    )
    func_sel = func_sel.assign(
        len_func_sel_scores_df = func_sel.func_sel_scores_df.apply(lambda x: len(x))
    )
    fun_sel = func_sel.assign(homolog = homolog)
    func_score_data = pd.concat([func_score_data, fun_sel]).reset_index(drop=True)
    
    # WT Protein sequence
    with open(f"../results/{homolog}/protein.fasta", "r") as seq_file:
        header = seq_file.readline()
        wt_seqs[homolog] = seq_file.readline().strip()

    # Sites
    sites[homolog] = (
        pd.read_csv(f"../results/{homolog}/site_numbering_map.csv")
        .rename({"sequential_site":f"{homolog}_site", "sequential_wt":f"{homolog}_wt"})
        .set_index(["reference_site"])
    )

# Add a column that gives a unique ID to each homolog/DMS experiment
func_score_data['homolog_exp'] = func_score_data.apply(
    lambda row: f"{row['homolog']}-{row['library']}-{row['replicate']}".replace('-Lib',''),
    axis=1
)


fit_params = {
    "fs_scaling_group_column" : "homolog_exp",
    "min_pre_counts" : 100,
    "pseudocount" : 0.1,
    "agg_variants" : True,
    "sample" : 1000,
    "min_pre_counts" : 100,
    "clip_target" : None,
    "func_score_target" : 'log2e',
    "experiment_ref" :'Delta-3-1',
    "experiment_2" : 'Omicron_BA.1-3-1',
    "shift_func_score_target_nonref" : -1,
    "warmup_to_ref" : False,
    "maxiter" : 5,
    "λ_lasso" : 5e-5,
    "λ_ridge" : 0
}

if not os.path.exists("results.pkl"):
    cols = list(fit_params.keys()) + ["tuned_model_params", "all_subs", "dataset_preds"]
    results = pd.DataFrame(columns = cols)
else:
    results = pickle.load(open("results.pkl", "rb"))


delta_exps = [exp_row.homolog_exp for exp, exp_row in func_score_data.iterrows() if "Delta" in exp_row.homolog_exp]
omicron_exps = [exp_row.homolog_exp for exp, exp_row in func_score_data.iterrows() if "Omicron" in exp_row.homolog_exp]

for delta_exp in delta_exps:
    for omicron_exp in omicron_exps:
        print(f"running {delta_exp} Vs. {omicron_exp}")
        fit_param_i = fit_params.copy()
        fit_param_i["experiment_ref"] = delta_exp
        fit_param_i["experiment_2"] = omicron_exp
        row = run_fit(func_score_data.copy(), fit_params = fit_param_i)
        results.loc[len(results)] = row
        break

pickle.dump(results, open("results.pkl", "wb"))
