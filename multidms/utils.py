#!/usr/bin/env python

import pandas as pd
import numpy as np
import re
import json
import math
from timeit import default_timer as timer
import binarymap as bmap
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxopt import ProximalGradient
from jax.experimental import sparse
import jaxopt
import numpy as onp
#from tqdm.notebook import tqdm
from tqdm import tqdm
tqdm.pandas()


from multidms.model import *

substitution_column = 'aa_substitutions_reference'
experiment_column = 'homolog_exp'
scaled_func_score_column = 'log2e'


def initialize_model_params(
        homologs: dict, 
        n_beta_shift_params: int,
        include_alpha=True,
        init_sig_range=10.,
        init_sig_min=-10.,
        latent_bias=5.
):
    """
    initialize a set of starting parameters for the JAX model.
    
    Parameters
    ----------
    
    homologs : list
        A list containing all possible target homolog 
        names.
    
    n_beta_shift_params: int
        The number of beta and shift parameters 
        (for each homolog) to initialize.
    
    include_alpha : book
        Initialize parameters for the sigmoid from
        the global epistasis function
    
    init_sig_range : float
        The range of observed phenotypes in the raw
        data, used to initialize the range of the
        sigmoid from the global epistasis function
    
    init_sig_min : float
        The lower bound of observed phenotypes in
        the raw data, used to initialize the minimum
        value of the sigmoid from the global epistasis
        funciton
        
    latent_bias : float
        bias parameter applied to the output of the
        linear model (latent prediction).
    
    Returns
    -------
    
    dict :
        all relevant parameters to be tuned with the JAX model.
    """
    
    params = {}
    seed = 0
    key = jax.random.PRNGKey(seed)

    # initialize beta parameters from normal distribution.
    params["β"] = jax.random.normal(shape=(n_beta_shift_params,), key=key)

    # initialize shift parameters
    for homolog in homologs:
        # We expect most shift parameters to be close to zero
        params[f"S_{homolog}"] = jnp.zeros(shape=(n_beta_shift_params,))
        params[f"C_{homolog}"] = jnp.zeros(shape=(1,))

    if include_alpha:
        params["α"]=dict(
            latent_bias=jnp.array([5.0]), # 5.0 is a guess, could update
            ge_scale=jnp.array([init_sig_range]),
            ge_bias=jnp.array([init_sig_min])
        )

    return params

def create_homolog_modeling_data(
    func_score_df:pd.DataFrame,
    homolog_name_col: str,
    reference_homolog: str,
    substitution_col: str,
    func_score_col: str
):
    """
    Takes a dataframe for making a `BinaryMap` object, and adds
    a column where each entry is a list of mutations in a variant
    relative to the amino-acid sequence of the reference homolog.
    
    Parameters
    ----------

    func_score_df : pandas.DataFrame
        This should be in the same format as described in BinaryMap.
    
    homolog_name_col : str
        The name of the column in func_score_df that identifies the
        homolog for a given variant. We require that the
        reference homolog variants are labeled as 'reference'
        in this column.
        
    reference_homolog : str
        The name of the homolog existing in ``homolog_name_col`` for
        which we should convert all substitution to be with respect to.
    
    substitution_col : str 
        The name of the column in func_score_df that
        lists mutations in each variant relative to the homolog wildtype
        amino-acid sequence where sites numbers must come from an alignment
        to a reference sequence (which may or may not be the same as the
        reference homolog).
        
    func_score_col : str
        Column in func_scores_df giving functional score for each variant.
        
    
    Returns
    -------
        
    tuple : (dict[BinaryMap], dict[jnp.array]), pd.DataFrame, np.array, pd.DataFrame
    
        This function return a tuple which can be unpacked into the following:
        
        - (X, y) Where X and y are both dictionaries containing the prepped data
            for training our JAX multidms model. The dictionary keys
            stratify the datasets by homolog
            
        - A pandas dataframe which primary contains the information from
            func_score_df, but has been curated to include only the variants
            deemed appropriate for training, as well as the substitutions
            converted to be wrt to the reference homolog.
            
        - A numpy array giving the substitutions (beta's) of the binary maps
            in the order that is preserved to match the matrices in X.
            
        - A pandas dataframe providing the site map indexed by alignment site to
            a column for each homolog wt amino acid. 
    
    """
   
    # TODO should we assert there's no mutations like, A154bT? 
    def split_sub(sub_string):
        """String match the wt, site, and sub aa
        in a given string denoting a single substitution"""
        
        pattern = r'(?P<aawt>[A-Z])(?P<site>[\d\w]+)(?P<aamut>[A-Z\*])'
        match = re.search(pattern, sub_string)
        assert match != None, sub_string
        return match.group('aawt'), str(match.group('site')), match.group('aamut')
    
    def split_subs(subs_string):
        """wrap the split_sub func to work for a 
        string contining multiple substitutions"""
        
        wts, sites, muts = [], [], []
        for sub in subs_string.split():
            wt, site, mut = split_sub(sub)
            wts.append(wt); sites.append(site); muts.append(mut)
        return wts, sites, muts

    # Add columns that parse mutations into wt amino acid, site,
    # and mutant amino acid
    ret_fs_df = func_score_df.reset_index()
    ret_fs_df["wts"], ret_fs_df["sites"], ret_fs_df["muts"] = zip(
        *ret_fs_df[substitution_col].map(split_subs)
    )

    # Use the substitution_col to infer the wildtype
    # amino-acid sequence of each homolog, storing this
    # information in a dataframe.
    site_map = pd.DataFrame(dtype="string")
    for hom, hom_func_df in ret_fs_df.groupby(homolog_name_col):
        for idx, row in hom_func_df.iterrows():
            for wt, site  in zip(row.wts, row.sites):
                site_map.loc[site, hom] = wt
    
    # Find all sites for which at least one homolog lacks data
    # (this can happen if there is a gap in the alignment)
    na_rows = site_map.isna().any(axis=1)
    print(f"Found {sum(na_rows)} site(s) lacking data in at least one homolog.")
    sites_to_throw = na_rows[na_rows].index
    site_map.dropna(inplace=True)
    
    # Remove all variants with a mutation at one of the above
    # "disallowed" sites lacking data
    def flags_disallowed(disallowed_sites, sites_list):
        """Check to see if a sites list contains 
        any disallowed sites"""
        for site in sites_list:
            if site in disallowed_sites:
                return False
        return True
    
    ret_fs_df["allowed_variant"] = ret_fs_df.sites.apply(
        lambda sl: flags_disallowed(sites_to_throw,sl)
    )
    n_var_pre_filter = len(ret_fs_df)
    ret_fs_df = ret_fs_df[ret_fs_df["allowed_variant"]]
    print(f"{n_var_pre_filter-len(ret_fs_df)} of the {n_var_pre_filter} variants"
          f" were removed because they had mutations at the above sites, leaving"
          f" {len(ret_fs_df)} variants.")

    # Duplicate the substitutions_col, then convert the respective subs to be wrt ref
    # using the function above
    ret_fs_df = ret_fs_df.assign(var_wrt_ref = ret_fs_df[substitution_col])
    for hom, hom_func_df in ret_fs_df.groupby(homolog_name_col):
        
        if hom == reference_homolog: continue
        variant_cache = {} 
        cache_hits = 0
        
        for idx, row in tqdm(hom_func_df.iterrows(), total=len(hom_func_df)):
            
            key = tuple(list(zip(row.wts, row.sites, row.muts)))
            if key in variant_cache:
                ret_fs_df.loc[idx, "var_wrt_ref"]  = variant_cache[key]
                cache_hits += 1
                continue
            
            var_map = site_map[[reference_homolog, hom]].copy()
            for wt, site, mut in zip(row.wts, row.sites, row.muts):
                var_map.loc[site, hom] = mut
            nis = var_map.where(
                var_map[reference_homolog] != var_map[hom]
            ).dropna()
            muts = nis[reference_homolog] + nis.index + nis[hom]
            
            mutated_seq = " ".join(muts.values)
            ret_fs_df.loc[idx, "var_wrt_ref"] = mutated_seq
            variant_cache[key] = mutated_seq
            
        print(f"There were {cache_hits} cache hits in total for homolog {hom}.")

    # Get list of all allowed substitutions for which we will tune beta parameters
    allowed_subs = {
        s for subs in ret_fs_df.var_wrt_ref
        for s in subs.split()
    }
    
    # Make BinaryMap representations for each homolog
    X, y = {}, {}
    for homolog, homolog_func_score_df in ret_fs_df.groupby(homolog_name_col):
        ref_bmap = bmap.BinaryMap(
            homolog_func_score_df,
            substitutions_col="var_wrt_ref",
            allowed_subs=allowed_subs
        )
        
        # convert binarymaps into sparse arrays for model input
        X[homolog] = sparse.BCOO.from_scipy_sparse(ref_bmap.binary_variants)
        
        # create jax array for functional score targets
        y[homolog] = jnp.array(homolog_func_score_df[func_score_col].values)
    
    ret_fs_df.drop(["wts", "sites", "muts"], axis=1, inplace=True)

    return (X, y), ret_fs_df, ref_bmap.all_subs, site_map 


def run_fit(func_score_data, fit_params:dict):
    """
    Awrapper for running a full analysis fit on 2 experiments
    from Bloom Lab.

    TODO Doc params and replace dict
    TODO dont need to mutate the incoming object.
    """

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

    if fit_params["sample"]:
        func_score_df = func_score_df.sample(fit_params["sample"])

    func_score_df.aa_substitutions_reference.fillna("", inplace=True)
    gapped_sub_vars = []
    stop_wt_vars = []
    non_numeric_sites = []
    for idx, row in tqdm(func_score_df.iterrows(), total=len(func_score_df)):
        if "-" in row[substitution_column]:
            gapped_sub_vars.append(idx)
        for sub in row[substitution_column].split():
            if sub[0] == "*":
                stop_wt_vars.append(idx)
            if not sub[-2].isnumeric():
                non_numeric_sites.append(idx)

    to_drop = set.union(set(gapped_sub_vars), set(stop_wt_vars), set(non_numeric_sites))
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
    
    # Drop barcoded variants with pre-counts below a threshold
    n_pre_threshold = len(func_score_df)
    func_score_df = func_score_df[func_score_df['pre_count'] >= fit_params["min_pre_counts"]]
#     print(f'Of {n_pre_threshold} variants, {n_pre_threshold - len(func_score_df)} had fewer than {min_pre_counts} counts before selection, and were filtered out')


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
    print(f"\nDONE :)")
    print(f"-------------------")

    row = fit_params.copy()
    row["tuned_model_params"] = params.copy()
    row["all_subs"] = all_subs.copy()
    row["variant_prediction_df"] = df.drop("index", axis=1)
    
    return pd.Series(row)
