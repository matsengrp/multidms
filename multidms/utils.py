#!/usr/bin/env python

import pandas as pd
import numpy as np
import re
import json
import binarymap as bmap
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental import sparse
import jaxopt
import numpy as onp
from tqdm import tqdm
tqdm.pandas()


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

        # compute bundle muts for a specific site
        for idx, row in tqdm(hom_func_df.iterrows(), total=len(hom_func_df)):
            var_map = site_map[[reference_homolog, hom]].copy()
            for wt, site, mut in zip(row.wts, row.sites, row.muts):
                var_map.loc[site, hom] = mut
            nis = var_map.where(
                var_map[reference_homolog] != var_map[hom]
            ).dropna()
            muts = nis[reference_homolog] + nis.index + nis[hom]
            ret_fs_df.loc[idx, "var_wrt_ref"] = " ".join(muts.values)

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
