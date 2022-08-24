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


def initialize_model_params(
        homologs: dict, 
        n_beta_shift_params: int, 
        n_perceptron_units=1,
        include_alpha=True,
        init_sig_range=10,
        init_sig_min=-10
):
    """
    initialize a set of starting parameters for the JAX model.
    
    Parameters
    ----------
    
    homologs : dict
        A dictionary containing all possible target homolog 
        names (keys) and sequences (values).
    
    n_beta_shift_params: int
        The number of beta and shift parameters 
        (for each homolog) to initialize.
        
    n_perceptron_units : int
        The number of connections that are tuned
        for the shape of global epistasis before
        being fed into a sigmoid.
    
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
    for homolog in homologs.keys():
        # We expect most shift parameters to be close to zero
        params[f"S_{homolog}"] = jnp.zeros(shape=(n_beta_shift_params,))

    # Perceptron parameterization 
    key, *subkeys = jax.random.split(key, num=4)
    if include_alpha:
        params["α"]=dict(
            biases=jnp.array([5.0] * n_perceptron_units), # 5.0 is a guess, could update
            a=jnp.array([init_sig_range] * n_perceptron_units),
            a_bias=jnp.array([init_sig_min] * n_perceptron_units)
        )

    return params


def create_homolog_modeling_data(
    func_score_df:pd.DataFrame, 
    homologs:dict,
    homolog_name_col: str,
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
        
    homologs : dict
        A dictionary containing all possible target homolog 
        names (keys) and sequences (values).
    
    homolog_name_col : str
        The name of the column in func_score_df that identifies the
        homolog for a given variant. We require that the
        reference homolog variants are labeled as 'reference'
        in this column.
    
    substitution_col : str 
        The name of the column in func_score_df that
        lists mutations in each variant relative to the wildtype
        amino-acid sequence of the homolog in which they occur.
        
    func_score_col : str
        Column in func_scores_df giving functional score for each variant.
    
    Returns
    -------
        
    TODO
    
    """
    
    def mutations_wrt_ref(mutations, hom_wtseq):
        """
        Takes a list of mutations for a given variant relative
        to its background homolog and returns a list of all
        mutations that separate the variant from the reference
        homolog.
        """
        
        # Compute the full amino-acid sequence of the
        # given variant
        mutated_homolog = list(hom_wtseq)
        for mutation in mutations.split():

            # TODO: Do we need to change the regex to allow
            # for gap '-' and stop '*' characters?
            pattern = r'(?P<aawt>\w)(?P<site>\d+)(?P<aamut>[\w\*])'
            match = re.search(pattern, mutation)
            assert match != None, mutation
            aawt = match.group('aawt')
            site = match.group('site')
            aamut = match.group('aamut')
            mutated_homolog[int(site)-1] = aamut
            
        hom_var_seq = ''.join(mutated_homolog)
        
        # Make a list of all mutations that separate the variant
        # from the reference homolog
        ref_muts = [
            f"{aaref}{i+1}{aavar}" 
            for i, (aaref, aavar) in enumerate(zip(homologs["reference"], hom_var_seq))
            if aaref != aavar
        ]
        
        return " ".join(ref_muts)

    # Duplicate the substitutions_col, then convert the respective functional scores
    func_score_df = func_score_df.assign(
            var_wrt_ref = func_score_df[substitution_col].values
    )
    for hom_name, hom_seq in homologs.items():
        if hom_name == "reference": continue
        hom_df = func_score_df.query(f"{homolog_name_col} == '{hom_name}'")
        hom_var_wrt_ref = [
            mutations_wrt_ref(muts, homologs[hom_name]) 
            for muts in hom_df[substitution_col]
        ]
        func_score_df.loc[hom_df.index.values, "var_wrt_ref"] = hom_var_wrt_ref   
    
    # Get list of all allowed substitutions that we will tune beta parameters for
    allowed_subs = {
        s for subs in func_score_df.var_wrt_ref
        for s in subs.split()
    }
    
    # Make BinaryMap representations for each homolog
    X, y = {}, {}
    for homolog, homolog_func_score_df in func_score_df.groupby("homolog"):
        ref_bmap = bmap.BinaryMap(
            homolog_func_score_df,
            substitutions_col="var_wrt_ref",
            allowed_subs=allowed_subs
        )
        
        # convert binarymaps into sparse arrays for model input
        X[homolog] = sparse.BCOO.fromdense(ref_bmap.binary_variants.toarray())
        
        # create jax array for functional score targets
        y[homolog] = jnp.array(homolog_func_score_df[func_score_col].values)
    
    return (X, y), func_score_df, ref_bmap.all_subs
