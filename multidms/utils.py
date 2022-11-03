#!/usr/bin/env python

import pandas as pd
import numpy as np
import re
import json
import math
from functools import partial
from timeit import default_timer as timer
import binarymap as bmap
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxopt import ProximalGradient
from jax.experimental import sparse
import jaxopt
import numpy as onp
from tqdm import tqdm

substitution_column = 'aa_substitutions_reference'
experiment_column = 'homolog_exp'
scaled_func_score_column = 'log2e'


def is_wt(string):
    return True if len(string.split()) == 0 else False

   
def split_sub(sub_string):
    """String match the wt, site, and sub aa
    in a given string denoting a single substitution"""
    
    pattern = r'(?P<aawt>[A-Z])(?P<site>[\d\w]+)(?P<aamut>[A-Z\*])'
    match = re.search(pattern, sub_string)
    assert match != None, sub_string
    return match.group('aawt'), str(match.group('site')), match.group('aamut')


def split_subs(subs_string, parser=split_sub):
    """wrap the split_sub func to work for a 
    string contining multiple substitutions"""
    
    wts, sites, muts = [], [], []
    for sub in subs_string.split():
        wt, site, mut = parser(sub)
        wts.append(wt); sites.append(site); muts.append(mut)
    return wts, sites, muts


def scale_func_score(
    func_score_df, 
    bottleneck=1e5, 
    pseudocount = 0.1
):
   
    ret = func_score_df.copy()
    for (h, hdf) in func_score_df.groupby("condition"):

        post_counts_sum = sum(hdf['post_count'])
        scaling_factor = bottleneck / post_counts_sum

        hdf['orig_post_count'] = hdf['post_count']
        hdf['post_count'] *= scaling_factor
        hdf['post_count_wt'] *= scaling_factor

        hdf['pre_count_ps'] = hdf['pre_count'] + pseudocount
        hdf['post_count_ps'] = hdf['post_count'] + pseudocount
        hdf['pre_count_wt_ps'] = hdf['pre_count_wt'] + pseudocount
        hdf['post_count_wt_ps'] = hdf['post_count_wt'] + pseudocount

        total_pre_count = sum(hdf['pre_count_ps'])
        total_post_count = sum(hdf['post_count_ps'])

        hdf['pre_freq'] = hdf['pre_count_ps'] / total_pre_count
        hdf['post_freq'] = hdf['post_count_ps'] / total_post_count
        hdf['pre_freq_wt'] = hdf['pre_count_wt_ps'] / total_pre_count
        hdf['post_freq_wt'] = hdf['post_count_wt_ps'] / total_post_count

        hdf['wt_e'] = hdf['post_freq_wt'] / hdf['pre_freq_wt']
        hdf['var_e'] = hdf['post_freq'] / hdf['pre_freq']
        hdf['e'] = hdf['var_e'] / hdf['wt_e']

        ret.loc[hdf.index, 'func_score'] = hdf['e'].apply(lambda x: math.log(x, 2))

    return ret
