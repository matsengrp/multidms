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


# TODO generalize to a single non-identical wt sequence
def convert_subs_wrt_ref_seq_n(cond_var_map, condition_func_df):
    """
    Convert mutations seen in a set of variants defined by a 
    wildtype sequence to be with respect another 'reference' wt sequence.
    """

    #ret=np.array([np.nan]*len(cond_func_df))

    ret = []
    for idx, row in tqdm(
        condition_func_df.iterrows(), total=len(condition_func_df)
    ):

        #key = tuple(list(zip(row.wts, row.sites, row.muts)))
        #if key in variant_cache:
        #    df.loc[idx, "var_wrt_ref"] = variant_cache[key]
        #    cache_hits += 1
        #    continue

        var_map = cond_var_map.copy()
        for wt, site, mut in zip(row.wts, row.sites, row.muts):
            var_map.loc[site, 'cond'] = mut

        nis = (
            var_map.where(var_map["ref"] != var_map['cond'])
            .dropna()
            .astype(str)
        )
        muts = nis['ref'] + nis.index.astype(str) + nis['cond']
        ret.append(" ".join(muts.values))

    return ret


def convert_subs_wrt_ref_seq_b(
    non_identical_sites,
    wts, 
    sites, 
    muts
):
    """
    
    """

    nis = non_identical_sites.copy()
    for wt, site, mut in zip(wts, sites, muts):
        if site not in non_identical_sites.index.values:
            nis.loc[site] = wt, mut
        else:
            ref_wt = non_identical_sites.loc[site, 'ref']
            if mut != ref_wt:
                nis.loc[site] = ref_wt, mut
            else:
                nis.drop(site, inplace=True)
    
    converted_muts = nis['ref'] + nis.index.astype(str) + nis['cond']
    nis = (
        var_map.where(var_map["ref"] != var_map['cond'])
        .dropna()
        .astype(str)
    )
    return " ".join(converted_muts)




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
