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
#from tqdm.notebook import tqdm
from tqdm import tqdm
tqdm.pandas()


# from multidms.model import *

substitution_column = 'aa_substitutions_reference'
experiment_column = 'homolog_exp'
scaled_func_score_column = 'log2e'

   
# TODO should we assert there's no mutations like, A154bT? 
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
