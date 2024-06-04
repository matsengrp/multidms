"""
==========
Utils
==========

Helpful utility functions for the `multidms` package
"""

import scipy as sp
import pandas as pd
import jax
import jax.numpy as jnp
import re
import itertools as it


def explode_params_dict(params_dict):
    """
    Given a dictionary of model parameters,
    of which any of the values can be a list of values,
    compute all combinations of model parameter sets
    and returns a list of dictionaries representing each
    of the parameter sets.
    """
    varNames = sorted(params_dict)
    return [
        dict(zip(varNames, prod))
        for prod in it.product(*(params_dict[varName] for varName in varNames))
    ]


def my_concat(dfs_list, axis=0):
    """
    simple pd.concat wrapper for bug with vscode pylance
    See https://github.com/matsengrp/multidms/issues/156 for more details
    """  # noqa
    return pd.concat(dfs_list, axis=axis)


def split_sub(sub_string):
    """
    A very simplistic function to split a mutations string into its constituent parts.
    This function is only designed to work with simple mutation strings of the form
    `A123B`, where `A` is the wildtype amino acid, `123` is the site, and `B` is the
    mutant amino acid.

    .. note::
        It is favorable to use the `polyclonal.utils.MutationParser` class instead,
        as that provides the ability to parse more complex mutation strings using allowed
        alphabets.

    Parameters
    ----------
    sub_string : str
        A string containing a single mutation

    Returns
    -------
    tuple
        A tuple containing the wildtype amino acid, site, and mutant amino acid
    """
    pattern = r"(?P<aawt>[A-Z])(?P<site>[\d\w]+)(?P<aamut>[A-Z\*])"
    match = re.search(pattern, sub_string)
    assert match is not None, sub_string
    return match.group("aawt"), str(match.group("site")), match.group("aamut")


def split_subs(subs_string, parser=split_sub):
    """
    Given a mutation parsing function, split a string of mutations
    into three lists containing the wildtype amino acids, sites, and
    mutant amino acids.

    Parameters
    ----------
    subs_string : str
        A string containing multiple mutations

    parser : function
        A function that can parse a single mutation string and returns a tuple
        containing the wildtype amino acid, site, and mutant amino acid

    Returns
    -------
    tuple
        A tuple containing the wildtype amino acids, sites, and mutant amino acids
        as lists
    """
    wts, sites, muts = [], [], []
    for sub in subs_string.split():
        wt, site, mut = parser(sub)
        wts.append(wt)
        sites.append(site)
        muts.append(mut)
    return wts, sites, muts


def difference_matrix(n, ref_index=0):
    """
    Given some number of conditions, return the difference matrix
    for computing shifts between adjacent conditional beta parameters.

    Parameters
    ----------
    n : int
        The number of conditions
    ref_index : int
        The index of the reference condition

    Returns
    -------
    jnp.ndarray
        A difference matrix for computing shifts between
        adjacent conditional beta parameters
    """
    D = jnp.eye(n, n).at[:, ref_index].set(-1).at[ref_index].set(0)

    return D


def transform(params, bundle_idxs):
    """
    Transforms the beta coefficient parameters of a `multidms` model to be
    negative for the bundles specified in `bundle_idxs`, and the updated
    beta0 parameters based on the new beta coefficients.
    See `issue #156 <https://github.com/matsengrp/multidms/issues/156>`_ for more
    on scaling parameters for training.

    Parameters
    ----------
    params : dict
        A dictionary containing the model parameters
        "beta", and "beta0".
    bundle_idxs : dict
        A dictionary, keyed by condition
        containing the bundle indices in the binarymap matrix.

    Returns
    -------
    dict
        A dictionary containing the transformed model parameters
        "beta", and "beta0".
    """
    params_transformed = {
        key: val.copy() for key, val in params.items() if key not in ["beta", "beta0"]
    }
    params_transformed["beta"] = {}
    params_transformed["beta0"] = {}
    for d in params["beta"]:
        params_transformed["beta"][d] = params["beta"][d].at[bundle_idxs[d]].mul(-1)
        params_transformed["beta0"][d] = (
            params["beta0"][d] + params["beta"][d][bundle_idxs[d]].sum()
        )
    return params_transformed


def rereference(X, bundle_idxs):
    """
    Given a binary matrix X and bundle indices, re-reference the matrix
    to flip the bit signs for the bundles specified in `bundle_idxs`.
    This function is used to scale the data matrix for training a model.
    See `issue #156 <https://github.com/matsengrp/multidms/issues/156>`_ for more
    on scaling parameters for training.

    Parameters
    ----------
    X : jax.experimental.sparse.BCOO
        A binary matrix
    bundle_idxs : jnp.ndarray
        An boolean array indicating the bundle indices in the binarymap matrix.

    Returns
    -------
    jax.experimental.sparse.BCOO
        A re-referenced binary matrix
    """
    if bundle_idxs.sum():
        X_scipy = sp.sparse.csr_matrix(
            (X.data, (X.indices[:, 0], X.indices[:, 1])), shape=X.shape
        ).tolil()
        tmp = X_scipy[:, bundle_idxs].copy()
        X_scipy[:, bundle_idxs] = 1
        X_scipy[:, bundle_idxs] -= tmp
        X_scaled = jax.experimental.sparse.BCOO.from_scipy_sparse(X_scipy)
        X_scaled = jax.experimental.sparse.BCOO(
            (X_scaled.data.astype(jnp.int8), X_scaled.indices), shape=X.shape
        )

        assert (
            X[:, bundle_idxs].sum(0).todense()
            + X_scaled[:, bundle_idxs].sum(0).todense()
            == X.shape[0]
        ).all()

    else:
        X_scaled = X

    return X_scaled
