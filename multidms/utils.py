"""
==========
Utils
==========

Helpful utility functions for the `multidms` package
"""

import scipy as sp
import jax
import jax.numpy as jnp


def difference_matrix(n):
    """
    Given some number of conditions, return the difference matrix
    for computing shifts between adjacent conditional beta parameters.
    This always assumes the reference condition is the first condition.
    """
    D = jnp.eye(n, n).at[:, 0].set(-1).at[0].set(0)

    return D


# TODO Finish, do shift params need to be transformed, as well?
# TODO test
def transform(params, bundle_idxs):
    """
    Transforms the bet coefficient parameters of a `multidms` model to be
    negative for the bundles specified in `bundle_idxs`.

    TODO Finish
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


def rereference(X, cols):
    """Flip bits on columns (bool idxs)"""
    if cols.sum():
        X_scipy = sp.sparse.csr_matrix(
            (X.data, (X.indices[:, 0], X.indices[:, 1])), shape=X.shape
        ).tolil()
        tmp = X_scipy[:, cols].copy()
        X_scipy[:, cols] = 1
        X_scipy[:, cols] -= tmp
        X_scaled = jax.experimental.sparse.BCOO.from_scipy_sparse(X_scipy)
        X_scaled = jax.experimental.sparse.BCOO(
            (X_scaled.data.astype(jnp.int8), X_scaled.indices), shape=X.shape
        )

        assert (
            X[:, cols].sum(0).todense() + X_scaled[:, cols].sum(0).todense()
            == X.shape[0]
        ).all()

    else:
        X_scaled = X

    return X_scaled
