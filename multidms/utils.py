"""
==========
Utils
==========

Helpful utility functions for the `multidms` package
"""

import scipy as sp
import jax
import jax.numpy as jnp


# TODO add
# scale_coeff_lasso_shift = lambda x:
# x.scale_coeff_lasso_shift.apply(lambda x: "{:.2e}".format(x))


def difference_matrix(n, ref_index=0):
    """
    Given some number of conditions, return the difference matrix
    for computing shifts between adjacent conditional beta parameters.
    This always assumes the reference condition is the first condition.
    """
    D = jnp.eye(n, n).at[:, ref_index].set(-1).at[ref_index].set(0)

    return D


# TODO test
def transform(params, bundle_idxs):
    """
    Transforms the beta coefficient parameters of a `multidms` model to be
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


# TODO test
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
