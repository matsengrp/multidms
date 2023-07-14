r"""
=================
biophysical_model
=================

Defines functions for
modeling global epistasis biophysical models
and their respective objective functions
in `multidms.MultiDmsModel` Objects.
"""

import jax.numpy as jnp
from jaxopt.loss import huber_loss
from jaxopt.prox import prox_lasso
import jax
jax.config.update("jax_enable_x64", True)


def identity_activation(d_params, act, **kwargs):
    """
    identity function :math:`f(x)=x`
    """
    return act


def softplus_activation(d_params, act, lower_bound=-3.5, hinge_scale=0.1, **kwargs):
    """
    A modified softplus that hinges at 'lower_bound'.
    The rate of change at the hinge is defined by 'hinge_scale'. This is derived from
    https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softplus.html
    """

    return (
        hinge_scale
        * (jnp.logaddexp(0, (act - (lower_bound + d_params["γ_d"])) / hinge_scale))
        + lower_bound
        + d_params["γ_d"]
    )


def additive_model(d_params: dict, X_h: jnp.array):
    """
    Model for predicting latent space with
    shift parameters.


    """

    return (
        d_params["C_ref"]
        + d_params["C_d"]
        + (X_h @ (d_params["β_m"] + d_params["s_md"]))
    )


def sigmoidal_global_epistasis(α: dict, z_h: jnp.array):
    """
    A flexible sigmoid function for
    modeling global epistasis.
    """

    activations = jax.nn.sigmoid(z_h[:, None])
    return (α["ge_scale"] @ activations.T) + α["ge_bias"]


def softplus_global_epistasis(α: dict, z_h: jnp.array):
    """
    A flexible sigmoid function for
    modeling global epistasis.
    """
    activations = jax.nn.softplus(-1 * z_h[:, None])
    return ((-1 * α["ge_scale"]) @ activations.T) + α["ge_bias"]


def nn_global_epistasis(α: dict, z_h: jnp.array):
    """
    A single-layer neural network for modeling global epistasis.
    """
    activations = jax.nn.sigmoid(α["p_weights_1"] * z_h[:, None] + α["p_biases"])
    return (α["p_weights_2"] @ activations.T) + α["output_bias"]


def abstract_from_latent(
    g,
    t,
    d_params: dict,
    z_h: jnp.array,
    **kwargs,
):
    """
    Biophysical model - compiled for optimization
    until model functions additive_model, g, and t are updated.
    Model may be composed & compiled
    with the required functions fixed
    using 'jax.tree_util.Partial.
    """
    return t(d_params, g(d_params["α"], z_h), **kwargs)


def abstract_epistasis(
    additive_model,
    g,
    t,
    d_params: dict,
    X_h: jnp.array,
    **kwargs,
):
    """
    Biophysical model - compiled for optimization
    until model functions additive_model, g, and t are updated.
    Model may be composed & compiled
    with the required functions fixed
    using 'jax.tree_util.Partial'.
    """
    return t(d_params, g(d_params["α"], additive_model(d_params, X_h)), **kwargs)


def lasso_lock_prox(
    params,
    hyperparams_prox=dict(
        lasso_params=None,
        lock_params=None,
    ),
    scaling=1.0,
):
    # enforce monotonic epistasis
    if "ge_scale" in params["α"]:
        params["α"]["ge_scale"] = params["α"]["ge_scale"].clip(0)

    if "p_weights_1" in params["α"]:
        params["α"]["p_weights_1"] = params["α"]["p_weights_1"].clip(0)
        params["α"]["p_weights_2"] = params["α"]["p_weights_2"].clip(0)
        # params["α"]["p_biases"] = params["α"]["p_biases"].clip(0)

    if hyperparams_prox["lasso_params"] is not None:
        for key, value in hyperparams_prox["lasso_params"].items():
            params[key] = prox_lasso(params[key], value, scaling)

    # Any params to constrain during fit
    if hyperparams_prox["lock_params"] is not None:
        for key, value in hyperparams_prox["lock_params"].items():
            params[key] = value

    return params


def gamma_corrected_cost_smooth(
    f,
    params,
    data,
    δ=1,
    λ_ridge_shift=0,
    λ_ridge_beta=0,
    λ_ridge_gamma=0,
    λ_ridge_cd=0,
    **kwargs,
):
    """Cost (Objective) function summed across all conditions"""

    X, y = data
    loss = 0

    # Sum the huber loss across all conditions
    # shift_ridge_penalty = 0
    for condition, X_d in X.items():
        # Subset the params for condition-specific prediction
        d_params = {
            "α": params["α"],
            "β_m": params["β"],
            "C_ref": params["C_ref"],
            "s_md": params[f"S_{condition}"],
            "C_d": params[f"C_{condition}"],
            "γ_d": params[f"γ_{condition}"],
        }

        # compute predictions
        y_d_predicted = f(d_params, X_d, **kwargs)

        # compute the Huber loss between observed and predicted
        # functional scores
        loss += huber_loss(y[condition] + d_params[f"γ_d"], y_d_predicted, δ).mean()

        # compute a regularization term that penalizes non-zero
        # parameters and add it to the loss function
        loss += λ_ridge_shift * jnp.sum(d_params["s_md"] ** 2)
        loss += λ_ridge_cd * jnp.sum(d_params["C_d"] ** 2)
        loss += λ_ridge_gamma * jnp.sum(d_params["γ_d"] ** 2)

    loss += λ_ridge_beta * jnp.sum(params["β"] ** 2)

    return loss
