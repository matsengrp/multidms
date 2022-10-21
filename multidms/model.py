#!/usr/bin/env python

import numpy as np
import json
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxlib
from jax.experimental import sparse
from jax.tree_util import Partial as jax_partial
from functools import partial as functools_partial
from frozendict import frozendict
import jaxopt
import numpy as onp

# activation functions
@jax.jit
def identity_activation(x, **kwargs):
    return x


@jax.jit
def softplus_activation(act, lower_bound=-3.5, hinge_scale=0.1, **kwargs):
    """A modified softplus that hinges at 'lower_bound'. 
    the rate of change at the hinge is defined by 'hinge_scale'."""

    return (
        hinge_scale * (
            jnp.log(
                1 + jnp.exp((act - lower_bound) / hinge_scale)
            )
        ) + lower_bound
    )


@jax.jit
def shifted_gelu(x, l=-3.5):
    sp = x - l 
    return (
        (sp/2) * (
            1 + jnp.tanh(
                jnp.sqrt(2/jnp.pi) * (sp+(0.044715*(sp**3)))
            )
        ) + l
    )


# latent space predictions
@jax.jit
def ϕ_shift(h_params:dict, X_h:jnp.array):
    """ Model for predicting latent space """
    
    return ((X_h @ (h_params["β"] + h_params["S"])) 
            + h_params["C"] 
            + h_params["C_ref"]
           )


# global epistasis models
@jax.jit
def sigmoidal_global_epistasis(α:dict, z_h:jnp.array):

    """ Model for global epistasis as 'flexible' sigmoid. """

    activations = jax.nn.sigmoid(z_h[:, None])
    return (α["ge_scale"] @ activations.T) + α["ge_bias"]


# JAX Engine
@functools_partial(jax.jit, static_argnums=(0, 1, 2,))
def abstract_epistasis(
    ϕ:jaxlib.xla_extension.CompiledFunction,
    g:jaxlib.xla_extension.CompiledFunction,
    t:jaxlib.xla_extension.CompiledFunction,
    h_params:dict, 
    X_h:jnp.array, 
    **kwargs
):
    """ Biophysical model - compiled for optimization 
    until model functions ϕ, g, and t are updated. 
    Model may be composed & compiled
    with the required functions fixed 
    using `jax.tree_util.Partial."""

    return t(g(h_params['α'], ϕ(h_params, X_h)), **kwargs)


@jax.jit
def lasso_lock_prox(
    params, 
    hyperparams_prox=dict(
        lasso_params=None, 
        lock_params=None,
    ), 
    scaling=1.0
):
    
    # Monotonic non-linearity, if non-linear model
    if "α" in params:
        params["α"]["ge_scale"] = params["α"]["ge_scale"].clip(0)
    
    if hyperparams_prox["lasso_params"] is not None:
        for key, value in hyperparams_prox["lasso_params"].items():
            params[key] = jaxopt.prox.prox_lasso(params[key], value, scaling)

    # Any params to constrain during fit
    if hyperparams_prox["lock_params"] is not None:
        for key, value in hyperparams_prox["lock_params"].items():
            params[key] = value

    return params


@functools_partial(jax.jit, static_argnums=(0,))
def gamma_corrected_cost_smooth(f, params, data, δ=1, λ_ridge=0, **kwargs):
    """Cost (Objective) function summed across all homologs"""

    X, y = data
    loss = 0   
    
    # Sum the huber loss across all homologs
    for homolog, X_h in X.items():   
        
        # Subset the params for homolog-specific prediction
        h_params = {
            "α":params["α"],
            "β":params["β"], 
            "C_ref":params["C_ref"],
            "S":params[f"S_{homolog}"], 
            "C":params[f"C_{homolog}"],
        }
       
        # compute predictions 
        y_h_predicted = f(h_params, X_h, **kwargs)
        
        # compute the Huber loss between observed and predicted
        # functional scores
        loss += jaxopt.loss.huber_loss(
            y[homolog] + params[f"γ_{homolog}"], y_h_predicted, δ
        ).mean()
        
        # compute a regularization term that penalizes non-zero
        # shift parameters and add it to the loss function
        ridge_penalty = λ_ridge * (params[f"S_{homolog}"] ** 2).sum()
        loss += ridge_penalty

    return loss


"""
Vanilla multidms model with shift parameters, sigmoidal epistatic function,
and linear activation on the output.
"""
compiled_pred = jax_partial(
        abstract_epistasis, # abstract function to compile
        ϕ_shift,
        sigmoidal_global_epistasis, 
        identity_activation
)
compiled_cost = jax_partial(
        gamma_corrected_cost_smooth, 
        compiled_pred
)
global_epistasis = frozendict({
    "predict" : compiled_pred,
    "objective" : compiled_cost,
    "proximal" : lasso_lock_prox 
})
