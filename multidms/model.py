#!/usr/bin/env python
"""
coding: utf-8

# multidms

## Overview of model

The `multidms` model applies to a case where you have DMS datasets for two or more homologs and are interested in identifying shifts in mutational effects between homologs.
To do so, the model defines one homolog as a reference homolog.
For each mutation, the model fits one parameter that quantifies the effect of the mutation in the reference homolog.
For each non-reference homolog, it also fits a shift parameter that quantifies the shift in the mutation's effect in the non-reference homolog relative to the reference.
Shift parameters can be regularized, encouraging most of them to be close to zero.
This regularization step is a useful way to eliminate the effects of experimental noise, and is most useful in cases where you expect most mutations to have the same effects between homologs, such as for homologs that are close relatives.

The model uses a global-epistasis function to disentangle the effects of multiple mutations on the same variant.
To do so, it assumes that mutational effects additively influence a latent biophysical property the protein (e.g., $\Delta G$ of folding).
The mutational-effect parameters described above operate at this latent level.
The global-epistasis function then assumes a sigmoidal relationship between a protein's latent property and its functional score measured in the experiment (e.g., log enrichment score).
Ultimately, mutational parameters, as well as ones controlling the shape of the sigmoid, are all jointly fit to maximize agreement between predicted and observed functional scores acorss all variants of all homologs.

## Detailed description of the model

For each variant $v$ from homolog $h$, we use a global-epistasis function $g$ to convert a latent phenotype $\phi$ to a functional score $f$:

$$f(v,h) = g_{\alpha}(\phi(v,h)) + γ_h$$

where $g$ is a sigmoid and $\alpha$ is a set of parameters,
`ge_scale`, and `ge_bias` which define the shape of the sigmoid.

The latent phenotype is computed in the following way:

$$\phi(v,h) = c + \sum_{m \in v} (x_m + s_{m,h})$$

where:
* $c$ is the wildtype latent phenotype for the reference homolog.
* $x_m$ is the latent phenotypic effect of mutation $m$. See details below.
* $s_{m,h}$ is the shift of the effect of mutation $m$ in homolog $h$. These parameters are fixed to zero for the reference homolog. For non-reference homologs, they are defined in the same way as $x_m$ parameters.
* $v$ is the set of all mutations relative to the reference wildtype sequence (including all mutations that separate homolog $h$ from the reference homolog).

The $x_m$ variable is defined such that mutations are always relative to the reference homolog.
For example, if the wildtype amino acid at site 30 is an A in the reference homolog, and a G in a non-reference homolog, then a Y30G mutation in the non-reference homolog is recorded as an A30G mutation relative to the reference.
This way, each homolog informs the exact same parameters, even at sites that differ in wildtype amino acid.
These are encoded in a `BinaryMap` object, where all sites that are non-identical to the reference are 1's.

Ultimately, we fit parameters using a loss function with one term that scores differences between predicted and observed values and another that uses L1 regularization to penalize non-zero $s_{m,h}$ values:

$$ L_{\text{total}} = \sum_{h} \left[\sum_{v} L_{\text{fit}}(y_{v,h}, f(v,h)) + \lambda \sum_{m} |s_{m,h}|\right]$$

where:
* $L_{\text{total}}$ is the total loss function.
* $L_{\text{fit}}$ is a loss function that penalizes differences in predicted vs. observed functional scores.
* $y_{v,h}$ is the experimentally measured functional score of variant $v$ from homolog $h$.

## Model using matrix algebra

We compute a vector or predicted latent phenotypes $P_{h}$ as:

$$P_{h} = c + (X_h \cdot (β + S_h))$$

where:
* $β$ is a vector of all $β_m$ values.
* $S_h$ is a matrix of all $s_{m,h}$ values.
* $X_h$ is a sparse matrix, where rows are variants, columns are mutations (all defined relative to the reference homolog), and values are weights of 0's and 1's. These weights are used to compute the phenotype of each variant given the mutations present.
* $c$ is the same as above.

In the matrix algebra, the sum of $β_m$ and $S_{m,h}$ gives a vector of mutational effects, with one entry per mutation.
Multiplying the matrix $X_h$ by this vector gives a new vector with one entry per variant, where values are the sum of mutational effects, weighted by the variant-specific weights in $X_h$.
Adding the $c$ value to this vector will give a vector of predicted latent phenotypes for each variant.

Next, the global-epistasis function can be used to convert a vector of predicted latent phenotypes to a vector of predicted functional scores.

$$F_{h,pred} = g_{\alpha}(P_h)$$

Finally, this vector could be fed into a loss function and compared with a vector of observed functional scores.
"""


import numpy as np
import json
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental import sparse
import jaxopt
import numpy as onp


@jax.jit
def ϕ(h_params:dict, X_h:jnp.array):
    """ Model for predicting latent space """
    
    return ((X_h @ (h_params["β"] + h_params["S"])) 
            + h_params["C"] 
            + h_params["C_ref"]
           )


@jax.jit
def g(α:dict, z_h:jnp.array):
    """ Model for global epistasis as 'flexible' sigmoid. """

    activations = jax.nn.sigmoid(z_h[:, None])
    return (α["ge_scale"] @ activations.T) + α["ge_bias"]


@jax.jit
def scaled_shifted_softplus(act, lower_bound=-1.0, hinge_scale=0.1):
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
def f(h_params:dict, X_h:jnp.array, **kwargs):
    """ TODO """

    return scaled_shifted_softplus(
        g(h_params["α"], ϕ(h_params, X_h)), # + h_params["γ"],
        **kwargs
    )


@jax.jit
def prox(
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


@jax.jit
def cost_smooth(params, data, δ=1, λ_ridge=0, **kwargs):
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


@jax.jit
def f_linear(h_params:dict, X_h:jnp.array, **kwargs):
    """ TODO """

    return scaled_shifted_softplus(
        ϕ(h_params, X_h) + h_params["γ"],
        **kwargs
    )

@jax.jit
def prox_linear(
    params, 
    hyperparams_prox=dict(
        lasso_params=None, 
        lock_params=None
    ), 
    scaling=1.0
):
    
    if hyperparams_prox["lasso_params"] is not None:
        for key, value in hyperparams_prox["lasso_params"].items():
            params[key] = jaxopt.prox.prox_lasso(params[key], value, scaling)

    # Any params to constrain during fit
    if hyperparams_prox["lock_params"] is not None:
        for key, value in hyperparams_prox["lock_params"].items():
            params[key] = value

    return params


@jax.jit
def cost_smooth_linear(params, data, δ=1, λ_ridge=0):
    """Cost (Objective) function summed across all homologs"""

    X, y = data
    loss = 0   
    
    # Sum the huber loss across all homologs
    for homolog, X_h in X.items():   
        
        # Subset the params for homolog-specific prediction
        h_params = {
            "β":params["β"], 
            "C_ref":params["C_ref"],
            "S":params[f"S_{homolog}"], 
            "C":params[f"C_{homolog}"],
            "γ":params[f"γ_{homolog}"]
        }
        
        y_h_predicted = f_linear(h_params, X_h)
        
        # compute the Huber loss between observed and predicted
        # functional scores
        loss += jaxopt.loss.huber_loss(y[homolog], y_h_predicted, δ).mean()
        
        # compute a regularization term that penalizes non-zero
        # shift parameters and add it to the loss function
        ridge_penalty = λ_ridge * (params[f"S_{homolog}"] ** 2).sum()
        loss += ridge_penalty

    return loss
