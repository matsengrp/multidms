r"""
===========
biophysical
===========

Defines functions for jointly
modeling global epistasis biophysical models -
as well as the ("private") objective functions required 
for parameter optimization in :mod:`multidms.model`.

`multidms.Model` Objects are defined with parameters
which take references to functions (such as the ones defined here) as arguments.
The object then handles the: 

 - parameter initialization/bookeeping 
 - composition of the provided functions using `functools.partial`
 - the subsequent jit-compilation, using `jax.jit`, on the composed model and objective 
   functions 
 - as well as the optimizations of the the model parameters.

This allows for the user/developer to define their own model components
for each of the latent phenotype, global epistasis, and output activation functions.
This module is kept separate from the :mod:`multidms.model`
primarily for the sake of readability and documentation.
This may change in the future.

.. note::
    In order to make use of the `jax` library,
    all components must be written in a functional style.
    This means that all functions must be written
    to take in all parameters as arguments
    and return a value with no side effects.
    See the `jax` documentation for more details.
"""

import jax.numpy as jnp
from jaxopt.loss import huber_loss
from jaxopt.prox import prox_lasso
import jax

jax.config.update("jax_enable_x64", True)


r"""
+++++++++++++++++++++++++++++
Latent phenotype joint models
+++++++++++++++++++++++++++++
"""


def additive_model(d_params: dict, X_d: jnp.array):
    r"""
    Model for predicting latent phenotype of a set
    of binary encoded variants :math:`v` for a given condition, :math:`d`
    and the corresponding
    beta ( :math:`\beta` ),
    shift ( :math:`\Delta_{d, m}` ),
    and latent offset ( :math:`\beta_0, \alpha_d` )
    parameters.

    .. math::
       \phi_d(v) = \beta_0 + \alpha_d + \sum_{m \in v} (\beta_{m} + \Delta_{d, m})

    Parameters
    ----------
    d_params : dict
        Dictionary of model defining parameters as jax arrays.
        note that shape of the parameters must be compatible with the
        input data.
    X_d : array-like
        Binary encoded mutations for a given set of variants
        from condition, :math:`d`.

    Returns
    -------
    jnp.array
        Predicted latent phenotypes for each row in ``X_d``
    """
    return (
        d_params["beta_naught"]
        + d_params["alpha_d"]
        + (X_d @ (d_params["beta_m"] + d_params["s_md"]))
    )


r"""
+++++++++++++++++++++++
Global epistasis models
+++++++++++++++++++++++

.. note::
    All global epistasis models take in a dictionary of parameters
"""


def sigmoidal_global_epistasis(theta: dict, z_d: jnp.array):
    r"""
    A flexible sigmoid function for modeling global epistasis.
    This function takes a set of latent phenotype values, :math:`z_d`
    and computes the predicted functional scores using the scaling parameters
    :math:`\theta_{\text{scale}}` and :math:`\theta_{\text{bias}}`
    such that:

    .. math::
        g(z) =  \frac{\theta_{\text{scale}}}{1 + e^{-z}} + \theta_{\text{bias}}

    .. note::
        this function is independent from the
        experimental condition from which a variant is observed.

    Parameters
    ----------
    theta : dict
        Dictionary of model defining parameters as jax arrays.
    z_d : jnp.array
        Latent phenotype values for a given set of variants

    Returns
    -------
    jnp.array
        Predicted functional scores for each latent phenotype in ``z_d``.
    """
    activations = jax.nn.sigmoid(z_d[:, None])
    return (theta["ge_scale"] @ activations.T) + theta["ge_bias"]


def softplus_global_epistasis(theta: dict, z_d: jnp.array):
    r"""
    A flexible softplus function for
    modeling global epistasis.
    This function takes a set of latent phenotype values, :math:`z_d`
    and computes the predicted functional scores such that

    .. math::
        g(z) =  -\theta_\text{scale}\log\left(1+e^{-z}\right) + \theta_\text{bias}

    .. note::
        This function has no natural lower bound, thus, it is recommended you use this
        model in conjuction with an output activation such as :func:`softplus_activation`


    Parameters
    ----------
    theta : dict
        Dictionary of model defining parameters as jax arrays.
    z_d : jnp.array
        Latent phenotype values for a given set of variants

    Returns
    -------
    jnp.array
        Predicted functional scores for each latent phenotype in ``z_d``.
    """
    activations = jax.nn.softplus(-1 * z_d[:, None])
    return ((-1 * theta["ge_scale"]) @ activations.T) + theta["ge_bias"]


def nn_global_epistasis(theta: dict, z_d: jnp.array):
    r"""
    A single-layer neural network for modeling global epistasis.
    This function takes a set of latent phenotype values, :math:`z_d`
    and computes the predicted functional scores.

    For this option, the user defines a number of units in the
    singular hidden layer of the model. For each hidden unit,
    we introduce three parameters (two weights and a bias) to be inferred.
    All weights are clipped at zero to maintain assumptions of
    monotonicity in the resulting epistasis function shape.
    The network applies a sigmoid activation to each internal unit
    before a final transformation and addition of a constant
    gives us our predicted functional score.

    More concretely, given latent phenotype, :math:`\phi_d(v) = z`, let

    .. math::
        g(z) = b^{o}+ \sum_{i}^{n} \frac{w^{o}_{i}}{1 + e^{w^{l}_{i}*z + b^{l}_{i}}}

    Where:

    - :math:`n` is the number of units in the hidden layer.
    - :math:`w^{l}_{i}` and :math:`w^{o}_{i}` are free parameters representing latent
      and output tranformations, respectively, associated with unit `i` in the
      hidden layer of the network.
    - :math:`b^{l}_{i}` is a free parameter, as an added bias term to unit `i`.
    - :math:`b^{o}` is a constant, singular free parameter.

    .. Note::
        This is an advanced feature and we advise against its use
        unless the other options are not sufficiently parameterized for particularly
        complex experimental conditions.

    Parameters
    ----------
    theta : dict
        Dictionary of model defining parameters as jax arrays.
    z_d : jnp.array
        Latent phenotype values for a given set of variants

    Returns
    -------
    jnp.array
        Predicted functional scores for each latent phenotype in ``z_d``.
    """
    activations = jax.nn.sigmoid(
        theta["p_weights_1"] * z_d[:, None] + theta["p_biases"]
    )
    return (theta["p_weights_2"] @ activations.T) + theta["output_bias"]


r"""
++++++++++++++++++++
Activation functions
++++++++++++++++++++
"""


def identity_activation(d_params, act, **kwargs):
    """
    Identity function :math:`f(x)=x`.
    Mostly a ghost function which helps compose
    the model when you don't want to use any final output activation e.g.
    you don't have a pre-defined lower bound.
    """
    return act


def softplus_activation(d_params, act, lower_bound=-3.5, hinge_scale=0.1, **kwargs):
    r"""
    A modified softplus that hinges at given lower bound.
    The rate of change at the hinge is defined by 'hinge_scale'.

    In essence, this is a modified _softplus_ activation,
    (:math:`\text{softplus}(x)=\log(1 + e^{x})`)
    with a lower bound at :math:`l + \gamma_{h}`,
    as well as a ramping coefficient, :math:`\lambda_{\text{sp}}`.

    Concretely, if we let :math:`z' = g(\phi_d(v))`, then the predicted functional score
    of our model is given by:

    .. math::
        t(z') = \lambda_{sp}\log(1 + e^{\frac{z' - l}{\lambda_{sp}}}) + l

    Functionally speaking, this truncates scores below a lower bound,
    while leaving scores above (mostly) unaltered.
    There is a small range of input values where the function smoothly
    transitions between a flat regime (where data is truncated)
    and a linear regime (where data is not truncated).

    .. note:
        We recommend leaving the :math:`\lambda_{sp}` parameter at
        it's default value of :math:`0.1`.
        this ensures a sharp transition between regimes similar to a
        [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
        function, but retain the differentiable property for gradient based optimization.
        However, the option is there in case you find the model will not converge.

    .. note::
        This is derived from
        https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softplus.html

    Parameters
    ----------
    d_params : dict
        Dictionary of model defining parameters as jax arrays.
    act : jnp.array
        Activations to apply the softplus function to.
    lower_bound : float
        Lower bound to hinge the softplus function at.
    hinge_scale : float
        Rate of change at the hinge point.
    kwargs : dict
        Additional keyword arguments to pass to the biophysical model function


    Returns
    -------
    jnp.array
        Transformed activations.
    """
    return (
        hinge_scale
        * (jnp.logaddexp(0, (act - (lower_bound + d_params["gamma_d"])) / hinge_scale))
        + lower_bound
        + d_params["gamma_d"]
    )


def _abstract_from_latent(
    g,
    t,
    d_params: dict,
    z_h: jnp.array,
    **kwargs,
):
    r"""
    Take in two compatible functions for global epistasis
    and output activation and compose them into a single
    function that takes in latent space and outputs a prediction.
    """
    return t(d_params, g(d_params["theta"], z_h), **kwargs)


def _abstract_epistasis(
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

    See the biophysical docs for more details.
    """
    return t(d_params, g(d_params["theta"], additive_model(d_params, X_h)), **kwargs)


def _lasso_lock_prox(
    params,
    hyperparams_prox=dict(
        lasso_params=None,
        lock_params=None,
    ),
    scaling=1.0,
):
    """
    Apply lasso and lock constraints to parameters

    Parameters
    ----------
    params : dict
        Dictionary of parameters to constrain
    hyperparams_prox : dict
        Dictionary of hyperparameters for proximal operators
    scaling : float
        Scaling factor for lasso penalty
    """
    # enforce monotonic epistasis
    if "ge_scale" in params["theta"]:
        params["theta"]["ge_scale"] = params["theta"]["ge_scale"].clip(0)

    if "p_weights_1" in params["theta"]:
        params["theta"]["p_weights_1"] = params["theta"]["p_weights_1"].clip(0)
        params["theta"]["p_weights_2"] = params["theta"]["p_weights_2"].clip(0)

    if hyperparams_prox["lasso_params"] is not None:
        for key, value in hyperparams_prox["lasso_params"].items():
            params[key] = prox_lasso(params[key], value, scaling)

    # Any params to constrain during fit
    if hyperparams_prox["lock_params"] is not None:
        for key, value in hyperparams_prox["lock_params"].items():
            params[key] = value

    return params


def _gamma_corrected_cost_smooth(
    f,
    params,
    data,
    huber_scale=1,
    scale_coeff_ridge_shift=0,
    scale_coeff_ridge_beta=0,
    scale_coeff_ridge_gamma=0,
    scale_coeff_ridge_alpha_d=0,
    **kwargs,
):
    """
    Cost (Objective) function summed across all conditions

    Parameters
    ----------
    f : function
        Biophysical model function
    params : dict
        Dictionary of parameters to optimize
    data : tuple
        Tuple of (X, y) data where each are dictionaries keyed by condition,
        return the respective binarymap and the row associated target functional scores
    huber_scale : float
        Scale parameter for Huber loss function
    scale_coeff_ridge_shift : float
        Ridge penalty coefficient for shift parameters
    scale_coeff_ridge_beta : float
        Ridge penalty coefficient for beta parameters
    scale_coeff_ridge_gamma : float
        Ridge penalty coefficient for gamma parameters
    scale_coeff_ridge_alpha_d : float
        Ridge penalty coefficient for alpha parameters
    kwargs : dict
        Additional keyword arguments to pass to the biophysical model function

    Returns
    -------
    loss : float
        Summed loss across all conditions.
    """
    X, y = data
    loss = 0

    # Sum the huber loss across all conditions
    # shift_ridge_penalty = 0
    for condition, X_d in X.items():
        # Subset the params for condition-specific prediction
        d_params = {
            "theta": params["theta"],
            "beta_m": params["beta"],
            "beta_naught": params["beta_naught"],
            "s_md": params[f"shift_{condition}"],
            "alpha_d": params[f"alpha_{condition}"],
            "gamma_d": params[f"gamma_{condition}"],
        }

        # compute predictions
        y_d_predicted = f(d_params, X_d, **kwargs)

        # compute the Huber loss between observed and predicted
        # functional scores
        loss += huber_loss(
            y[condition] + d_params["gamma_d"], y_d_predicted, huber_scale
        ).mean()

        # compute a regularization term that penalizes non-zero
        # parameters and add it to the loss function
        loss += scale_coeff_ridge_shift * jnp.sum(d_params["s_md"] ** 2)
        loss += scale_coeff_ridge_alpha_d * jnp.sum(d_params["alpha_d"] ** 2)
        loss += scale_coeff_ridge_gamma * jnp.sum(d_params["gamma_d"] ** 2)

    loss += scale_coeff_ridge_beta * jnp.sum(params["beta"] ** 2)

    return loss
