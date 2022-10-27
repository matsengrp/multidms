r"""
==========
model
==========

Defines JIT compiled functions for to use for 
composing models and their respective objective functions.

For the sake of modularity, abstraction, and code-reusibility,
we would like to be able to modularize the 
individual pieces of code that define a model.
To this end, we are mainly constrained by the necessity of
Just-in-time (JIT) compilation of our model code for effecient 
training and parameter optimization.
To achieve this, we must take advantage of the
``jax.tree_util.Partial`` utility, which allows
for jit-compiled functions to be "pytree compatible".
In other words, by decorating functions with 
``Partial``, we can clearly mark which arguments will
in a function are themselves, statically compiled functions
in order to achieve function composition.

For a simple example using the ``Partial`` function,
see https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.Partial.html

Here, we do it slightly differently than the example given
by the documentation where they highlight the feeding 
of partial functions into jit-compiled functions as 
arguments that are pytree compatible.
Instead, we first use partial in the more traditional sense
such that _calling_ functions (i.e. functions that call on other 
parameterized functions) are defined as being a "partially" jit-compiled
function, until the relevant static arguments are provided using another
call to Partial. 

Consider the small example of a composed model formula f(t, g, x) = t(g(x))


>>> import jax.numpy as jnp
>>> import jax
>>> jax.config.update("jax_enable_x64", True)
>>> from jax.tree_util import Partial

>>> @jax.jit
>>> def identity(x):
>>>     return x

>>> @Partial(jax.jit, static_argnums=(0,1,))
>>> def f(t, g, x):
>>>     print(f"compiling")
>>>     return t(g(x))

Here, we defined a simple ``identity`` activation function that is fully
jit-compiled, as well as the partially jit compiled calling function, ``f``,
Where the 0th and 1st arguments have been marked as static arguments.
Next, we'll compile ``f`` by providing the static arguments using another
call to ``Partial``.

>>> identity_f = Partial(f, identity, identity)

Now, we can call upon the compiled function without any reference to our
static arguments.

>>> identity_f(5)
  compiling
  5

Note that upon the first call our model was JIT-compiled, 
but subsequent calls need not re-compile

>>> identity_f(7)
  7

We may also want to evaluate the loss of our simple model using MSE.
We can again use the partial function to define an abstract SSE loss
which is compatible with any pre-compiled model function, ``f``.

>>> @Partial(jax.jit, static_argnums=(0,))
>>> def sse(f, x, y):
>>>     print(f"compiling")
>>>     return jnp.sum((f(x) - y)**2)

And finally we provide some examples and targets to evauate the cost.

>>> compiled_cost = Partial(sse, identity_f)
>>> x = jnp.array([10, 12])
>>> y = jnp.array([11, 11])
>>> compiled_cost(x,y)
  compiling
  compiling
  2
"""


import numpy as np
import json
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxlib
from jax.experimental import sparse
from jax.tree_util import Partial
from frozendict import frozendict
import jaxopt
import numpy as onp

@jax.jit
def identity_activation(act, **kwargs):
    return act


@jax.jit
def softplus_activation(act, lower_bound=-3.5, hinge_scale=0.1, **kwargs):
    """A modified softplus that hinges at 'lower_bound'. 
    The rate of change at the hinge is defined by 'hinge_scale'.

    This is derived from 
    https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softplus.html
    """

    return (
        hinge_scale * (
            jnp.log(
                1 + jnp.exp((act - lower_bound) / hinge_scale)
            )
        ) + lower_bound
    )


@jax.jit
def gelu_activation(act, lower_bound=-3.5):
    """ A modified Gaussian error linear unit activation function,
    where the lower bound asymptote is defined by `lower_bound`.

    This is derived from 
    https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.gelu.html
    """
    sp = act - lower_bound
    return (
        (sp/2) * (
            1 + jnp.tanh(
                jnp.sqrt(2/jnp.pi) * (sp+(0.044715*(sp**3)))
            )
        ) + lower_bound
    )


@jax.jit
def ϕ(d_params:dict, X_h:jnp.array):
    """ Model for predicting latent space with
    shift parameters."""
   
    return (
            d_params["C_ref"]
            + d_params["C_d"] 
            + (X_h @ (d_params["β_m"] + d_params["s_md"])) 
           )

    
@jax.jit
def sigmoidal_global_epistasis(α:dict, z_h:jnp.array):
    """ A flexible sigmoid function for
    modeling global epistasis."""

    activations = jax.nn.sigmoid(z_h[:, None])
    return (α["ge_scale"] @ activations.T) + α["ge_bias"]


# ?? Should hugh's model include the t()? or should we
# separate the f, and t functions and make that explicit?
@Partial(jax.jit, static_argnums=(0, 1, 2,))
def abstract_epistasis(
    ϕ:jaxlib.xla_extension.CompiledFunction,
    g:jaxlib.xla_extension.CompiledFunction,
    t:jaxlib.xla_extension.CompiledFunction,
    d_params:dict, 
    X_h:jnp.array, 
    **kwargs
):
    """ Biophysical model - compiled for optimization 
    until model functions ϕ, g, and t are updated. 
    Model may be composed & compiled
    with the required functions fixed 
    using `jax.tree_util.Partial."""

    return t(g(d_params['α'], ϕ(d_params, X_h)), **kwargs)


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


@Partial(jax.jit, static_argnums=(0,))
def gamma_corrected_cost_smooth(f, params, data, δ=1, λ_ridge=0, **kwargs):
    """Cost (Objective) function summed across all conditions"""

    X, y = data
    loss = 0   
    
    # Sum the huber loss across all conditions
    for condition, X_d in X.items():   
        
        # Subset the params for condition-specific prediction
        d_params = {
            "α":params["α"],
            "β_m":params["β"], 
            "C_ref":params["C_ref"],
            "s_md":params[f"S_{condition}"], 
            "C_d":params[f"C_{condition}"],
            "γ_d":params[f"γ_{condition}"]
        }
       
        # compute predictions 
        y_d_predicted = f(d_params, X_d, **kwargs)
        
        # compute the Huber loss between observed and predicted
        # functional scores
        loss += jaxopt.loss.huber_loss(
            y[condition] + d_params[f"γ_d"], y_d_predicted, δ
        ).mean()
        
        # compute a regularization term that penalizes non-zero
        # shift parameters and add it to the loss function
        ridge_penalty = λ_ridge * (d_params["s_md"] ** 2).sum()
        loss += ridge_penalty

    return loss


"""
Vanilla multidms model with shift parameters, sigmoidal epistatic function,
and linear activation on the output.
"""

# OPTION 1
compiled_pred = Partial(
        abstract_epistasis, # abstract function to compile
        ϕ,
        sigmoidal_global_epistasis, 
        identity_activation
)
compiled_cost = Partial(
        gamma_corrected_cost_smooth, 
        compiled_pred
)
global_epistasis = frozendict({
    "predict" : compiled_pred,
    "objective" : compiled_cost,
    "proximal" : lasso_lock_prox 
})

#class MultidmsModel:
#    """
#    This class takes available (partially and fully)
#    jit compiled functions from the functions defined
#    in 
#    """
#    pass



