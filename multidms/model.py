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

import multidms

@jax.jit
def identity_activation(d_params, act, **kwargs):
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

"""
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
"""

"""
latent models must take in a binarymap
representation of variants, and predict
a single value representing the latent
phenotype prediction.
"""
latent_models = {
    "phi" : ϕ
}

"""
epistatic and output activations have the same shape
input and output, however, epistatic models
are parameterized by "α", which can differ depending
on the function chosen. For example, the identity
function requires no alpha parameters, sigmoid requires 2,
and perceptron may have up to n hidden nodes and layers.
"""
epistatic_models = {
    "sigmoid" : sigmoidal_global_epistasis,
    "identity" : identity_activation 
}
output_activations = {
    "softplus" : softplus_activation,
    "gelu" : gelu_activation
}


class MultiDmsModel:
    """
    A class for compiling, and fitting 
    multidms models and objective functions
    as a composition of (fully and partially) 
    jit-compiled functions.
    """

    # TODO, should the user provide available strings?
    def __init__(
        self,
        data:multidms.MultiDmsData,
        latent_model="phi",
        epistatic_model="identity",
        output_activation="identity",
        # objective_function=,
        # proximal_function=,
        init_sig_range=10.,
        init_sig_min=-10.,
        PRNGKey = 0
    ):
        """
        initialize a set of starting parameters for the JAX model.
        
        Parameters
        ----------
        
        conditions : list
            A list containing all possible target condition 
            names.
        
        n_beta_shift_params: int
            The number of beta and shift parameters 
            (for each condition) to initialize.
        
        include_alpha : book
            Initialize parameters for the sigmoid from
            the global epistasis function
        
        init_sig_range : float
            The range of observed phenotypes in the raw
            data, used to initialize the range of the
            sigmoid from the global epistasis function
        
        init_sig_min : float
            The lower bound of observed phenotypes in
            the raw data, used to initialize the minimum
            value of the sigmoid from the global epistasis
            funciton
            
        latent_bias : float
            bias parameter applied to the output of the
            linear model (latent prediction).
        
        Returns
        -------
        
        dict :
            all relevant parameters to be tuned with the JAX model.
        """
        
        # DATA
        # store all the relevant data as a reference to the
        # original MultiDmsData object

        # when we return the mutations_df or data_to_fit
        # propertied from this model object, we'll return
        # copies of the relevant snippets of data plus
        # the attributes defined by a model fit within this object.
        self.data = data
        
        # PARAMS
        # we now initialize parameters based upon models
        # chosen. After this, we will simply 
        self.params = {}
        key = jax.random.PRNGKey(seed)

        # initialize beta parameters from normal distribution.

        # TODO do we even have another model?
        # we could always lock shifts to zero in prox function
        if latent_model == "phi":

            n_beta_shift_parameters = len(self.data.all_subs)
            self.params["β"] = jax.random.normal(shape=(n_beta_shift_self.params,), key=key)

            # initialize shift parameters
            for condition in data.conditions:
                # We expect most shift parameters to be close to zero
                self.params[f"S_{condition}"] = jnp.zeros(shape=(n_beta_shift_self.params,))
                self.params[f"C_{condition}"] = jnp.zeros(shape=(1,))

            # all mode
            self.params["C_ref"] = jnp.array([5.0]) # 5.0 is a guess, could update

        if gamma_corrected:
            self.params[f"γ_{condition}"] = jnp.zeros(shape=(1,))

        # TODO softplus, perceptron, ispline
        if epistatic_model = "identity":
            continue
        elif epistatic_model = "sigmoid":

            self.params["α"]=dict(
                ge_scale=jnp.array([init_sig_range]),
                ge_bias=jnp.array([init_sig_min])
            )
        else:
            raise ValueError(f"{epistatic_model} not recognized},"
                "please use one from: {epistatic_models.values()}"
                )

        # COMPILED MODEL
        # Here is where we compile the main features of the model.
        # you could imagine 'abstract epistasis' could also be
        # a parameter and thus this class could be abstracted to
        # multiple biophysical models.
        compiled_pred = Partial(
                abstract_epistasis, # abstract function to compile
                latent_models[latent_model],
                epistatic_models[epistatic_model], 
                output_activations[output_activation]
        )
        compiled_cost = Partial(
                gamma_corrected_cost_smooth, 
                compiled_pred
        )
        self.model = frozendict({
            "ϕ" : latent_mode
            "f" : compiled_pred,
            "objective" : compiled_cost,
            "proximal" : lasso_lock_prox 
        })


    @property
    def data_to_fit(self):
        """ Get all functional score attributes from self.data
        updated with all model predictions"""
        # HERE
        # TODO How about fit_data? bc it may have already been fit.
        data_to_fit = copy.copy(self.data._data_to_fit)

        data_to_fit["predicted_latent"] = onp.nan
        data_to_fit[f"predicted_func_score"] = onp.nan
        data_to_fit[f"corrected_func_score"] = data_to_fit[f"func_score"]
        for condition, condition_dtf in data_to_fit.groupby("condition"):

            h_params = self.get_condition_params(condition)

            y_h_pred = self.model['f'](
                h_params, 
                self.data.binarymaps['X'][condition]
            )
            data_to_fit.loc[condition_dtf.index, f"predicted_func_score"] = y_h_pred

            if gamma_corrected:
                data_to_fit.loc[condition_dtf.index, f"corrected_func_score"] += h_params[f"γ_d"]

            # TODO is there any reason we would gamma correct the latent pred?
            y_h_latent = self.model['ϕ'](
                h_params, 
                self.data.binarymaps['X'][condition]

            )
            data_to_fit.loc[condition_dtf.index, f"predicted_latent"] = y_h_latent

            # TODO I imagine the 
            data_to_fit.loc[condition_dtf.index, f"predicted_func_score"] = y_h_pred

        # TODO assert that none of the values are nan? 
        return data_to_fit


    @property
    def mutations_df(self):
        """ Get all single mutational attributes from self.data
        updated with all model specific attributes"""

        mutations_df = copy.copy(self.data._mutations_df)

        # update the betas
        mutations_df.loc[:, "β"] = self.params["β"]

        # make predictions
        binary_single_subs = sparse.BCOO.fromdense(onp.identity(len(self.mutations)))
        for condition in self.conditions:
            
            # collect relevant params
            h_params = self.get_condition_params(condition)

            # attach relevent params to mut effects df
            mutations_df[f"S_{condition}"] = self.params[f"S_{condition}"]

            
            # predictions for all single subs
            mutations_df[f"F_{condition}"] = self.model['f'](
                h_params, 
                binary_single_subs
            )

        return mutations_df


    def get_condition_params(self, condition=None):
        """ get the relent parameters for a model prediction"""

        condition = self.data.reference if condition is None else condition

        if condition not in self.conditions:
            raise ValueError("condition does not exist in model")

        return {
            "α":self.params[f"α"],
            "β_m":self.params["β"], 
            "C_ref":self.params["C_ref"],
            "s_md":self.params[f"S_{condition}"], 
            "C_d":self.params[f"C_{condition}"],
            "γ_d":self.params[f"γ_{condition}"]
        }


    # TODO finish documentation.
    def f(self, X, condition=None):
        """ condition specific prediction on X using the biophysical model
        given current model parameters. """

        # TODO assert X is correct shape.
        # TODO assert that the substitutions exist?
        # TODO require the user
        h_params = get_condition_params(condition)
        return self.model['f'](h_params, X)

    # TODO latent prediction
    #def phi(self, X, condition): 
    

    # TODO finish documentation.
    # TODO lasso etc paramerters (**kwargs ?)
    def fit(
        self, 
        λ_lasso=1e-5,
        λ_ridge=0,
        **kwargs
    ):
        """ use jaxopt.ProximalGradiant to optimize parameters on
        `self._data_to_fit` 
        """
        # Use partial 
        # compiled_smooth_cost = Partial(smooth_cost, self.model['f'])

        solver = ProximalGradient(
            self._objective_function,
            self._proximal_function,
            tol=1e-6,
            maxiter=1000
        )

        # the reference shift and gamma parameters forced to be zero
        lock_params = {
            f"S_{self.reference}" : jnp.zeros(len(self.params['β'])),
            f"γ_{self.reference}" : jnp.zeros(shape=(1,)),
        }

        # currently we lock C_h because of odd model behavior
        for condition in self.conditions:
            lock_params[f"C_{condition}"] = jnp.zeros(shape=(1,))

        # lasso regularization on the Shift parameters
        lasso_params = {}
        for non_ref_condition in self.conditions:
            if non_ref_condition == self.reference: continue
            lasso_params[f"S_{non_ref_condition}"] = λ_lasso

        # run the optimizer
        self.params, state = solver.run(
            self.params,
            hyperparams_prox = dict(
                lasso_params = lasso_params,
                lock_params = lock_params
            ),
            data=(self.binarymaps['X'], self.binarymaps['y']),
            λ_ridge=λ_ridge,
            **kwargs
            #lower_bound=fit_params['lower_bound'],
            #hinge_scale=fit_params['hinge_scale']
        )
        













