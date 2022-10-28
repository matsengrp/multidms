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


import json
import copy

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxlib
from jax.experimental import sparse
from jax.tree_util import Partial
from frozendict import frozendict
from jaxopt import ProximalGradient
from jaxopt.loss import huber_loss
from jaxopt.prox import prox_lasso
import numpy as onp

import multidms

@jax.jit
def identity_activation(d_params, act, **kwargs):
    return act


@jax.jit
def softplus_activation(d_params, act, lower_bound=-3.5, hinge_scale=0.1, **kwargs):
    """
    A modified softplus that hinges at 'lower_bound'. 
    The rate of change at the hinge is defined by 'hinge_scale'.

    This is derived from 
    https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softplus.html
    """

    return (
        hinge_scale * (
            jnp.log(
                1 + jnp.exp(
                    (act - (lower_bound + d_params["γ_d"])) / hinge_scale)
            )
        ) + lower_bound + d_params["γ_d"]
    )


@jax.jit
def gelu_activation(d_params, act, lower_bound=-3.5):
    """ 
    A modified Gaussian error linear unit activation function,
    where the lower bound asymptote is defined by `lower_bound`.

    This is derived from 
    https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.gelu.html
    """

    sp = act - (lower_bound + d_params["γ_d"])
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

    return t(d_params, g(d_params['α'], ϕ(d_params, X_h)), **kwargs)


@jax.jit
def lasso_lock_prox(
    params, 
    hyperparams_prox=dict(
        lasso_params=None, 
        lock_params=None,
    ),
    scaling=1.0
):
    
    # enforce monotonic epistasis
    for param, value in params["α"].items():
        params["α"][param] = params["α"][param].clip(0)
    
    if hyperparams_prox["lasso_params"] is not None:
        for key, value in hyperparams_prox["lasso_params"].items():
            params[key] = prox_lasso(params[key], value, scaling)

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
        loss += huber_loss(
            y[condition] + d_params[f"γ_d"], y_d_predicted, δ
        ).mean()
        
        # compute a regularization term that penalizes non-zero
        # shift parameters and add it to the loss function
        ridge_penalty = λ_ridge * (d_params["s_md"] ** 2).sum()
        loss += ridge_penalty

    return loss


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
    "gelu" : gelu_activation,
    "identity" : identity_activation
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
        data:multidms.data.MultiDmsData,
        latent_model="phi",
        epistatic_model="identity",
        output_activation="identity",
        gamma_corrected=True,
        conditional_shifts=True,
        conditional_c=False, # should we remove this entirely?
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
        self.gamma_corrected = gamma_corrected
        self.conditional_shifts = conditional_shifts
        self.conditional_c = conditional_c
        
        ######
        # DATA
        ######

        # Store all the relevant data as a reference to the
        # original MultiDmsData object.

        # all the checks are done for date.

        # This object overides the mutations_df and variants_df
        # properties from this model object, we'll return
        # copies of the relevant snippets of data plus
        # the attributes defined by a model fit within this object.
        self._data = data
        
        ########
        # PARAMS
        ########

        # we now initialize parameters based upon models
        # chosen. After this, we will simply 
        self.params = {}
        key = jax.random.PRNGKey(PRNGKey)


        if latent_model not in latent_models.keys():
            raise ValueError(f"{latent_model} not recognized,"
                f"please use one from: {latent_models.keys()}"
                )

        if latent_model == "phi":

            # initialize beta parameters from normal distribution.
            n_beta_shift = len(self._data.mutations)
            self.params["β"] = jax.random.normal(shape=(n_beta_shift,), key=key)

            # initialize shift parameters
            for condition in data.conditions:
                # We expect most shift parameters to be close to zero
                self.params[f"S_{condition}"] = jnp.zeros(shape=(n_beta_shift,))
                self.params[f"C_{condition}"] = jnp.zeros(shape=(1,))

            # all mode
            self.params["C_ref"] = jnp.array([5.0]) # 5.0 is a guess, could update



        # TODO softplus, perceptron, ispline
        if epistatic_model == "sigmoid":
            self.params["α"]=dict(
                ge_scale=jnp.array([init_sig_range]),
                ge_bias=jnp.array([init_sig_min])
        )
        elif epistatic_model == "identity":
            # this will never be used ...
            self.params["α"]=dict(
                ghost_param = jnp.zeros(shape=(1,))
            )
            

        else:
            raise ValueError(f"{epistatic_model} not recognized,"
                f"please use one from: {epistatic_models.keys()}"
                )

        for condition in data.conditions:
            self.params[f"γ_{condition}"] = jnp.zeros(shape=(1,))

        ################
        # COMPILED MODEL
        ################

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
        self._model = frozendict({
            "ϕ" : latent_models[latent_model],
            "f" : compiled_pred,
            "objective" : compiled_cost,
            "proximal" : lasso_lock_prox 
        })


    @property
    def variants_df(self):
        """ Get all functional score attributes from self._data
        updated with all model predictions"""
        # HERE
        # TODO How about fit_data? bc it may have already been fit.
        variants_df = copy.copy(self._data.variants_df)

        variants_df["predicted_latent"] = onp.nan
        variants_df[f"predicted_func_score"] = onp.nan
        variants_df[f"corrected_func_score"] = variants_df[f"func_score"]
        for condition, condition_dtf in variants_df.groupby("condition"):

            d_params = self.get_condition_params(condition)

            y_h_pred = self._model['f'](
                d_params, 
                self._data.binarymaps['X'][condition]
            )
            variants_df.loc[condition_dtf.index, f"predicted_func_score"] = y_h_pred

            if self.gamma_corrected:
                variants_df.loc[condition_dtf.index, f"corrected_func_score"] += d_params[f"γ_d"]

            # TODO is there any reason we would gamma correct the latent pred?
            y_h_latent = self._model['ϕ'](
                d_params, 
                self._data.binarymaps['X'][condition]
            )
            variants_df.loc[condition_dtf.index, f"predicted_latent"] = y_h_latent

            # TODO I imagine the 
            variants_df.loc[condition_dtf.index, f"predicted_func_score"] = y_h_pred

        # TODO assert that none of the values are nan? 
        return variants_df


    @property
    def mutations_df(self):
        """ Get all single mutational attributes from self._data
        updated with all model specific attributes"""

        mutations_df = copy.copy(self._data.mutations_df)

        # update the betas
        mutations_df.loc[:, "β"] = self.params["β"]

        # make predictions
        binary_single_subs = sparse.BCOO.fromdense(onp.identity(len(self._data.mutations)))
        for condition in self._data.conditions:
            
            # collect relevant params
            d_params = self.get_condition_params(condition)

            # attach relevent params to mut effects df
            mutations_df[f"S_{condition}"] = self.params[f"S_{condition}"]

            
            # predictions for all single subs
            mutations_df[f"F_{condition}"] = self._model['f'](
                d_params, 
                binary_single_subs
            )

        return mutations_df


    @property
    def loss(self):
        data=(self._data.binarymaps['X'], self._data.binarymaps['y'])
        return self._model['objective'](self.params, data)


    @property
    def data(self):
        return self._data


    def get_condition_params(self, condition=None):
        """ get the relent parameters for a model prediction"""

        condition = self._data.reference if condition is None else condition

        if condition not in self._data.conditions:
            raise ValueError("condition does not exist in model")

        return {
            "α":self.params["α"],
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
        d_params = get_condition_params(condition)
        return self._model['f'](d_params, X)


    # TODO finish documentation.
    def latent_prediction(self, X, condition=None):
        """ condition specific prediction on X using the biophysical model
        given current model parameters. """

        # TODO assert X is correct shape.
        # TODO assert that the substitutions exist?
        # TODO require the user
        d_params = get_condition_params(condition)
        return self._model['ϕ'](d_params, X)


    # TODO finish documentation.
    # TODO lasso etc paramerters (**kwargs ?)
    def fit(
        self, 
        lasso_shift = 1e-5,
        tol=1e-6,
        maxiter=1000,
        **kwargs
    ):
        """ Use jaxopt.ProximalGradiant to optimize parameters """

        solver = ProximalGradient(
            self._model["objective"],
            self._model["proximal"],
            tol=tol,
            maxiter=maxiter
        )

        # the reference shift and gamma parameters forced to be zero
        lock_params = {
            f"S_{self._data.reference}" : jnp.zeros(len(self.params['β'])),
            f"γ_{self._data.reference}" : jnp.zeros(shape=(1,))
        }

        if not self.conditional_shifts:
            for condition in self._data.conditions:
                lock_params["S_{condition}"] = jnp.zeros(shape=(1,))

        if not self.gamma_corrected:
            for condition in self._data.conditions:
                lock_params["γ_{condition}"] = jnp.zeros(shape=(1,))

        # currently we lock C_h because of odd model behavior
        if not self.conditional_c:
            for condition in self._data.conditions:
                lock_params[f"C_{condition}"] = jnp.zeros(shape=(1,))

        # lasso regularization on the Shift parameters
        lasso_params = {}
        for non_ref_condition in self._data.conditions:
            if non_ref_condition == self._data.reference: continue
            lasso_params[f"S_{non_ref_condition}"] = lasso_shift

        # run the optimizer
        self.params, state = solver.run(
            self.params,
            hyperparams_prox = dict(
                lasso_params = lasso_params,
                lock_params = lock_params
            ),
            data=(self._data.binarymaps['X'], self._data.binarymaps['y']),
            **kwargs
        )
