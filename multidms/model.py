r"""
==========
model
==========

Defines JIT compiled functions for to use for 
composing models and their respective objective functions.

To obfuscate the complexity of this behavior,
We implement the ``MultiDmsModel`` class for handling 
"""

from functools import reduce

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

import pandas
import numpy as onp
import math
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from multidms import MultiDmsData
from multidms.utils import is_wt

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
    """ 
    Model for predicting latent space with
    shift parameters.
    """
   
    return (
            d_params["C_ref"]
            + d_params["C_d"] 
            + (X_h @ (d_params["β_m"] + d_params["s_md"])) 
           )

    
@jax.jit
def sigmoidal_global_epistasis(α:dict, z_h:jnp.array):
    """ 
    A flexible sigmoid function for
    modeling global epistasis.
    """

    activations = jax.nn.sigmoid(z_h[:, None])
    return (α["ge_scale"] @ activations.T) + α["ge_bias"]

    
@jax.jit
def softplus_global_epistasis(α:dict, z_h:jnp.array):
    """ 
    A flexible sigmoid function for
    modeling global epistasis.
    """

    activations = jax.nn.softplus(-1*z_h[:, None])
    return ((-1*α["ge_scale"]) @ activations.T) + α["ge_bias"]


@Partial(jax.jit, static_argnums=(0, 1))
def abstract_from_latent(
    g:jaxlib.xla_extension.CompiledFunction,
    t:jaxlib.xla_extension.CompiledFunction,
    d_params:dict, 
    z_h:jnp.array, 
    **kwargs
):
    """ 
    Biophysical model - compiled for optimization 
    until model functions ϕ, g, and t are updated. 
    Model may be composed & compiled
    with the required functions fixed 
    using `jax.tree_util.Partial.
    """

    return t(d_params, g(d_params['α'], z_h), **kwargs)


@Partial(jax.jit, static_argnums=(0, 1, 2,))
def abstract_epistasis(
    ϕ:jaxlib.xla_extension.CompiledFunction,
    g:jaxlib.xla_extension.CompiledFunction,
    t:jaxlib.xla_extension.CompiledFunction,
    d_params:dict, 
    X_h:jnp.array, 
    **kwargs
):
    """ 
    Biophysical model - compiled for optimization 
    until model functions ϕ, g, and t are updated. 
    Model may be composed & compiled
    with the required functions fixed 
    using `jax.tree_util.Partial.
    """

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
    if "ge_scale" in params["α"]:
        params["α"]["ge_scale"] = params["α"]["ge_scale"].clip(0)
    
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
    "softplus" : softplus_global_epistasis,
    "identity" : identity_activation 
}
# TODO softplus, perceptron, ispline
output_activations = {
    "softplus" : softplus_activation,
    "gelu" : gelu_activation,
    "identity" : identity_activation
}


# TODO update model description based upon hugh's recent updates
# TODO finish documenting parameters, attributes, methods, and examples.
# TODO plotting methods -> altair
class MultiDmsModel:
    r"""
    Represent one or more DMS experiments
    to obtain a predictive model for new variants,
    and tuned parameters that provide insight into
    individual mutational effects and conditional shifts
    of those effects on all non-reference conditions.

    Class instantiation
    -------------------

    To instantiate the object:
    
    1. A ``multidms.MultiDmsData`` Object for fitting.
    2. String arguments that exists within the pre-defined
    set of functions for latent and epistatic models as well
    as a final activation function. By default, we use the
    linear model with shift parameters (see Model Overview),
    and identity functions for the epistatic model 
    as well as for the output activation (i.e. the linear model).
    
    Conceptual Overview of model
    ----------------------------
    
    The ``multidms`` model applies to a case where you have DMS datasets 
    for two or more conditions and are interested in identifying shifts 
    in mutational effects between conditions.
    To do so, the model defines one condition as a reference condition.
    For each mutation, the model fits one parameter that quantifies 
    the effect of the mutation in the reference condition.
    For each non-reference condition, it also fits a shift 
    parameter that quantifies the shift in the mutation's 
    effect in the non-reference condition relative to the reference.
    Shift parameters can be regularized, encouraging most of them to be 
    close to zero. This regularization step is a useful way to eliminate 
    the effects of experimental noise, and is most useful in cases where 
    you expect most mutations to have the same effects between conditions, 
    such as for conditions that are close relatives.
    
    The model uses a global-epistasis function to disentangle the effects 
    of multiple mutations on the same variant. To do so, it assumes 
    that mutational effects additively influence a latent biophysical 
    property the protein (e.g., $\Delta G$ of folding).
    The mutational-effect parameters described above operate at this latent level.
    
    The global-epistasis function then assumes a sigmoidal relationship between 
    a protein's latent property and its functional score measured in the experiment 
    (e.g., log enrichment score). Ultimately, mutational parameters, as well as ones 
    controlling the shape of the sigmoid, are all jointly fit to maximize agreement 
    between predicted and observed functional scores acorss all variants of all conditions.
    
    Detailed description of the model
    ---------------------------------
    
    For each variant $v$ from condition $h$, we use a global-epistasis function 
    $g$ to convert a latent phenotype $\phi$ to a functional score $f$:
    
    $$f(v,h) = g_{\alpha}(\phi(v,h)) + γ_h$$
    
    where $g$ is a sigmoid and $\alpha$ is a set of parameters,
    ``ge_scale``\ , and ``ge_bias`` which define the shape of the sigmoid.
    
    The latent phenotype is computed in the following way:
    
    $$\phi(v,h) = c + \sum_{m \in v} (x\ *m + s*\ {m,h})$$
    
    where:
    
    
    * $c$ is the wild type latent phenotype for the reference condition.
    * $x_m$ is the latent phenotypic effect of mutation $m$. See details below.
    * $s_{m,h}$ is the shift of the effect of mutation $m$ in condition $h$. 
      These parameters are fixed to zero for the reference condition. For 
      non-reference conditions, they are defined in the same way as $x_m$ parameters.
    * $v$ is the set of all mutations relative to the reference wild type sequence 
      (including all mutations that separate condition $h$ from the reference condition).
    
    The $x_m$ variable is defined such that mutations are always relative to the 
    reference condition. For example, if the wild type amino acid at site 30 is an 
    A in the reference condition, and a G in a non-reference condition, 
    then a Y30G mutation in the non-reference condition is recorded as an A30G 
    mutation relative to the reference. This way, each condition informs 
    the exact same parameters, even at sites that differ in wild type amino acid.
    These are encoded in a ``BinaryMap`` object, where all sites that are non-identical 
    to the reference are 1's.
    
    Ultimately, we fit parameters using a loss function with one term that 
    scores differences between predicted and observed values and another that 
    uses L1 regularization to penalize non-zero $s_{m,h}$ values:
    
    $$ L\ *{\text{total}} = \sum*\ {h} \left[\sum\ *{v} L*\ {\text{fit}}(y\ *{v,h}, f(v,h)) + \lambda \sum*\ {m} |s_{m,h}|\right]$$
    
    where:
    
    
    * $L_{\text{total}}$ is the total loss function.
    * $L_{\text{fit}}$ is a loss function that penalizes differences 
        in predicted vs. observed functional scores.
    * $y_{v,h}$ is the experimentally measured functional score of 
        variant $v$ from condition $h$.
    
    Model using matrix algebra
    --------------------------
    
    We compute a vector or predicted latent phenotypes $P_{h}$ as:
    
    $$P_{h} = c + (X_h \cdot (β + S_h))$$
    
    where:
    
    
    * $β$ is a vector of all $β_m$ values.
    * $S\ *h$ is a matrix of all $s*\ {m,h}$ values.
    * $X_h$ is a sparse matrix, where rows are variants, 
        columns are mutations (all defined relative to the reference condition), 
        and values are weights of 0's and 1's. These weights are used to 
        compute the phenotype of each variant given the mutations present.
    * $c$ is the same as above.
    
    In the matrix algebra, the sum of $β\ *m$ and $S*\ {m,h}$ 
    gives a vector of mutational effects, with one entry per mutation.
    Multiplying the matrix $X_h$ by this vector gives a new 
    vector with one entry per variant, where values are the 
    sum of mutational effects, weighted by the variant-specific weights in $X_h$.
    Adding the $c$ value to this vector will give a vector of 
    predicted latent phenotypes for each variant.
    
    Next, the global-epistasis function can be used to convert 
        a vector of predicted latent phenotypes to a vector of 
        predicted functional scores.
    
    $$F\ *{h,pred} = g*\ {\alpha}(P_h)$$
    
    Finally, this vector could be fed into a loss function and 
    compared with a vector of observed functional scores.

    `JIT` compiled model composition
    ------------------------------

    ``multidms.model.latent_models``,
    ``multidms.model.epistatic_models``,
    and
    ``multidms.model.output_activations``
    provide JIT compiled functions for to use for 
    composing models the abstract epistasis 
    byphysical model and their respective 
    objective functions.
    For the sake of modularity, abstraction, and code-reusibility,
    we would like to be able to separate and provide options for
    the individual pieces of code that define a model such as this.
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

    def __init__(
        self,
        data:MultiDmsData,
        latent_model="phi",
        epistatic_model="identity",
        output_activation="identity",
        gamma_corrected=True,
        conditional_shifts=True,
        conditional_c=False,
        init_g_range=None,
        init_g_min=None,
        PRNGKey = 0
    ):
        """
        See class docstring.
        """

        self.gamma_corrected = gamma_corrected
        self.conditional_shifts = conditional_shifts
        self.conditional_c = conditional_c
        
        self._data = data
        
        self.params = {}
        key = jax.random.PRNGKey(PRNGKey)


        if latent_model not in latent_models.keys():
            raise ValueError(f"{latent_model} not recognized,"
                f"please use one from: {latent_models.keys()}"
                )

        if latent_model == "phi":

            n_beta_shift = len(self._data.mutations)
            self.params["β"] = jax.random.normal(shape=(n_beta_shift,), key=key)
            for condition in data.conditions:
                self.params[f"S_{condition}"] = jnp.zeros(shape=(n_beta_shift,))
                self.params[f"C_{condition}"] = jnp.zeros(shape=(1,))
            self.params["C_ref"] = jnp.array([5.0]) # 5.0 is a guess, could update

        if epistatic_model == "sigmoid":
            if init_g_range == None:
                init_g_range = 5.
            if init_g_min == None:
                init_g_min = -5.
            self.params["α"]=dict(
                ge_scale=jnp.array([init_g_range]),
                ge_bias=jnp.array([init_g_min])
            )
            
        elif epistatic_model == "softplus":
            if init_g_range == None:
                init_g_range = 1.
            if init_g_min == None:
                init_g_min = 0.
            self.params["α"]=dict(
                ge_scale=jnp.array([init_g_range]),
                ge_bias=jnp.array([init_g_min])
            )

        elif epistatic_model == "identity":
            self.params["α"]=dict(
                ghost_param = jnp.zeros(shape=(1,))
            )
            

        else:
            raise ValueError(f"{epistatic_model} not recognized,"
                f"please use one from: {epistatic_models.keys()}"
                )

        for condition in data.conditions:
            self.params[f"γ_{condition}"] = jnp.zeros(shape=(1,))

        compiled_pred = Partial(
                abstract_epistasis, # abstract function to compile
                latent_models[latent_model],
                epistatic_models[epistatic_model], 
                output_activations[output_activation]
        )
        compiled_from_latent = Partial(
            abstract_from_latent,
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
            "g" : epistatic_models[epistatic_model],
            "from_latent" : compiled_from_latent,
            "objective" : compiled_cost,
            "proximal" : lasso_lock_prox 
        })


    @property
    def variants_df(self):
        """ Get all functional score attributes from self._data
        updated with all model predictions"""

        variants_df = self._data.variants_df.copy()

        variants_df["predicted_latent"] = onp.nan
        variants_df[f"predicted_func_score"] = onp.nan
        variants_df[f"corrected_func_score"] = variants_df[f"func_score"]
        for condition, condition_dtf in variants_df.groupby("condition"):

            d_params = self.get_condition_params(condition)
            y_h_pred = self._model['f'](
                d_params, 
                self._data.training_data['X'][condition]
            )
            variants_df.loc[condition_dtf.index, f"predicted_func_score"] = y_h_pred
            if self.gamma_corrected:
                variants_df.loc[condition_dtf.index, f"corrected_func_score"] += d_params[f"γ_d"]
            y_h_latent = self._model['ϕ'](
                d_params, 
                self._data.training_data['X'][condition]
            )
            variants_df.loc[condition_dtf.index, f"predicted_latent"] = y_h_latent
            variants_df.loc[condition_dtf.index, f"predicted_func_score"] = y_h_pred

        return variants_df


    @property
    def mutations_df(self):
        """ Get all single mutational attributes from self._data
        updated with all model specific attributes"""

        mutations_df = self._data.mutations_df.copy()
        mutations_df.loc[:, "β"] = self.params["β"]
        binary_single_subs = sparse.BCOO.fromdense(onp.identity(len(self._data.mutations)))
        for condition in self._data.conditions:
            
            d_params = self.get_condition_params(condition)
            mutations_df[f"S_{condition}"] = self.params[f"S_{condition}"]
            mutations_df[f"F_{condition}"] = self._model['f'](
                d_params, 
                binary_single_subs
            )

        return mutations_df


    def sites_df(self, agg=onp.mean):
        """ Get all single mutational attributes from self._data
        updated with all model specific attributes, summarize by site"""

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        sites_df = self.mutations_df.select_dtypes(include=numerics)
        return sites_df.groupby("sites").aggregate(onp.mean)


    @property
    def wildtype_df(self):
        """ 
        Get a dataframe indexed by condition wildtype
        containing the prediction features for each.
        """
        
        wildtype_df = pandas.DataFrame(index=self._data.conditions)
        wildtype_df = wildtype_df.assign(predicted_latent=onp.nan)
        wildtype_df = wildtype_df.assign(predicted_func_score=onp.nan)
        for condition, nis in self._data.non_identical_mutations.items():
            
            binmap = self._data.binarymaps[condition]
            wt_binary = binmap.sub_str_to_binary(nis)
            d_params = self.get_condition_params(condition)

            wildtype_df.loc[f"{condition}", "predicted_latent"] = self._model['ϕ'](
                d_params, 
                wt_binary
            )

            wildtype_df.loc[f"{condition}", "predicted_func_score"] = self._model['f'](
                d_params, 
                wt_binary
            )

        return wildtype_df


    @property
    def loss(self):
        data=(self._data.training_data['X'], self._data.training_data['y'])
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


    def f(self, X, condition=None):
        """ condition specific prediction on X using the biophysical model
        given current model parameters. """

        # TODO assert X is correct shape.
        # TODO assert that the substitutions exist?
        # TODO require the user
        d_params = get_condition_params(condition)
        return self._model['f'](d_params, X)


    def predicted_latent(self, X, condition=None):
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

        if not self.conditional_c:
            for condition in self._data.conditions:
                lock_params[f"C_{condition}"] = jnp.zeros(shape=(1,))

        lasso_params = {}
        for non_ref_condition in self._data.conditions:
            if non_ref_condition == self._data.reference: continue
            lasso_params[f"S_{non_ref_condition}"] = lasso_shift

        self.params, state = solver.run(
            self.params,
            hyperparams_prox = dict(
                lasso_params = lasso_params,
                lock_params = lock_params
            ),
            data=(self._data.training_data['X'], self._data.training_data['y']),
            **kwargs
        )

    def plot_pred_accuracy(
        self,
        hue=True,
        saveas=None,
        show=False,
        figsize=[6, 5],
        xlim=None,
        ylim=None,
        **kwargs
    ):

        """
        Create a figure which visualizes the correlation
        between model predicted functional score of all
        variants in the training with ground truth measurements.
        """

        df = self.variants_df
        
        df = df.assign(is_wt = df["aa_substitutions"].apply(is_wt))
        fig, ax = plt.subplots(figsize=figsize)

        sns.scatterplot(
            data=df, x=f"predicted_func_score",
            y=f"corrected_func_score",
            hue="condition" if hue else None,
            palette=self.data.condition_colors, 
            ax=ax,
            legend=False, 
            **kwargs
        )

        for condition, values in self.wildtype_df.iterrows():
            ax.axvline(
                values.predicted_func_score, 
                label=condition, 
                c=self._data.condition_colors[condition],
            )

        xlb, xub = [-1, 1] + onp.quantile(df.predicted_func_score, [0.05, 1.0])
        ylb, yub = [-1, 1] + onp.quantile(df.corrected_func_score, [0.05, 1.0])

        ax.plot([ylb, yub], [ylb, yub], "k--", lw=1)
        start_y = 0.9
        for c, cdf in df.groupby("condition"):
            r = pearsonr(cdf[f'corrected_func_score'], cdf[f'predicted_func_score'])[0]
            ax.annotate(
                f"$r = {r:.2f}$", (.07, start_y), 
                xycoords="axes fraction", fontsize=12,
                c=self._data.condition_colors[c]
            )
            start_y += -0.05
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_ylabel("functional score")
        ax.set_xlabel("predicted functional score")
        ax.set_xlim([xlb, xub])
        ax.set_ylim([ylb, yub])

        ax.axhline(0, color="k", ls="--", lw=1)
        ax.axvline(0, color="k", ls="--", lw=1)
        
        ax.set_ylabel("functional score + γ$_{d}$")
        plt.tight_layout()
        if saveas: fig.savefig(saveas)
        if show: plt.show()
        return fig, ax


    def plot_epistasis(
        self,
        hue=True,
        saveas=None,
        show=False,
        figsize=[6, 5],
        **kwargs
    ):

        """
        Plot latent predictions against
        gamma corrected ground truth measurements
        of all samples in the training set.
        """

        df = self.variants_df
        
        df = df.assign(is_wt = df["aa_substitutions"].apply(is_wt))
        fig, ax = plt.subplots(figsize=figsize)

        sns.scatterplot(
            data=df, x="predicted_latent",
            y=f"corrected_func_score",
            hue="condition" if hue else None,
            palette=self.data.condition_colors, 
            ax=ax,
            legend=False,
            **kwargs
        )

        for condition, values in self.wildtype_df.iterrows():
            ax.axvline(
                values.predicted_latent, 
                label=condition, 
                c=self._data.condition_colors[condition]
            )

        xlb, xub = [-1, 1] + onp.quantile(df.predicted_latent, [0.05, 1.0])
        ylb, yub = [-1, 1] + onp.quantile(df.corrected_func_score, [0.05, 1.0])
        
        ϕ_grid = onp.linspace(
            xlb, xub,
            num=1000
        )
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        params = self.get_condition_params(self._data.reference)
        latent_preds = self._model["g"](params["α"], ϕ_grid)
        shape = (ϕ_grid, latent_preds)
        ax.plot(*shape, color="k", lw=1)

        ax.axhline(0, color="k", ls="--", lw=1)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.set_xlim([xlb, xub])
        ax.set_ylim([ylb, yub])
        ax.set_ylabel("functional score + γ$_{d}$")
        plt.tight_layout()

        if saveas: fig.savefig(saveas)
        if show: plt.show()
        return fig, ax


    def plot_all_param_hist(
        self,
        show=True, 
        saveas=False, 
        times_seen_threshold=3,
        figsize=None,
    ):
        # Make a list of metrics to plot
        metrics = [] # ['β']
        for condition in self._data.conditions:
            if condition == self._data.reference:
                continue
            metrics.append(f'S_{condition}')

        df = self.mutations_df
        times_seen_cols = [
            col for col in df.columns.values
            if 'times_seen' in col
        ]
        for col in times_seen_cols:
            df = df[df[col] >= times_seen_threshold]
        # Plot data
        ncols = 3
        nrows = math.ceil(len(metrics) / ncols)
        figsize=[6*ncols,5*nrows] if not figsize else figsize
        (fig, axs) = plt.subplots(
            ncols=ncols, nrows=nrows,
            figsize=figsize,
            sharey=True
        )
        axs = axs.reshape(-1)
        pal = sns.color_palette('colorblind')
        bin_width = 0.25
        min_val = -8 - bin_width/2 # math.floor(df[metrics].min().min()) - 0.25/2
        max_val = 4 # math.ceil(df[metrics].max().max())
        for (i, metric) in enumerate(metrics):
            sns.histplot(
                x=metric, data=df, ax=axs[i],
                stat='density', color=pal.as_hex()[0],
                label='muts to amino acids',
                binwidth=bin_width, binrange=(min_val, max_val),
                alpha=0.5
            )
            # Plot data for mutations leading to stop codons
            df_stops = df[
                (df['muts'] == '*')
            ]
            sns.histplot(
                x=metric, data=df_stops, ax=axs[i],
                stat='density', color=pal.as_hex()[1],
                label='muts to stop codons',
                binwidth=bin_width, binrange=(min_val, max_val),
                alpha=0.5
            )
            axs[i].set(
                xlabel=metric,
                ylim=[0,3]
            )
            axs[i].grid()

        sns.despine()
        plt.tight_layout()

        if saveas: fig.savefig(saveas)
        if show: plt.show()

    def plot_param_hist(
        self,
        show=True, 
        saveas=False, 
        times_seen_threshold=3,
        figsize=[6, 5],
    ):

        """
        Plot the histogram of a parameter.
        """

        mut_effects_df = self.mutations_df
        fig, ax = plt.subplots(figsize=figsize)

        times_seen_cols = [c for c in mut_effects_df.columns if "times" in c]
        for c in times_seen_cols:
            mut_effects_df = mut_effects_df[mut_effects_df[c]>=times_seen_threshold]

        # Plot data for all mutations
        data = mut_effects_df[mut_effects_df['muts'] != '*']
        bin_width = 0.25
        min_val = math.floor(data[param].min()) - 0.25/2
        max_val = math.ceil(data[param].max())
        sns.histplot(
            x=param, data=data, ax=ax,
            stat='density', color=self._data.condition_colors[self._data.reference],
            label='muts to amino acids',
            binwidth=bin_width, binrange=(min_val, max_val),
            alpha=0.5
        )

        # Plot data for mutations leading to stop codons
        data = mut_effects_df[mut_effects_df['muts'] == '*']
        if len(data) > 0:
            sns.histplot(
                x=param, data=data, ax=ax,
                stat='density', color="orange",
                label='muts to stop codons',
                binwidth=bin_width, binrange=(min_val, max_val),
                alpha=0.5
            )

        ax.set(xlabel=param)
        plt.tight_layout()
        ax.legend()

        if saveas: fig.savefig(saveas)
        if show: plt.show()
        return fig, ax


    def plot_param_heatmap(
        self,
        param, 
        show=True, 
        saveas=False, 
        figsize=[20, 5],
        vmin=-1,
        vmax=1
    ):

        """
        plot the heatmap of a parameters associated with specific sites and substitutions.
        """

        if not param.startswith("β") and not param.startswith("S"):
            raise ValueError("Parameter to visualize must be an existing beta, or shift parameter")

        mut_effects_df = self.mutations_df
        mut_effects_df = self.mutations_df.copy()
        mut_effects_df.sites = mut_effects_df.sites.astype(int)
        mutation_effects = mut_effects_df.pivot(
            index="muts",
            columns="sites", values=param
        )

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            mutation_effects,
            mask=mutation_effects.isnull(),
            cmap="coolwarm_r",
            center=0,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={"label": param},
            ax=ax
        )

        plt.tight_layout()
        if saveas: fig.savefig(saveas)
        if show: plt.show()
        return fig, ax


    def plot_params_by_sites(
        self,
        show=True, 
        saveas=False, 
        figsize=[20, 5],
        sites_agg_func=onp.mean
    ):
        """
        summarize shift parameter values by associated sites and conditions.
        """

        sites_df = self.sites_df(sites_agg_func).reset_index()

        fig, ax = plt.subplots(figsize=figsize)
        ax.axhline(0, color="k", ls="--", lw=1)

        for condition in self._data.conditions:
            if condition == self._data.reference: continue
            sns.lineplot(
                data=sites_df,
                x="sites", y=f"S_{condition}",
                color=self._data.condition_colors[condition],
                ax=ax
            )

        plt.tight_layout()
        if saveas: fig.savefig(saveas)
        if show: plt.show()
        return fig, ax


    def plot_fit_param_comp_scatter(
        self,
        other,
        self_param="β",
        other_param="β",
        figsize=[5,4],
        saveas=None,
        show=True,
        site_agg_func=None
    ):

        if not site_agg_func:
            dfs = [self.mutations_df, other.mutations_df]
        else:
            dfs = [
                self.sites_df(agg=site_agg_func).reset_index(), 
                other.sites_df(agg=site_agg_func).reset_index()
            ]

        combine_on = "mutation" if site_agg_func is None else "sites"
        print(dfs[0])
        comb_mut_effects = reduce(
            lambda l, r: pandas.merge(
                l, r, how="inner", on=combine_on
            ), dfs
        )
        comb_mut_effects["is_stop"] = [
            True if "*" in s else False for s in comb_mut_effects[combine_on]
        ]

        same = self_param == other_param
        x = f"{self_param}_x" if same else self_param
        y = f"{other_param}_y" if same else other_param

        fig, ax = plt.subplots(figsize=figsize)
        r = pearsonr(comb_mut_effects[x], comb_mut_effects[y])[0]
        sns.scatterplot(
            data=comb_mut_effects,
            x=x, y=y,
            hue="is_stop",
            alpha=0.6,
            palette="deep",
            ax=ax
        )

        xlb, xub = [-1, 1] + onp.quantile(comb_mut_effects[x], [0.00, 1.0])
        ylb, yub = [-1, 1] + onp.quantile(comb_mut_effects[y], [0.00, 1.0])
        min1 = min(xlb, ylb)
        max1 = max(xub, yub)
        ax.plot([min1, max1], [min1, max1], ls="--", c="k")
        ax.annotate(f"$r = {r:.2f}$", (.7, .1), xycoords="axes fraction", fontsize=12)
        plt.tight_layout()

        if saveas: fig.saveas(saveas)
        if show: plt.show()

        return fig, ax

