r"""
==========
model
==========

Defines JIT compiled functions for to use for 
composing global epistasis biophysical models
and their respective objective functions.

To hide the complexity of this behavior,
as well as 
We implement the ``MultiDmsModel`` class.

`JIT` compiled model composition
--------------------------------

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

Here, we do it slightly differently than this example (given
by the documentation above) where they emphasize feeding 
of partial functions into jit-compiled functions as 
static arguments that are pytree compatible.
Instead, we use partial in the more traditional sense.
The _calling_ functions (i.e. functions that call on other 
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

The compiled cost function is now suitable for 
`jaxopt <https://jaxopt.github.io/stable/index.html>`_
"""

from functools import reduce
import warnings

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
import matplotlib.patches as patches
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
    The rate of change at the hinge is defined by 'hinge_scale'. This is derived from
    https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.softplus.html
    """

    return (
        hinge_scale
        * (jnp.log(1 + jnp.exp((act - (lower_bound + d_params["γ_d"])) / hinge_scale)))
        + lower_bound
        + d_params["γ_d"]
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
        (sp / 2) * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (sp + (0.044715 * (sp**3)))))
        + lower_bound
        + d_params["γ_d"]
    )


@jax.jit
def ϕ(d_params: dict, X_h: jnp.array):
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
def sigmoidal_global_epistasis(α: dict, z_h: jnp.array):
    """
    A flexible sigmoid function for
    modeling global epistasis.
    """

    activations = jax.nn.sigmoid(z_h[:, None])
    return (α["ge_scale"] @ activations.T) + α["ge_bias"]


@jax.jit
def softplus_global_epistasis(α: dict, z_h: jnp.array):
    """
    A flexible sigmoid function for
    modeling global epistasis.
    """

    activations = jax.nn.softplus(-1 * z_h[:, None])
    return ((-1 * α["ge_scale"]) @ activations.T) + α["ge_bias"]


@Partial(jax.jit, static_argnums=(0, 1))
def abstract_from_latent(
    g: jaxlib.xla_extension.CompiledFunction,
    t: jaxlib.xla_extension.CompiledFunction,
    d_params: dict,
    z_h: jnp.array,
    **kwargs,
):
    """
    Biophysical model - compiled for optimization
    until model functions ϕ, g, and t are updated.
    Model may be composed & compiled
    with the required functions fixed
    using `jax.tree_util.Partial.
    """

    return t(d_params, g(d_params["α"], z_h), **kwargs)


@Partial(
    jax.jit,
    static_argnums=(
        0,
        1,
        2,
    ),
)
def abstract_epistasis(
    ϕ: jaxlib.xla_extension.CompiledFunction,
    g: jaxlib.xla_extension.CompiledFunction,
    t: jaxlib.xla_extension.CompiledFunction,
    d_params: dict,
    X_h: jnp.array,
    **kwargs,
):
    """
    Biophysical model - compiled for optimization
    until model functions ϕ, g, and t are updated.
    Model may be composed & compiled
    with the required functions fixed
    using `jax.tree_util.Partial.
    """

    return t(d_params, g(d_params["α"], ϕ(d_params, X_h)), **kwargs)


@jax.jit
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
latent_models = {"phi": ϕ}

"""
epistatic and output activations have the same shape
input and output, however, epistatic models
are parameterized by "α", which can differ depending
on the function chosen. For example, the identity
function requires no alpha parameters, sigmoid requires 2,
and perceptron may have up to n hidden nodes and layers.
"""
epistatic_models = {
    "sigmoid": sigmoidal_global_epistasis,
    "softplus": softplus_global_epistasis,
    "identity": identity_activation,
}

output_activations = {
    "softplus": softplus_activation,
    "gelu": gelu_activation,
    "identity": identity_activation,
}


class MultiDmsModel:
    r"""
    Represent one or more DMS experiments
    to obtain tuned parameters that provide insight into
    individual mutational effects and conditional shifts
    of those effects on all non-reference conditions.


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

    Parameters
    ----------

    data : ``multidms.MultiDmsData``
        a reference to the data you wish to fit & explore.
        This is useful for accessing various attributes
        of the data from a model class.
        See the main class description for more
        about the ``MultiDmsData`` Object.
    latent_model : str
        The latent phenotype model you wish to use.
        Currently, only 'phi' is available. This
        may be deprecated in future versions.
    epistatic_model : str
        The epistatic transform function you wish to
        use for the model. i
        See multidms.model.epistatic_models
        for the dictionary mapping a given string
        to a compiled model function.
    output_activation : str,
        The output activation function you would like to
        use for the model. See multidms.model.output_activations
        for the dictionary mapping a given string
        to a compiled model function.
    gamma_corrected : bool
        If True, use the scalar gamma (γ_d) parameter to correct
        for wildtype differences between conditions. For models
        where all conditions have the same wildtype amino acid
        sequence this behavior is generally not useful.
    conditional_shifts : bool
        If True, include conditional shift parameters.
        In most scenerios where you have multiple conditions
        in a dataset, this is a key feature of the model to
        identify where mutational effects differ.
        This may be set to False if you would like to fit
        a more traditional global epistasis model.
    conditional_c : bool
        Initialize a latent bias parameter,like C_ref,
        but for each of the individual condition in a dataset.
    PRNGKey : int
        The starting PRNG key for all random
        parameter initialization. See the ``Jax`` documentation,
        for an explanation on how this differs from
        a more traditional seed value.
    init_g_range : float or None
        If using a two-parameter epistatic,
        model, use this as the starting range of the model.
    init_g_min : float or None
        If using a two-parameter epistatic model
        (like sigmoidor softplus), use this as the
        starting minimum value of the model range.

    Attributes
    ----------

    variants_df : pandas.DataFrame
        A copy of this object's data.variants_df attribute,
        but with the added model predictions (latent and epistatic)
        given the current parameters of this model.
    mutations_df : pandas.DataFrame
        A copy of this object's data.mutations_df attribute,
        but with the added respective parameter values
        (Beta and contional shift) as well as
        model predictions (latent and epistatic)
        given the current parameters of this model.
    mutation_site_summary_df : pandas.DataFrame
        Similar to the mutations_df attribute, but aggregate
        the numerical columns by site using a chosen function.
    wildtype_df : pandas.DataFrame
        Get a dataframe indexed by condition wildtype
        containing the prediction features for each.
    loss : float
        Compute and return the current loss of the model
        with current parameters

    Parameters
    ----------

    data : MultiDmsData
        A reference to the dataset which will define the parameters
        of the model to be fit.
    latent_model : str
        A string encoding of the compiled function for
        a latent prediction from binary one-hot encodings
        of variants. Currently, we only support the 'phi'
        model.
        See Model Description section for more.
    epistatic_model : str
        A string encoding of the compiled function for
        the function mapping latent phenotype, to
        some monotonic function.
        Set to 'identity' if you wish to effectively
        skip this step in the model.
        See Model Description section for more.
    output_activation : str
        A string encoding of the compiled function for
        a final transformation on the data - primarily
        we have this for our gamma corrected clipping
        functions.
        Set to 'identity' if you wish to effectively
        skip this step in the model.
        See Model Description section for more.
    gamma_corrected : bool
        If true (default), introduce the 'gamma' parameter
        for each non-reference parameter to
        account for differences between wild type
        behavior relative to it's variants. This
        is essentially a bias added to the functional
        scores during fitting.
        See Model Description section for more.
    conditional_shifts : bool
        If true (default) initialize and fit the shift
        parameters for each non-reference condition.
        See Model Description section for more.
    conditional_c : bool
        If True (default False) introduce a bias parameter
        on the latent space parameter for each condition.
        See Model Description section for more.
    init_g_range : float or None
        Initialize the range of two parameter
        epistatic models.
    init_g_range : float or None
        Initialize the min of a two parameter
        epistatic models.
    PRNGKey : int
        The initial seed key for random parameters
        assigned to Beta's and any other randomly
        initialized parameters.

    Example
    -------

    To create a MultiDmsModel object, all you need is
    the respective MultiDmsModel object for parameter fitting,
    as well as the string encoded options for choosing a model.

    >>> model = multidms.MultiDmsModel(
    ...     data,
    ...     epistatic_model="sigmoid",
    ...     output_activation="softplus"
    ... )

    Upon initialization, you will now have access to the underlying data
    and parameters.

    >>> model.data.mutations
    ('M1E', 'M1W', 'G3P', 'G3R')
    >>> model.data.conditions
    ('1', '2')
    >>> model.data.reference
    '1'
    >>> model.data.condition_colors
    {'1': '#0072B2', '2': '#009E73'}

    The mutations_df and variants_df may of course also be accessed.

    >>> model.data.mutations_df
      mutation wts  sites muts  times_seen_1  times_seen_2
    0      M1E   M      1    E             1           3.0
    1      M1W   M      1    W             1           0.0
    2      G3P   G      3    P             1           1.0
    3      G3R   G      3    R             1           2.0

    However, if accessed directly through the Model object, you will
    get the same information, along with model/parameter specific
    features included. These are automatically updated each time you
    request the property.

    >>> model.mutations_df
      mutation wts  sites muts  times_seen_1  ...         β  S_1       F_1  S_2       F_2
    0      M1E   M      1    E             1  ...  0.080868  0.0 -0.030881  0.0 -0.030881
    1      M1W   M      1    W             1  ... -0.386247  0.0 -0.049086  0.0 -0.049086
    2      G3P   G      3    P             1  ... -0.375656  0.0 -0.048574  0.0 -0.048574
    3      G3R   G      3    R             1  ...  1.668974  0.0 -0.006340  0.0 -0.006340

    Notice the respective single mutation effects ("β"), conditional shifts (S_d),
    and predicted functional score (F_d) of each mutation in the model are now
    easily accessible. Similarly, we can take a look at the variants_df for the model,

    >>> model.variants_df
      condition aa_substitutions  ...  predicted_func_score  corrected_func_score
    0         1              G3P  ...             -0.048574                  -0.5
    1         1              G3R  ...             -0.006340                  -7.0
    2         1              M1E  ...             -0.030881                   2.0
    3         1              M1W  ...             -0.049086                   2.3
    4         2              M1E  ...             -0.044834                   1.0
    5         2          M1E P3G  ...             -0.030881                   2.7
    6         2          M1E P3R  ...             -0.005848                  -2.7
    8         2              P3G  ...             -0.033464                   0.4
    9         2              P3R  ...             -0.006340                  -5.0

    We now have access to the predicted (and gamma corrected) functional scores
    as predicted by the models current parameters.

    So far, these parameters and predictions results from them have not been tuned
    to the dataset. Let's take a look at the loss on the training dataset
    given our initialized parameters

    >>> model.loss
    DeviceArray(4.40537408, dtype=float64)

    Next, we fit the model with some chosen hyperparameters.

    >>> model.fit(maxiter=1000, lasso_shift=1e-5)
    >>> model.loss
    DeviceArray(1.01582782, dtype=float64)

    The model tunes it's parameters in place, and the subsequent call to retrieve
    the loss reflects our models loss given it's updated parameters.
    """

    def __init__(
        self,
        data: MultiDmsData,
        latent_model="phi",
        epistatic_model="identity",
        output_activation="identity",
        gamma_corrected=True,
        conditional_shifts=True,
        conditional_c=False,
        init_g_range=None,
        init_g_min=None,
        PRNGKey=0,
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
            raise ValueError(
                f"{latent_model} not recognized,"
                f"please use one from: {latent_models.keys()}"
            )

        if latent_model == "phi":

            n_beta_shift = len(self._data.mutations)
            self.params["β"] = jax.random.normal(shape=(n_beta_shift,), key=key)
            for condition in data.conditions:
                self.params[f"S_{condition}"] = jnp.zeros(shape=(n_beta_shift,))
                self.params[f"C_{condition}"] = jnp.zeros(shape=(1,))
            self.params["C_ref"] = jnp.array([5.0])  # 5.0 is a guess, could update

        if epistatic_model == "sigmoid":
            if init_g_range == None:
                init_g_range = 5.0
            if init_g_min == None:
                init_g_min = -5.0
            self.params["α"] = dict(
                ge_scale=jnp.array([init_g_range]), ge_bias=jnp.array([init_g_min])
            )

        elif epistatic_model == "softplus":
            if init_g_range == None:
                init_g_range = 1.0
            if init_g_min == None:
                init_g_min = 0.0
            self.params["α"] = dict(
                ge_scale=jnp.array([init_g_range]), ge_bias=jnp.array([init_g_min])
            )

        elif epistatic_model == "identity":
            self.params["α"] = dict(ghost_param=jnp.zeros(shape=(1,)))

        else:
            raise ValueError(
                f"{epistatic_model} not recognized,"
                f"please use one from: {epistatic_models.keys()}"
            )

        for condition in data.conditions:
            self.params[f"γ_{condition}"] = jnp.zeros(shape=(1,))

        compiled_pred = Partial(
            abstract_epistasis,  # abstract function to compile
            latent_models[latent_model],
            epistatic_models[epistatic_model],
            output_activations[output_activation],
        )
        compiled_from_latent = Partial(
            abstract_from_latent,
            epistatic_models[epistatic_model],
            output_activations[output_activation],
        )
        compiled_cost = Partial(gamma_corrected_cost_smooth, compiled_pred)
        self._model = frozendict(
            {
                "ϕ": latent_models[latent_model],
                "f": compiled_pred,
                "g": epistatic_models[epistatic_model],
                "from_latent": compiled_from_latent,
                "objective": compiled_cost,
                "proximal": lasso_lock_prox,
            }
        )

    @property
    def variants_df(self):
        """Get all functional score attributes from self._data
        updated with all model predictions"""

        variants_df = self._data.variants_df.copy()

        variants_df["predicted_latent"] = onp.nan
        variants_df[f"predicted_func_score"] = onp.nan
        variants_df[f"corrected_func_score"] = variants_df[f"func_score"]
        for condition, condition_dtf in variants_df.groupby("condition"):

            d_params = self.get_condition_params(condition)
            y_h_pred = self._model["f"](
                d_params, self._data.training_data["X"][condition]
            )
            variants_df.loc[condition_dtf.index, f"predicted_func_score"] = y_h_pred
            if self.gamma_corrected:
                variants_df.loc[
                    condition_dtf.index, f"corrected_func_score"
                ] += d_params[f"γ_d"]
            y_h_latent = self._model["ϕ"](
                d_params, self._data.training_data["X"][condition]
            )
            variants_df.loc[condition_dtf.index, f"predicted_latent"] = y_h_latent
            variants_df.loc[condition_dtf.index, f"predicted_func_score"] = y_h_pred

        return variants_df

    @property
    def mutations_df(self):
        """
        Get all single mutational attributes from self._data
        updated with all model specific attributes.
        """

        mutations_df = self._data.mutations_df.copy()
        mutations_df.loc[:, "β"] = self.params["β"]
        binary_single_subs = sparse.BCOO.fromdense(
            onp.identity(len(self._data.mutations))
        )
        for condition in self._data.conditions:

            d_params = self.get_condition_params(condition)
            mutations_df[f"S_{condition}"] = self.params[f"S_{condition}"]
            mutations_df[f"F_{condition}"] = self._model["f"](
                d_params, binary_single_subs
            )

        return mutations_df

    def mutation_site_summary_df(self, agg_func=onp.mean, times_seen_threshold=0):
        """
        Get all single mutational attributes from self._data
        updated with all model specific attributes, then aggregate
        all numerical columns by "sites" using
        ``agg`` function. The mean values are given by default.
        """

        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        mut_df = self.mutations_df.select_dtypes(include=numerics)
        times_seen_cols = [c for c in mut_df.columns if "times" in c]
        for c in times_seen_cols:
            mut_df = mut_df[mut_df[c] >= times_seen_threshold]

        return mut_df.groupby("sites").aggregate(agg_func)

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

            wildtype_df.loc[f"{condition}", "predicted_latent"] = self._model["ϕ"](
                d_params, wt_binary
            )

            wildtype_df.loc[f"{condition}", "predicted_func_score"] = self._model["f"](
                d_params, wt_binary
            )

        return wildtype_df

    @property
    def loss(self):
        data = (self._data.training_data["X"], self._data.training_data["y"])
        return self._model["objective"](self.params, data)

    @property
    def data(self):
        return self._data

    def get_condition_params(self, condition=None):
        """get the relent parameters for a model prediction"""

        condition = self._data.reference if condition is None else condition

        if condition not in self._data.conditions:
            raise ValueError("condition does not exist in model")

        return {
            "α": self.params["α"],
            "β_m": self.params["β"],
            "C_ref": self.params["C_ref"],
            "s_md": self.params[f"S_{condition}"],
            "C_d": self.params[f"C_{condition}"],
            "γ_d": self.params[f"γ_{condition}"],
        }

    def f(self, X, condition=None):
        """condition specific prediction on X using the biophysical model
        given current model parameters."""

        d_params = get_condition_params(condition)
        return self._model["f"](d_params, X)

    def predicted_latent(self, X, condition=None):
        """condition specific prediction on X using the biophysical model
        given current model parameters."""

        d_params = get_condition_params(condition)
        return self._model["ϕ"](d_params, X)

    def fit(self, lasso_shift=1e-5, tol=1e-6, maxiter=1000, **kwargs):
        """Use jaxopt.ProximalGradiant to optimize parameters"""

        solver = ProximalGradient(
            self._model["objective"], self._model["proximal"], tol=tol, maxiter=maxiter
        )

        lock_params = {
            f"S_{self._data.reference}": jnp.zeros(len(self.params["β"])),
            f"γ_{self._data.reference}": jnp.zeros(shape=(1,)),
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
            if non_ref_condition == self._data.reference:
                continue
            lasso_params[f"S_{non_ref_condition}"] = lasso_shift

        self.params, state = solver.run(
            self.params,
            hyperparams_prox=dict(lasso_params=lasso_params, lock_params=lock_params),
            data=(self._data.training_data["X"], self._data.training_data["y"]),
            **kwargs,
        )

    def plot_pred_accuracy(
        self, hue=True, show=True, saveas=None, annotate_corr=True, ax=None, **kwargs
    ):

        """
        Create a figure which visualizes the correlation
        between model predicted functional score of all
        variants in the training with ground truth measurements.
        """

        df = self.variants_df

        df = df.assign(is_wt=df["aa_substitutions"].apply(is_wt))

        if ax is None:
            fig, ax = plt.subplots(figsize=[3, 3])

        sns.scatterplot(
            data=df.sample(frac=1),
            x=f"predicted_func_score",
            y=f"corrected_func_score",
            hue="condition" if hue else None,
            palette=self.data.condition_colors,
            ax=ax,
            **kwargs,
        )

        for condition, values in self.wildtype_df.iterrows():
            ax.axvline(
                values.predicted_func_score,
                label=condition,
                c=self._data.condition_colors[condition],
                lw=2
            )

        xlb, xub = [-1, 1] + onp.quantile(df.predicted_func_score, [0.05, 1.0])
        ylb, yub = [-1, 1] + onp.quantile(df.corrected_func_score, [0.05, 1.0])

        ax.plot([ylb, yub], [ylb, yub], "k--", lw=2)
        if annotate_corr:
            start_y = 0.95
            for c, cdf in df.groupby("condition"):
                r = pearsonr(
                    cdf[f"corrected_func_score"], cdf[f"predicted_func_score"]
                )[0]
                ax.annotate(
                    f"$r = {r:.2f}$",
                    (0.01, start_y),
                    xycoords="axes fraction",
                    fontsize=12,
                    c=self._data.condition_colors[c],
                )
                start_y += -0.05
        ax.set_ylabel("functional score")
        ax.set_xlabel("predicted functional score")

        ax.axhline(0, color="k", ls="--", lw=2)
        ax.axvline(0, color="k", ls="--", lw=2)

        ax.set_ylabel("functional score + γ$_{d}$")
        plt.tight_layout()
        if saveas:
            fig.savefig(saveas)
        if show:
            plt.show()
        return ax

    def plot_epistasis(self, hue=True, show=True, saveas=None, ax=None, **kwargs):

        """
        Plot latent predictions against
        gamma corrected ground truth measurements
        of all samples in the training set.
        """

        df = self.variants_df

        df = df.assign(is_wt=df["aa_substitutions"].apply(is_wt))

        if ax is None:
            fig, ax = plt.subplots(figsize=[3, 3])

        sns.scatterplot(
            #data=df,
            data=df.sample(frac=1),
            x="predicted_latent",
            y=f"corrected_func_score",
            hue="condition" if hue else None,
            palette=self.data.condition_colors,
            ax=ax,
            **kwargs,
        )

        for condition, values in self.wildtype_df.iterrows():
            ax.axvline(
                values.predicted_latent,
                label=condition,
                c=self._data.condition_colors[condition],
                lw=2
            )

        xlb, xub = [-1, 1] + onp.quantile(df.predicted_latent, [0.05, 1.0])
        ylb, yub = [-1, 1] + onp.quantile(df.corrected_func_score, [0.05, 1.0])

        ϕ_grid = onp.linspace(xlb, xub, num=1000)

        params = self.get_condition_params(self._data.reference)
        latent_preds = self._model["g"](params["α"], ϕ_grid)
        shape = (ϕ_grid, latent_preds)
        ax.plot(*shape, color="k", lw=2)

        ax.axhline(0, color="k", ls="--", lw=2)
        ax.axvline(0, color="k", ls="--", lw=2)
        ax.set_xlim([xlb, xub])
        ax.set_ylim([ylb, yub])
        ax.set_ylabel("functional score + γ$_{d}$")
        ax.set_xlabel("predicted latent phenotype (ϕ)")
        plt.tight_layout()

        if saveas:
            fig.savefig(saveas)
        if show:
            plt.show()
        return ax

    def plot_param_hist(
        self, param, show=True, saveas=False, times_seen_threshold=3, ax=None, **kwargs
    ):

        """
        Plot the histogram of a parameter.
        """

        mut_effects_df = self.mutations_df

        if ax is None:
            fig, ax = plt.subplots(figsize=[3, 3])

        times_seen_cols = [c for c in mut_effects_df.columns if "times" in c]
        for c in times_seen_cols:
            mut_effects_df = mut_effects_df[mut_effects_df[c] >= times_seen_threshold]

        # Plot data for all mutations
        data = mut_effects_df[mut_effects_df["muts"] != "*"]
        bin_width = 0.25
        min_val = math.floor(data[param].min()) - 0.25 / 2
        max_val = math.ceil(data[param].max())
        sns.histplot(
            x=param,
            data=data,
            ax=ax,
            stat="density",
            label="muts to amino acids",
            binwidth=bin_width,
            binrange=(min_val, max_val),
            alpha=0.5,
            **kwargs,
        )

        # Plot data for mutations leading to stop codons
        data = mut_effects_df[mut_effects_df["muts"] == "*"]
        if len(data) > 0:
            sns.histplot(
                x=param,
                data=data,
                ax=ax,
                stat="density",
                label="muts to stop codons",
                binwidth=bin_width,
                binrange=(min_val, max_val),
                alpha=0.5,
                **kwargs,
            )

        ax.set(xlabel=param)
        plt.tight_layout()

        if saveas:
            fig.savefig(saveas)
        if show:
            plt.show()
        return fig, ax

    def plot_param_heatmap(
        self, param, show=True, saveas=False, times_seen_threshold=3, ax=None, **kwargs
    ):

        """
        plot the heatmap of a parameters associated with specific sites and substitutions.
        """

        if not param.startswith("β") and not param.startswith("S"):
            raise ValueError(
                "Parameter to visualize must be an existing beta, or shift parameter"
            )

        mut_effects_df = self.mutations_df

        if ax is None:
            fig, ax = plt.subplots(figsize=[12, 3])

        times_seen_cols = [c for c in mut_effects_df.columns if "times" in c]
        for c in times_seen_cols:
            mut_effects_df = mut_effects_df[mut_effects_df[c] >= times_seen_threshold]

        mut_effects_df.sites = mut_effects_df.sites.astype(int)
        mutation_effects = mut_effects_df.pivot(
            index="muts", columns="sites", values=param
        )

        sns.heatmap(
            mutation_effects,
            mask=mutation_effects.isnull(),
            cmap="coolwarm_r",
            center=0,
            cbar_kws={"label": param},
            ax=ax,
            **kwargs,
        )

        plt.tight_layout()
        if saveas:
            fig.savefig(saveas)
        if show:
            plt.show()
        return ax

    def plot_shifts_by_site(
        self,
        condition,
        show=True,
        saveas=False,
        times_seen_threshold=3,
        agg_func=onp.mean,
        ax=None,
        **kwargs,
    ):
        """
        summarize shift parameter values by associated sites and conditions.
        """

        mutation_site_summary_df = self.mutation_site_summary_df(
            agg_func, times_seen_threshold=times_seen_threshold
        ).reset_index()

        if ax is None:
            fig, ax = plt.subplots(figsize=[12, 3])

        max_value = 0
        ax.axhline(0, color="k", ls="--", lw=1)

        sns.lineplot(
            data=mutation_site_summary_df,
            x="sites",
            y=f"S_{condition}",
            color=self._data.condition_colors[condition],
            ax=ax,
            **kwargs,
        )
        color = [
            self.data.condition_colors[condition]
            if not s in self.data.non_identical_sites[condition]
            else (0.0, 0.0, 0.0)
            for s in mutation_site_summary_df.sites
        ]
        size = [
            0.5 if not s in self.data.non_identical_sites[condition] else 1.0
            for s in mutation_site_summary_df.sites
        ]
        sns.scatterplot(
            data=mutation_site_summary_df,
            x="sites",
            y=f"S_{condition}",
            size=size,
            color=onp.array(color),
            ax=ax,
            legend=False,
            **kwargs,
        )

        if show:
            plt.show()
        return ax

    def plot_fit_param_comp_scatter(
        self,
        other,
        self_param="β",
        other_param="β",
        figsize=[5, 4],
        saveas=None,
        show=True,
        site_agg_func=None,
    ):

        if not site_agg_func:
            dfs = [self.mutations_df, other.mutations_df]
        else:
            dfs = [
                self.mutation_site_summary_df(agg=site_agg_func).reset_index(),
                other.mutation_site_summary_df(agg=site_agg_func).reset_index(),
            ]

        combine_on = "mutation" if site_agg_func is None else "sites"
        comb_mut_effects = reduce(
            lambda l, r: pandas.merge(l, r, how="inner", on=combine_on), dfs
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
            x=x,
            y=y,
            hue="is_stop",
            alpha=0.6,
            palette="deep",
            ax=ax,
        )

        xlb, xub = [-1, 1] + onp.quantile(comb_mut_effects[x], [0.00, 1.0])
        ylb, yub = [-1, 1] + onp.quantile(comb_mut_effects[y], [0.00, 1.0])
        min1 = min(xlb, ylb)
        max1 = max(xub, yub)
        ax.plot([min1, max1], [min1, max1], ls="--", c="k")
        ax.annotate(f"$r = {r:.2f}$", (0.7, 0.1), xycoords="axes fraction", fontsize=12)
        plt.tight_layout()

        if saveas:
            fig.saveas(saveas)
        if show:
            plt.show()

        return fig, ax
