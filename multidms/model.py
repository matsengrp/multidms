r"""
=============
MultiDmsModel
=============

Defines :class:`MultiDmsModel` objects.
"""

from functools import reduce
import warnings

import jax

# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
import jax.numpy as jnp
import jaxlib
from jax.experimental import sparse
from jax.tree_util import Partial
from frozendict import frozendict
from jaxopt import ProximalGradient
from jaxopt.loss import huber_loss
from jaxopt.prox import prox_lasso
from jaxopt.linear_solve import solve_normal_cg

import pandas
import numpy as onp
import math
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import seaborn as sns
from scipy.stats import pearsonr
import polyclonal.plot

from multidms.utils import is_wt
from multidms import MultiDmsData
import multidms.plot
from multidms.biophysical import *


class MultiDmsModel:
    r"""
    Represent one or more DMS experiments
    to obtain tuned parameters that provide insight into
    individual mutational effects and conditional shifts
    of those effects on all non-reference conditions.

    Note
    ----
    For each variant :math:`v` from condition :math:`h`, 
    we use a global-epistasis function :math:`g` to convert a latent phenotype 
    :math:`\phi` to a functional score :math:`f`:

    .. math::

        f(v,h) = g_{\alpha}(\phi(v,h)) + γ_h

    where :math:`g` is a post-latent model
    and :math:`\alpha` is the set of parameters that define the model

    The latent phenotype is computed in the following way:

    .. math::

        \phi(v,h) = c + \sum_{m \in v} (x\ *m + s*\ {m,h})

    where:

    * :math:`c` is the wild type latent phenotype for the reference condition.
    * :math:`x_m` is the latent phenotypic effect of mutation $m$. See details below.
    * :math:`s_{m,h}` is the shift of the effect of mutation $m$ in condition $h$.
      These parameters are fixed to zero for the reference condition. For
      non-reference conditions, they are defined in the same way as :math:`x_m` parameters.
    * :math:`v` is the set of all mutations relative to the reference wild type sequence
      (including all mutations that separate condition :math:`h` from the reference condition).

    For more see the `biophysical model documentation <todo>`_

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
        with current parameters.

    Parameters
    ----------
    data : multidms.MultiDmsData
        A reference to the dataset which will define the parameters
        of the model to be fit.
    latent_model : str
        A string encoding of the compiled function for
        a latent prediction from binary one-hot encodings
        of variants. Currently, we only support the 'ϕ'
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
    init_g_range : float
        Initialize the range of two-parameter epistatic models
        (Sigmoid or Softplus).
    init_g_min : float
        Initialize the min of a two parameter epistatic models.
        (Sigmoid or Softplus).
    init_C_ref : float
        Initialize the $C_{ref}$ parameter.
    PRNGKey : int
        The initial seed key for random parameters
        assigned to Beta's and any other randomly
        initialized parameters.
    n_percep_units : int or None
        If using a perceptron global epistasis
        model, this is the number of hidden units
        used in the transform.

    TODO - re-write these examples after finalizing API 

    Example
    -------
    To create a MultiDmsModel object, all you need is
    the respective MultiDmsData object for parameter fitting,
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
        latent_model=ϕ,
        epistatic_model=identity_activation,
        output_activation=identity_activation,
        gamma_corrected=True,
        conditional_shifts=True,
        conditional_c=False,
        init_g_range=None,
        init_g_min=None,
        init_C_ref=5.0,
        PRNGKey=0,
        n_percep_units=5,
    ):
        """
        See class docstring.
        """

        self.gamma_corrected = gamma_corrected
        self.conditional_shifts = conditional_shifts
        self.conditional_c = conditional_c

        # self.latent_model = latent_model
        # self.epistatic_model = epistatic_model
        # self.output_activation = output_activation

        self._data = data

        self.params = {}
        key = jax.random.PRNGKey(PRNGKey)

        if latent_model == ϕ:

            n_beta_shift = len(self._data.mutations)
            self.params["β"] = jax.random.normal(shape=(n_beta_shift,), key=key)
            for condition in data.conditions:
                self.params[f"S_{condition}"] = jnp.zeros(shape=(n_beta_shift,))
                self.params[f"C_{condition}"] = jnp.zeros(shape=(1,))
            self.params["C_ref"] = jnp.array(
                [init_C_ref]
            )
        else:
            raise ValueError(f"{latent_model} not recognized,")

        if epistatic_model == sigmoidal_global_epistasis:
            if init_g_range == None:
                init_g_range = 5.0
            if init_g_min == None:
                init_g_min = -5.0
            self.params["α"] = dict(
                ge_scale=jnp.array([init_g_range]), ge_bias=jnp.array([init_g_min])
            )

        elif epistatic_model == softplus_global_epistasis:
            if init_g_range == None:
                init_g_range = 1.0
            if init_g_min == None:
                init_g_min = 0.0
            self.params["α"] = dict(
                ge_scale=jnp.array([init_g_range]), ge_bias=jnp.array([init_g_min])
            )

        elif epistatic_model == identity_activation:
            self.params["α"] = dict(ghost_param=jnp.zeros(shape=(1,)))

        elif epistatic_model == perceptron_global_epistasis:
            key, key1, key2, key3, key4 = jax.random.split(key, num=5)
            self.params["α"] = dict(
                p_weights_1=jax.random.normal(shape=(n_percep_units,), key=key1).clip(
                    0
                ),
                p_weights_2=jax.random.normal(shape=(n_percep_units,), key=key2).clip(
                    0
                ),
                p_biases=jax.random.normal(shape=(n_percep_units,), key=key3),
                output_bias=jax.random.normal(shape=(1,), key=key4),
            )

        else:
            raise ValueError(f"{epistatic_model} not recognized,")

        for condition in data.conditions:
            self.params[f"γ_{condition}"] = jnp.zeros(shape=(1,))

        compiled_pred = Partial(
            abstract_epistasis,  # abstract function to compile
            latent_model,
            epistatic_model,
            output_activation,
        )
        compiled_from_latent = Partial(
            abstract_from_latent,
            epistatic_model,
            output_activation,
        )
        compiled_cost = Partial(gamma_corrected_cost_smooth, compiled_pred)
        self._model = frozendict(
            {
                "ϕ": latent_model,
                "f": compiled_pred,
                "g": epistatic_model,
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
        # binary_single_subs = sparse.BCOO.fromdense(
        #    onp.identity(len(self._data.mutations))
        # )
        for condition in self._data.conditions:

            # d_params = self.get_condition_params(condition)
            if condition == self._data.reference:
                continue
            mutations_df[f"S_{condition}"] = self.params[f"S_{condition}"]
            # mutations_df[f"F_{condition}"] = self._model["f"](
            #    d_params, binary_single_subs
            # )

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
        kwargs = {
            'λ_ridge_beta': 0.,
            'λ_ridge_shift': 0.,
            'λ_ridge_gamma': 0.
        }
        data = (self._data.training_data["X"], self._data.training_data["y"])
        return self._model["objective"](self.params, data, **kwargs)


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

    def fit_reference_beta(self, **kwargs):
        """Fit the Model β's to the reference data"""

        ref_X = self._data.training_data["X"][self._data.reference]
        ref_y = self._data.training_data["y"][self._data.reference]

        self.params["β"] = solve_normal_cg(
            lambda β: ref_X @ β, ref_y, init=self.params["β"], **kwargs
        )

    def fit(self, lasso_shift=1e-5, tol=1e-6, maxiter=1000, lock_params={}, **kwargs):
        """Use jaxopt.ProximalGradiant to optimize parameters"""

        solver = ProximalGradient(
            self._model["objective"], self._model["proximal"], tol=tol, maxiter=maxiter
        )

        lock_params[f"S_{self._data.reference}"] = jnp.zeros(len(self.params["β"]))
        lock_params[f"γ_{self._data.reference}"] = jnp.zeros(shape=(1,))

        # lock_params = {
        #    f"S_{self._data.reference}": jnp.zeros(len(self.params["β"])),
        #    f"γ_{self._data.reference}": jnp.zeros(shape=(1,)),
        # }

        if not self.conditional_shifts:
            for condition in self._data.conditions:
                lock_params[f"S_{condition}"] = jnp.zeros(shape=(1,))

        if not self.gamma_corrected:
            for condition in self._data.conditions:
                lock_params[f"γ_{condition}"] = jnp.zeros(shape=(1,))

        if not self.conditional_c:
            for condition in self._data.conditions:
                lock_params[f"C_{condition}"] = jnp.zeros(shape=(1,))
        else:
            lock_params[f"C_{self._data.reference}"] = jnp.zeros(shape=(1,))

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
                lw=2,
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
            # data=df,
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
                lw=2,
            )

        xlb, xub = [-1, 1] + onp.quantile(df.predicted_latent, [0.05, 1.0])
        ylb, yub = [-1, 1] + onp.quantile(df.corrected_func_score, [0.05, 1.0])

        ϕ_grid = onp.linspace(xlb, xub, num=1000)

        params = self.get_condition_params(self._data.reference)
        latent_preds = self._model["g"](params["α"], ϕ_grid)
        shape = (ϕ_grid, latent_preds)
        ax.plot(*shape, color="k", lw=2)

        ax.axhline(0, color="k", ls="--", lw=2)
        # ax.axvline(0, color="k", ls="--", lw=2)
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

        # ax.set_yscale("log")

        ax.set(xlabel=param)
        plt.tight_layout()

        if saveas:
            fig.savefig(saveas)
        if show:
            plt.show()
        return ax

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
            # legend=True,
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

    def mut_shift_plot(self, **kwargs):
        """
        Wrapper method for visualizing the shift plot.
        see `multidms.plot.mut_shift_plot()` for more
        """

        return multidms.plot.mut_shift_plot(self, **kwargs)
