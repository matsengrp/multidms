r"""
=====
model
=====

Defines :class:`Model` objects.
"""

import math
import warnings
from functools import lru_cache, partial, cached_property
from frozendict import frozendict

from multidms import Data
import multidms.biophysical
from multidms.plot import _lineplot_and_heatmap
from multidms.utils import transform, difference_matrix

import jax
import jax.numpy as jnp
import numpy as onp
import pandas as pd
import scipy
import pylops
from scipy.stats import pearsonr
from jax.experimental import sparse
from jaxopt import ProximalGradient

import seaborn as sns
from matplotlib import pyplot as plt

jax.config.update("jax_enable_x64", True)


class Model:
    r"""
    Represent one or more DMS experiments
    to obtain tuned parameters that provide insight into
    individual mutational effects and conditional shifts
    of those effects on all non-reference conditions.
    For more see the biophysical model documentation

    Parameters
    ----------
    data : multidms.Data
        A reference to the dataset which will define the parameters
        of the model to be fit.
    epistatic_model : <class 'function'>
        A function which will transform the latent
        effects of mutations into a functional score.
        See the biophysical model documentation
        for more.
    output_activation : <class 'function'>
        A function which will transform the output of the
        global epistasis function. Defaults to the identity function
        (no activation). See the biophysical model documentation
    conditional_shifts : bool
        If true (default) initialize and fit the shift
        parameters for each non-reference condition.
        See Model Description section for more.
        Defaults to True.
    alpha_d : bool
        If True introduce a latent offset parameter
        for each condition.
        See the biophysical docs section for more.
        Defaults to True.
    gamma_corrected : bool
        If true (default), introduce the 'gamma' parameter
        for each non-reference parameter to
        account for differences between wild type
        behavior relative to its variants. This
        is essentially a bias added to the functional
        scores during fitting.
        See Model Description section for more.
        Defaults to False.
    PRNGKey : int
        The initial seed key for random parameters
        assigned to Betas and any other randomly
        initialized parameters.
        for more.
    init_beta_naught : float
        Initialize the latent offset parameter
        applied to all conditions.
        See the biophysical docs section for more.
    init_theta_scale : float
        Initialize the scaling parameter :math:`\theta_{\text{scale}}` of
        a two-parameter epistatic model (Sigmoid or Softplus).
    init_theta_bias : float
        Initialize the bias parameter :math:`\theta_{\text{bias}}` of
        a two parameter epistatic model (Sigmoid or Softplus).
    init_beta_variance : float
        Beta parameters are initialized by sampling from
        a normal distribution. This parameter specifies the
        variance of the distribution being sampled.
    n_hidden_units : int or None
        If using :func:`multidms.biophysical.nn_global_epistasis`
        as the epistatic model, this is the number of hidden units
        used in the transform.
    lower_bound : float or None
        If using :func:`multidms.biophysical.softplus_activation`
        as the output activation, this is the lower bound of the
        softplus function.
    name : str or None
        Name of the Model object. If None, will be assigned
        a unique name based upon the number of data objects
        instantiated.

    Example
    -------
    To create a :class:`Model` object, all you need is
    the respective :class:`Data` object for parameter fitting.

    >>> import multidms
    >>> from tests.test_data import data
    >>> model = multidms.Model(data)

    Upon initialization, you will now have access to the underlying data
    and parameters.

    >>> model.data.mutations
    ('M1E', 'M1W', 'G3P', 'G3R')
    >>> model.data.conditions
    ('a', 'b')
    >>> model.data.reference
    'a'
    >>> model.data.condition_colors
    {'a': '#0072B2', 'b': '#CC79A7'}

    The mutations_df and variants_df may of course also be accessed.
    First, we set pandas to display all rows and columns.

    >>> import pandas as pd
    >>> pd.set_option('display.max_rows', None)
    >>> pd.set_option('display.max_columns', None)

    >>> model.data.mutations_df  # doctest: +NORMALIZE_WHITESPACE
      mutation wts  sites muts  times_seen_a  times_seen_b
    0      M1E   M      1    E             1             3
    1      M1W   M      1    W             1             0
    2      G3P   G      3    P             1             4
    3      G3R   G      3    R             1             2

    However, if accessed directly through the :class:`Model` object, you will
    get the same information, along with model/parameter specific
    features included. These are automatically updated each time you
    request the property.

    >>> model.get_mutations_df()  # doctest: +NORMALIZE_WHITESPACE
             wts  sites muts  times_seen_a  times_seen_b  beta_a  beta_b  shift_b  \
    mutation
    M1E        M      1    E             1             3     0.0     0.0      0.0
    M1W        M      1    W             1             0     0.0    -0.0      0.0
    G3P        G      3    P             1             4    -0.0    -0.0     -0.0
    G3R        G      3    R             1             2    -0.0     0.0     -0.0
    <BLANKLINE>
              predicted_func_score_a  predicted_func_score_b
    mutation
    M1E                          0.0                     0.0
    M1W                          0.0                     0.0
    G3P                          0.0                     0.0
    G3R                          0.0                     0.0

    Notice the respective single mutation effects (``"beta"``), conditional shifts
    (``shift_d``),
    and predicted functional score (``F_d``) of each mutation in the model are now
    easily accessible. Similarly, we can take a look at the variants_df for the model,

    >>> model.get_variants_df()  # doctest: +NORMALIZE_WHITESPACE
       condition aa_substitutions  func_score var_wrt_ref  predicted_latent  \
    0         a              M1E         2.0         M1E               0.0
    1         a              G3R        -7.0         G3R               0.0
    2         a              G3P        -0.5         G3P               0.0
    3         a              M1W         2.3         M1W               0.0
    4         b              M1E         1.0     G3P M1E               0.0
    5         b              P3R        -5.0         G3R               0.0
    6         b              P3G         0.4                           0.0
    7         b          M1E P3G         2.7         M1E               0.0
    8         b          M1E P3R        -2.7     G3R M1E               0.0
       predicted_func_score
    0                   0.0
    1                   0.0
    2                   0.0
    3                   0.0
    4                   0.0
    5                   0.0
    6                   0.0
    7                   0.0
    8                   0.0


    We now have access to the predicted (and gamma corrected) functional scores
    as predicted by the models current parameters.

    So far, these parameters and predictions results from them have not been tuned
    to the dataset. Let's take a look at the loss on the training dataset
    given our initialized parameters

    >>> model.loss
    2.9370000000000003

    Next, we fit the model with some chosen hyperparameters.

    >>> model.fit(maxiter=10, lasso_shift=1e-5, warn_unconverged=False)
    >>> model.loss
    0.3483478119356665

    The model tunes its parameters in place, and the subsequent call to retrieve
    the loss reflects our models loss given its updated parameters.
    """  # noqa: E501

    def __init__(
        self,
        data: Data,
        epistatic_model=multidms.biophysical.sigmoidal_global_epistasis,
        output_activation=multidms.biophysical.identity_activation,
        # gamma_corrected=False,
        PRNGKey=0,
        lower_bound=None,
        n_hidden_units=5,
        init_theta_scale=5.0,
        init_theta_bias=-5.0,
        init_beta_variance=0.0,
        name=None,
    ):
        """See class docstring."""
        # self.gamma_corrected = gamma_corrected GAMMA

        self._data = data

        self._scaled_data_params = {}
        key = jax.random.PRNGKey(PRNGKey)

        # initialize beta and shift parameters
        # note that the only option is the additive model
        # as defined in multidms.biophysical.additive_model
        latent_model = multidms.biophysical.additive_model
        if latent_model == multidms.biophysical.additive_model:
            self._scaled_data_params["beta0"] = {
                cond: jnp.zeros(shape=(1,)) for cond in data.conditions
            }

            n_beta_shift = len(self._data.mutations)
            beta_keys = jax.random.split(key, num=len(self.data.conditions))
            self._scaled_data_params["beta"] = {
                cond: init_beta_variance
                * jax.random.normal(shape=(n_beta_shift,), key=ikey)
                for cond, ikey in zip(data.conditions, beta_keys)
            }
            self._scaled_data_params["shift"] = {
                cond: self._scaled_data_params["beta"][self.data.reference]
                - self._scaled_data_params["beta"][cond]
                for cond in data.conditions
            }
            # GAMMA
            # self._params["gamma"] = {
            #     cond: jnp.zeros(shape=(1,)) for cond in data.conditions
            # }
        else:
            raise ValueError(f"{latent_model} not recognized,")

        # initialize theta parameters
        if epistatic_model == multidms.biophysical.sigmoidal_global_epistasis:
            self._scaled_data_params["theta"] = dict(
                ge_scale=jnp.array([init_theta_scale]),
                ge_bias=jnp.array([init_theta_bias]),
            )

        elif epistatic_model == multidms.biophysical.softplus_global_epistasis:
            if output_activation != multidms.biophysical.softplus_activation:
                warnings.warn(
                    "softplus_global_epistasis has no natural lower bound,"
                    " we highly suggest using a softplus output activation"
                    "with a lower bound specified when using this model."
                )

            self._scaled_data_params["theta"] = dict(
                ge_scale=jnp.array([init_theta_scale]),
                ge_bias=jnp.array([init_theta_bias]),
            )

        elif epistatic_model == multidms.biophysical.identity_activation:
            self._scaled_data_params["theta"] = dict(
                ge_scale=jnp.zeros(shape=(1,)),
                ge_bias=jnp.zeros(shape=(1,)),
            )

        elif epistatic_model == multidms.biophysical.nn_global_epistasis:
            if n_hidden_units is None:
                raise ValueError(
                    "n_hidden_units must be specified for nn_global_epistasis"
                )
            key, key1, key2, key3, key4 = jax.random.split(key, num=5)
            self._scaled_data_params["theta"] = dict(
                p_weights_1=jax.random.normal(shape=(n_hidden_units,), key=key1).clip(
                    0
                ),
                p_weights_2=jax.random.normal(shape=(n_hidden_units,), key=key2).clip(
                    0
                ),
                p_biases=jax.random.normal(shape=(n_hidden_units,), key=key3),
                output_bias=jax.random.normal(shape=(1,), key=key4),
            )

        else:
            raise ValueError(f"{epistatic_model} not recognized,")

        if output_activation == multidms.biophysical.softplus_activation:
            if lower_bound is None:
                raise ValueError(
                    "softplus activation requires a lower bound be specified"
                )
            if not isinstance(lower_bound, float):
                raise ValueError("lower_bound must be a float")

            output_activation = partial(
                multidms.biophysical.softplus_activation, lower_bound=lower_bound
            )

        # compile the model components
        pred = partial(
            multidms.biophysical._abstract_epistasis,  # abstract function to compile
            latent_model,
            epistatic_model,
            output_activation,
        )
        from_latent = partial(
            multidms.biophysical._abstract_from_latent,
            epistatic_model,
            output_activation,
        )
        objective = partial(multidms.biophysical.smooth_objective, pred)

        proximal = (
            multidms.biophysical.proximal_objective
            if len(self._data.conditions) > 1
            else multidms.biophysical.proximal_box_constraints
        )

        self._model_components = frozendict(
            {
                "latent_model": multidms.biophysical.additive_model,
                "g": epistatic_model,
                "output_activation": output_activation,
                "f": pred,
                "from_latent": from_latent,
                "objective": objective,
                "proximal": proximal,
            }
        )

        self._name = name if isinstance(name, str) else "unnamed"

        # None of the following are set until the fit() is called.
        self._state = None
        self._convergence_trajectory = None
        self._converged = False

    def __repr__(self):
        """Returns a string representation of the object."""
        return f"{self.__class__.__name__}"

    def __str__(self):
        """
        Returns a string representation of the object with a few helpful
        attributes.
        """
        return (
            f"{self.__class__.__name__}\n"
            f"Name: {self.name}\n"
            f"Data: {self.data.name}\n"
            f"Converged: {self.converged}\n"
        )

    def _clear_cache(self):
        """
        identify and clear cached properties. This is useful
        after a model has been fit and the parameters have been
        updated.
        """
        # find all cached properties and clear them
        cls = self.__class__
        cached_properties = [
            cp for cp in dir(self) if isinstance(getattr(cls, cp, cls), cached_property)
        ]
        for a in cached_properties:
            self.__dict__.pop(a, None)

        # find all lru_cache methods and call clear_cache on them
        for a in dir(self):
            attr = getattr(self, a)
            if hasattr(attr, "cache_clear"):
                attr.cache_clear()

    @property
    def name(self) -> str:
        """The name of the data object."""
        return self._name

    @property
    def state(self) -> dict:
        """The current state of the model."""
        return self._state

    @property
    def converged(self) -> bool:
        """Whether the model tolerance threshold was passed on last fit."""
        return self._converged

    @property
    def data(self) -> multidms.Data:
        """
        multidms.Data Object this model references for fitting
        its parameters.
        """
        return self._data

    @property
    def model_components(self) -> frozendict:
        """
        A frozendict which hold the individual components of the model
        as well as the objective and forward functions.
        """
        return self._model_components

    @property
    def convergence_trajectory_df(self):
        """
        The state.error through each training iteration.
        Currentlty, this is reset each time the fit() method is called
        """
        return self._convergence_trajectory

    @cached_property
    def params(self) -> dict:
        """A copy of all current model parameters"""
        return transform(self._scaled_data_params, self.data.bundle_idxs)

    @cached_property
    def loss(self) -> float:
        """
        Compute un-penalized model loss on all experimental training data
        without ridge or lasso penalties included.
        """
        data = (self.data.training_data["X"], self.data.training_data["y"])
        return float(jax.jit(self.model_components["objective"])(self.params, data))

    @cached_property
    def conditional_loss(self) -> float:
        """Compute un-penalized loss individually for each condition."""
        X, y = self.data.training_data["X"], self.data.training_data["y"]
        loss_fxn = jax.jit(self.model_components["objective"])
        ret = {}
        for condition in self.data.conditions:
            condition_data = ({condition: X[condition]}, {condition: y[condition]})
            ret[condition] = float(loss_fxn(self.params, condition_data))
        ret["total"] = sum(ret.values()) / len(ret.values())
        return ret

    @cached_property
    def wildtype_df(self):
        """
        Get a dataframe indexed by condition wildtype
        containing the prediction features for each.
        """
        wildtype_df = (
            pd.DataFrame(index=self.data.conditions)
            .assign(predicted_latent=onp.nan)
            .assign(predicted_func_score=onp.nan)
        )
        for condition in self.data.conditions:
            for pheno, model in zip(
                ["latent", "func_score"],
                [self.latent_fromsubs, self.phenotype_fromsubs],
            ):
                wildtype_df.loc[condition, f"predicted_{pheno}"] = model("", condition)

        return wildtype_df

    @lru_cache(maxsize=3)
    def get_variants_df(self, phenotype_as_effect=True):
        """
        Training data with model predictions for latent,
        and functional score phenotypes.

        Parameters
        ----------
        phenotype_as_effect : bool
            if True, phenotypes (both latent, and func_score)
            are calculated as the _difference_ between predicted
            phenotype of a given variant and the respective experimental
            wildtype prediction. Otherwise, report the unmodified
            model prediction.

        Returns
        -------
        pandas.DataFrame
            A copy of the training data, `self.data.variants_df`,
            with the phenotypes added. Phenotypes are predicted
            based on the current state of the model.
        """
        # this is what well update and return
        variants_df = self.data.variants_df.copy()

        # initialize new columns
        for pheno in ["latent", "func_score"]:
            variants_df[f"predicted_{pheno}"] = onp.nan

        # if we're a gamma corrected model, also report the "corrected"
        # observed func score, as we do during training.
        # GAMMA
        # if self.gamma_corrected:
        # variants_df["corrected_func_score"] = variants_df["func_score"]

        # get the wildtype predictions for each condition
        if phenotype_as_effect:
            wildtype_df = self.wildtype_df

        models = {
            "latent": jax.jit(self.model_components["latent_model"]),
            "func_score": jax.jit(self.model_components["f"]),
        }

        for condition, condition_df in variants_df.groupby("condition"):
            d_params = self.get_condition_params(condition)
            X = self._data.training_data["X"][condition]

            # prediction and effect
            for pheno in ["latent", "func_score"]:
                Y_pred = onp.array(models[pheno](d_params, X))
                if phenotype_as_effect:
                    Y_pred -= wildtype_df.loc[condition, f"predicted_{pheno}"]

                variants_df.loc[condition_df.index, f"predicted_{pheno}"] = Y_pred

            # GAMMA
            # if self.gamma_corrected:
            #     variants_df.loc[
            # condition_df.index,
            # "corrected_func_score"
            # ] += d_params[
            #         "gamma"
            #     ]

        return variants_df

    @lru_cache(maxsize=3)
    def get_mutations_df(
        self, times_seen_threshold=0, phenotype_as_effect=True, return_split=True
    ):
        """
        Mutation attributes and phenotypic effects
        based on the current state of the model.

        Parameters
        ----------
        times_seen_threshold : int, optional
            Only report mutations that have been seen at least
            this many times in each condition. Defaults to 0.
        phenotype_as_effect : bool, optional
            if True, phenotypes are reported as the difference
            from the conditional wildtype prediction. Otherwise,
            report the unmodified model prediction.
        return_split : bool, optional
            If True, return the split mutations as separate columns:
            'wts', 'sites', and 'muts'.
            Defaults to True.

        Returns
        -------
        pandas.DataFrame
            A copy of the mutations data, `self.data.mutations_df`,
            with the mutations column set as the index, and columns
            with the mutational attributes (e.g. betas, shifts) and
            conditional functional score effect (e.g. ) added.

            The columns are ordered as follows:
            - beta_a, beta_b, ... : the latent effect of the mutation
            - shift_b, shift_c, ... : the conditional shift of the mutation
            - predicted_func_score_a, predicted_func_score_b, ... : the
                predicted functional score of the mutation.
        """
        mutations_df = self.data.mutations_df.set_index("mutation")  # returns a copy
        if not return_split:
            mutations_df.drop(
                ["wts", "sites", "muts"],
                axis=1,
                inplace=True,
            )

        # make sure the mutations_df matches the binarymaps
        for condition in self.data.conditions:
            assert onp.all(
                mutations_df.index.values == self.data.binarymaps[condition].all_subs
            ), f"mutations_df does not match binarymaps for condition {condition}"

        # make sure the indices into the bmap are ordered 0-n
        for i, sub in enumerate(mutations_df.index.values):
            assert sub == self.data.binarymaps[self.data.reference].i_to_sub(
                i
            ), f"mutation {sub} df index does not match binarymaps respective index"

        params = self.params
        for condition in self.data.conditions:
            mutations_df[f"beta_{condition}"] = params["beta"][condition]
            if condition != self._data.reference:
                mutations_df[f"shift_{condition}"] = params["shift"][condition]

        for condition in self.data.conditions:
            single_mut_binary = self.data.single_mut_encodings[condition]
            mutations_df[
                f"predicted_func_score_{condition}"
            ] = self.phenotype_frombinary(single_mut_binary, condition=condition)

            if phenotype_as_effect:
                wt_func_score = self.wildtype_df.loc[condition, "predicted_func_score"]
                mutations_df[f"predicted_func_score_{condition}"] -= wt_func_score

        # filter by times seen
        if times_seen_threshold > 0:
            for condition in self._data.conditions:
                mutations_df = mutations_df[
                    mutations_df[f"times_seen_{condition}"] >= times_seen_threshold
                ]

        return mutations_df

    def get_df_loss(self, df, error_if_unknown=False, verbose=False, conditional=False):
        """
        Get the loss of the model on a given data frame.

        Parameters
        ----------
        df : pandas.DataFrame
            Data frame containing variants. Requirements are the same as
            those used to initialize the `multidms.Data` object - except
            the indices must be unique.
        error_if_unknown : bool
            If some of the substitutions in a variant are not present in
            the model (not in :attr:`AbstractEpistasis.binarymap`)
            then by default we do not include those variants
            in the loss calculation. If `True`, raise an error.
        verbose : bool
            If True, print the number of valid and invalid variants.
        conditional : bool
            If True, return the loss for each condition as a dictionary.
            If False, return the total loss.

        Returns
        -------
        float or dict
            The loss of the model on the given data frame.
        """
        substitutions_col = "aa_substitutions"
        condition_col = "condition"
        func_score_col = "func_score"
        ref_bmap = self.data.binarymaps[self.data.reference]

        if substitutions_col not in df.columns:
            raise ValueError("`df` lacks `substitutions_col` " f"{substitutions_col}")
        if condition_col not in df.columns:
            raise ValueError("`df` lacks `condition_col` " f"{condition_col}")

        kwargs = {
            "scale_coeff_ridge_beta": 0.0,
            "scale_coeff_ridge_shift": 0.0,
            "scale_coeff_ridge_gamma": 0.0,
            "scale_ridge_alpha_d": 0.0,
        }
        loss_fxn = jax.jit(self.model_components["objective"])

        ret = {}
        for condition, condition_df in df.groupby(condition_col):
            X, y = {}, {}
            variant_subs = condition_df[substitutions_col]
            if condition not in self.data.reference_sequence_conditions:
                variant_subs = condition_df.apply(
                    lambda x: self.data.convert_subs_wrt_ref_seq(
                        condition, x[substitutions_col]
                    ),
                    axis=1,
                )

            # build binary variants as csr matrix, make prediction, and append
            valid, invalid = 0, 0  # row indices of elements that are one
            variant_targets = []
            row_ind = []  # row indices of elements that are one
            col_ind = []  # column indices of elements that are one

            for subs, target in zip(variant_subs, condition_df[func_score_col]):
                try:
                    for isub in ref_bmap.sub_str_to_indices(subs):
                        row_ind.append(valid)
                        col_ind.append(isub)
                    variant_targets.append(target)
                    valid += 1

                except ValueError:
                    if error_if_unknown:
                        raise ValueError(
                            "Variant has substitutions not in model:"
                            f"\n{subs}\nMaybe use `unknown_as_nan`?"
                        )
                    else:
                        invalid += 1

            if verbose:
                print(
                    f"condition: {condition}, n valid variants: "
                    f"{valid}, n invalid variants: {invalid}"
                )

            # X[condition] = sparse.BCOO.from_scipy_sparse(
            # scipy.sparse.csr_matrix(onp.vstack(binary_variants))
            # )
            X[condition] = sparse.BCOO.from_scipy_sparse(
                scipy.sparse.csr_matrix(
                    (onp.ones(len(row_ind), dtype="int8"), (row_ind, col_ind)),
                    shape=(valid, ref_bmap.binarylength),
                    dtype="int8",
                )
            )

            y[condition] = jnp.array(variant_targets)
            ret[condition] = float(loss_fxn(self.params, (X, y), **kwargs))

        ret["total"] = sum(ret.values()) / len(ret.values())

        if not conditional:
            return ret["total"]
        return ret

    def add_phenotypes_to_df(
        self,
        df,
        substitutions_col="aa_substitutions",
        condition_col="condition",
        latent_phenotype_col="predicted_latent",
        observed_phenotype_col="predicted_func_score",
        converted_substitutions_col="aa_subs_wrt_ref",
        overwrite_cols=False,
        unknown_as_nan=False,
        phenotype_as_effect=True,
    ):
        """Add predicted phenotypes to data frame of variants.

        Parameters
        ----------
        df : pandas.DataFrame
            Data frame containing variants. Requirements are the same as
            those used to initialize the `multidms.Data` object - except
            the indices must be unique.
        substitutions_col : str
            Column in `df` giving variants as substitution strings
            with respect to a given variants condition.
            These will be converted to be with respect to the reference sequence
            prior to prediction. Defaults to 'aa_substitutions'.
        condition_col : str
            Column in `df` giving the condition from which a variant was
            observed. Values must exist in the self.data.conditions and
            and error will be raised otherwise. Defaults to 'condition'.
        latent_phenotype_col : str
            Column added to `df` containing predicted latent phenotypes.
        observed_phenotype_col : str
            Column added to `df` containing predicted observed phenotypes.
        converted_substitutions_col : str or None
            Columns added to `df` containing converted substitution strings
            for non-reference conditions if they do not share a wildtype seq.
        overwrite_cols : bool
            If the specified latent or observed phenotype column already
            exist in `df`, overwrite it? If `False`, raise an error.
        unknown_as_nan : bool
            If some of the substitutions in a variant are not present in
            the model (not in :attr:`AbstractEpistasis.binarymap`) set the
            phenotypes to `nan` (not a number)? If `False`, raise an error.
        phenotype_as_effect : bool
            if True, phenotypes (both latent, and func_score)
            are calculated as the _difference_ between predicted
            phenotype of a given variant and the respective experimental
            wildtype prediction. Otherwise, report the unmodified
            model prediction.

        Returns
        -------
        pandas.DataFrame
            A copy of `df` with the phenotypes added. Phenotypes are predicted
            based on the current state of the model.
        """
        ref_bmap = self.data.binarymaps[self.data.reference]
        if substitutions_col is None:
            substitutions_col = ref_bmap.substitutions_col
        if substitutions_col not in df.columns:
            raise ValueError("`df` lacks `substitutions_col` " f"{substitutions_col}")
        if condition_col not in df.columns:
            raise ValueError("`df` lacks `condition_col` " f"{condition_col}")
        if not df.index.is_unique:
            raise ValueError("`df` must have unique indices")

        # return copy
        ret = df.copy()

        if phenotype_as_effect:
            wildtype_df = self.wildtype_df

        # initialize new columns
        for col in [
            latent_phenotype_col,
            observed_phenotype_col,
            converted_substitutions_col,
        ]:
            if col is None:
                continue
            if col in df.columns and not overwrite_cols:
                raise ValueError(f"`df` already contains column {col}")
            ret[col] = onp.nan

        # if the user has provided a name for the converted subs, then
        # we need to add it to the dataframe
        if converted_substitutions_col is not None:
            ret[converted_substitutions_col] = ""

        for condition, condition_df in df.groupby(condition_col):
            variant_subs = condition_df[substitutions_col]
            if condition not in self.data.reference_sequence_conditions:
                variant_subs = condition_df.apply(
                    lambda x: self.data.convert_subs_wrt_ref_seq(
                        condition, x[substitutions_col]
                    ),
                    axis=1,
                )

            if converted_substitutions_col is not None:
                ret.loc[condition_df.index, converted_substitutions_col] = variant_subs

            # build binary variants as csr matrix, make prediction, and append
            row_ind = []  # row indices of elements that are one
            col_ind = []  # column indices of elements that are one
            nan_variant_indices = []  # indices of variants that are nan

            for ivariant, subs in enumerate(variant_subs):
                try:
                    for isub in ref_bmap.sub_str_to_indices(subs):
                        row_ind.append(ivariant)
                        col_ind.append(isub)
                except ValueError:
                    if unknown_as_nan:
                        nan_variant_indices.append(ivariant)
                    else:
                        raise ValueError(
                            "Variant has substitutions not in model:"
                            f"\n{subs}\nMaybe use `unknown_as_nan`?"
                        )

            X = sparse.BCOO.from_scipy_sparse(
                scipy.sparse.csr_matrix(
                    (onp.ones(len(row_ind), dtype="int8"), (row_ind, col_ind)),
                    shape=(len(condition_df), ref_bmap.binarylength),
                    dtype="int8",
                )
            )

            # latent predictions on binary variants, X
            latent_predictions = onp.array(
                self.latent_frombinary(X, condition=condition)
            )
            assert len(latent_predictions) == len(condition_df)
            if phenotype_as_effect:
                latent_predictions -= wildtype_df.loc[condition, "predicted_latent"]
            latent_predictions[nan_variant_indices] = onp.nan
            ret.loc[
                condition_df.index.values, latent_phenotype_col
            ] = latent_predictions

            # func_score predictions on binary variants, X
            phenotype_predictions = onp.array(
                self.phenotype_frombinary(X, condition=condition)
            )
            assert len(phenotype_predictions) == len(condition_df)
            if phenotype_as_effect:
                phenotype_predictions -= wildtype_df.loc[
                    condition, "predicted_func_score"
                ]
            phenotype_predictions[nan_variant_indices] = onp.nan
            ret.loc[
                condition_df.index.values, observed_phenotype_col
            ] = phenotype_predictions

        return ret

    def mutation_site_summary_df(self, agg_func="mean", **kwargs):
        """
        Get all single mutational attributes from self._data
        updated with all model specific attributes, then aggregate
        all numerical columns by "sites"

        Parameters
        ----------
        agg_func : str
            Aggregation function to use on the numerical columns.
            Defaults to "mean".
        **kwargs
            Additional keyword arguments to pass to get_mutations_df.

        Returns
        -------
        pandas.DataFrame
            A summary of the mutation attributes aggregated by site.
        """
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        mut_df = self.get_mutations_df(**kwargs).select_dtypes(include=numerics)
        return mut_df.groupby("sites").agg(agg_func)

    def get_condition_params(self, condition=None):
        """Get the relent parameters for a model prediction"""
        condition = self.data.reference if condition is None else condition
        if condition not in self.data.conditions:
            raise ValueError(f"condition {condition} does not exist in model")
        return {
            "beta0": self.params["beta0"][condition],
            "beta": self.params["beta"][condition],
            "shift": self.params["shift"][condition],
            # GAMMA
            # "gamma": self.params["gamma"][condition],
            "theta": self.params["theta"],
        }

    def phenotype_fromsubs(self, aa_subs, condition=None):
        """
        take a single string of subs which are
        not already converted wrt reference, convert them and
        then make a functional score prediction and return the result.
        """
        converted_subs = self.data.convert_subs_wrt_ref_seq(condition, aa_subs)
        X = jnp.array(
            [
                self.data.binarymaps[self.data.reference].sub_str_to_binary(
                    converted_subs
                )
            ]
        )
        return float(self.phenotype_frombinary(X, condition)[0])

    def latent_fromsubs(self, aa_subs, condition=None):
        """
        take a single string of subs which are
        not already converted wrt reference, convert them and
        them make a latent prediction and return the result.
        """
        converted_subs = self.data.convert_subs_wrt_ref_seq(condition, aa_subs)
        X = jnp.array(
            [
                self.data.binarymaps[self.data.reference].sub_str_to_binary(
                    converted_subs
                )
            ]
        )
        return self.latent_frombinary(X, condition)

    def phenotype_frombinary(self, X, condition=None):
        """
        Condition specific functional score prediction
        on X using the biophysical model
        given current model parameters.

        Parameters
        ----------
        X : jnp.array
            Binary encoded variants to make predictions on.
        condition : str
            Condition to make predictions for. If None, use the reference
        """
        d_params = self.get_condition_params(condition)
        if X.shape[0] > 1000:
            return jax.jit(self.model_components["f"])(d_params, X)
        else:
            return self.model_components["f"](d_params, X)

    def latent_frombinary(self, X, condition=None):
        """
        Condition specific latent phenotype prediction
        on X using the biophysical model
        given current model parameters.

        Parameters
        ----------
        X : jnp.array
            Binary encoded variants to make predictions on.
        condition : str
            Condition to make predictions for. If None, use the reference
        """
        d_params = self.get_condition_params(condition)
        return jax.jit(self.model_components["latent_model"])(d_params, X)

    def fit(
        self,
        scale_coeff_lasso_shift=1e-5,
        tol=1e-4,
        maxiter=1000,
        maxls=15,
        acceleration=True,
        lock_params={},
        admm_niter=50,
        admm_tau=1.0,
        warn_unconverged=True,
        upper_bound_ge_scale="infer",
        convergence_trajectory_resolution=10,
        **kwargs,
    ):
        r"""
        Use jaxopt.ProximalGradiant to optimize the model's free parameters.

        Parameters
        ----------
        scale_coeff_lasso_shift : float
            L1 penalty coefficient applied "shift" in beta_d parameters.
            Defaults to 1e-4. This parameter is used to regularize the
            shift parameters in the model if there's more than one condition.
        tol : float
            Tolerance for the optimization convergence criteria. Defaults to 1e-4.
        maxiter : int
            Maximum number of iterations for the optimization. Defaults to 1000.
        maxls : int
            Maximum number of iterations to perform during line search.
        acceleration : bool
            If True, use FISTA acceleration. Defaults to True.
        lock_params : dict
            Dictionary of parameters, and desired value to constrain
            them at during optimization. By default, no parameters are locked.
        admm_niter : int
            Number of iterations to perform during the ADMM optimization.
            Defaults to 50. Note that in the case of single-condition models,
            This is set to zero as the generalized
            lasso ADMM optimization is not used.
        admm_tau : float
            ADMM step size. Defaults to 1.0.
        warn_unconverged : bool
            If True, raise a warning if the optimization does not converge.
            convergence is defined by whether the model tolerance (''tol'') threshold
            was passed during the optimization process.
            Defaults to True.
        upper_bound_ge_scale : float, None, or 'infer'
            The positive upper bound of the theta scale parameter -
            negative values are not allowed.
            Passing ``None`` allows the scale of the sigmoid to be unconstrained.
            Passing the string literal 'infer' results in the
            scale being set to double the range of the training data.
            Defaults to 'infer'.
        convergence_trajectory_resolution : int
            The resolution of the loss and error trajectory recorded
            during optimization. Defaults to 100.
        **kwargs : dict
            Additional keyword arguments passed to the objective function.
            See the multidms.biophysical.smooth_objective docstring for
            details on the other hyperparameters that may be supplied to
            regularize and otherwise modify the objective function
            being optimized.
        """
        # CONFIG PROXIMAL
        # infer the range of the training data, and double it
        # to set the upper bound of the theta (post-latent e.g. sigmoid) scale parameter.
        # see https://github.com/matsengrp/multidms/issues/143 for details
        if not isinstance(upper_bound_ge_scale, (float, int, type(None), str)):
            raise ValueError(
                "upper_bound_theta_ge_scale must be a float, None, or 'infer'"
            )
        if isinstance(upper_bound_ge_scale, (float, int)):
            if upper_bound_ge_scale < 0:
                raise ValueError("upper_bound_theta_ge_scale must be non-negative")

        if upper_bound_ge_scale == "infer":
            y = jnp.concatenate(list(self.data.training_data["y"].values()))
            y_range = y.max() - y.min()
            upper_bound_ge_scale = 2 * y_range

        compiled_proximal = self._model_components["proximal"]
        compiled_objective = jax.jit(self._model_components["objective"])

        # if we have more than one condition, we need to set up the ADMM optimization
        if len(self.data.conditions) > 1:
            non_identical_signs = {
                condition: jnp.where(self.data._bundle_idxs[condition], -1, 1)
                for condition in self.data.conditions
            }
            non_identical_sign_matrix = jnp.vstack(
                [non_identical_signs[d] for d in self.data.conditions]
            )
            diff_matrix = difference_matrix(
                len(self.data.conditions), self.data.reference_index
            )
            D_block_diag = scipy.sparse.block_diag(
                [
                    jnp.array(diff_matrix) @ jnp.diag(non_identical_sign_matrix[:, col])
                    for col in range(len(self.data.mutations))
                ]
            )
            Dop = pylops.LinearOperator(
                Op=scipy.sparse.linalg.aslinearoperator(D_block_diag),
                dtype=diff_matrix.dtype,
                shape=D_block_diag.shape,
            )
            eig = jnp.real((Dop.H * Dop).eigs(neigs=1, which="LM")[0])

            admm_mu = 0.99 * admm_tau / eig

            if len(self.data.conditions) > 1:
                assert 0 < admm_mu < admm_tau / eig

            hyperparams_prox = (
                scale_coeff_lasso_shift,
                admm_niter,
                admm_tau,
                admm_mu,
                upper_bound_ge_scale,
                lock_params,
                self.data.bundle_idxs,
            )

            compiled_proximal = partial(self._model_components["proximal"], Dop)

        else:
            hyperparams_prox = (
                upper_bound_ge_scale,
                lock_params,
            )
            compiled_proximal = jax.jit(self._model_components["proximal"])

        solver = ProximalGradient(
            compiled_objective,
            compiled_proximal,
            tol=tol,
            maxiter=maxiter,
            acceleration=acceleration,
            maxls=maxls,
            jit=False,
        )

        # get training data
        scaled_training_data = (
            self._data.scaled_training_data["X"],
            self._data.scaled_training_data["y"],
        )

        self._state = solver.init_state(
            self._scaled_data_params,
            hyperparams_prox=hyperparams_prox,
            data=scaled_training_data,
            **kwargs,
        )

        convergence_trajectory = pd.DataFrame(
            index=range(0, maxiter + 1, convergence_trajectory_resolution)
        ).assign(loss=onp.nan, error=onp.nan)

        convergence_trajectory.index.name = "step"

        # record initial loss and error
        convergence_trajectory.loc[0, "loss"] = float(
            compiled_objective(self._scaled_data_params, scaled_training_data)
        )

        convergence_trajectory.loc[0, "error"] = float(self._state.error)

        for i in range(maxiter):
            # perform single optimization step
            self._scaled_data_params, self._state = solver.update(
                self._scaled_data_params,
                self._state,
                hyperparams_prox=hyperparams_prox,
                data=scaled_training_data,
                **kwargs,
            )
            # record loss and error trajectories at regular intervals
            if (i + 1) % convergence_trajectory_resolution == 0:
                obj_loss = float(
                    compiled_objective(
                        self._scaled_data_params, scaled_training_data, **kwargs
                    )
                )
                lasso_penalty = scale_coeff_lasso_shift * jnp.sum(
                    jnp.abs(jnp.vstack(self._scaled_data_params["shift"].values()))
                )
                convergence_trajectory.loc[i + 1, "loss"] = obj_loss + lasso_penalty
                convergence_trajectory.loc[i + 1, "error"] = float(self._state.error)

            # early stopping criteria
            if self._state.error < tol:
                self._converged = True
                break

        if not self.converged:
            if warn_unconverged:
                warnings.warn(
                    "Model training error did not reach the tolerance threshold. "
                    f"Final error: {self._state.error}, tolerance: {tol}",
                    RuntimeWarning,
                )

        self._convergence_trajectory = convergence_trajectory
        self._clear_cache()

        return None

    def plot_pred_accuracy(
        self,
        hue=True,
        show=True,
        saveas=None,
        annotate_corr=True,
        ax=None,
        r=2,
        **kwargs,
    ):
        """
        Create a figure which visualizes the correlation
        between model predicted functional score of all
        variants in the training with ground truth measurements.
        """
        df = self.get_variants_df(phenotype_as_effect=False)

        df = df.assign(
            is_wt=df["aa_substitutions"].apply(
                lambda string: True if len(string.split()) == 0 else False
            )
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=[3, 3])

        # GAMMA
        # func_score = "corrected_func_score" if self.gamma_corrected else "func_score"
        func_score = "func_score"
        sns.scatterplot(
            data=df.sample(frac=1),
            x="predicted_func_score",
            y=func_score,
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
        ylb, yub = [-1, 1] + onp.quantile(df[func_score], [0.05, 1.0])

        ax.plot([ylb, yub], [ylb, yub], "k--", lw=2)
        if annotate_corr:
            start_y = 0.95
            for c, cdf in df.groupby("condition"):
                corr = pearsonr(cdf[func_score], cdf["predicted_func_score"])[0] ** r
                metric = "pearson" if r == 1 else "R^2"
                ax.annotate(
                    f"{metric} = {corr:.2f}",
                    (0.01, start_y),
                    xycoords="axes fraction",
                    fontsize=12,
                    c=self._data.condition_colors[c],
                )
                start_y += -0.05
        # ax.set_ylabel("functional score")
        ax.set_xlabel("predicted functional score")

        ax.axhline(0, color="k", ls="--", lw=2)
        ax.axvline(0, color="k", ls="--", lw=2)

        # ax.set_ylabel("functional score + gamma$_{d}$")
        ax.set_ylabel("measured functional score")
        plt.tight_layout()
        if saveas:
            fig.savefig(saveas)
        if show:
            plt.show()
        return ax

    def plot_epistasis(
        self, hue=True, show=True, saveas=None, ax=None, sample=1.0, **kwargs
    ):
        """
        Plot latent predictions against
        gamma corrected ground truth measurements
        of all samples in the training set.
        """
        df = self.get_variants_df(phenotype_as_effect=False)

        df = df.assign(
            is_wt=df["aa_substitutions"].apply(
                lambda string: True if len(string.split()) == 0 else False
            )
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=[3, 3])

        # GAMMA
        # func_score = "corrected_func_score" if self.gamma_corrected else "func_score"
        func_score = "func_score"
        sns.scatterplot(
            data=df.sample(frac=sample),
            x="predicted_latent",
            y=func_score,
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
        ylb, yub = [-1, 1] + onp.quantile(df[func_score], [0.05, 1.0])

        latent_model_grid = onp.linspace(xlb, xub, num=1000)

        params = self.get_condition_params(self._data.reference)
        latent_preds = self._model_components["g"](params["theta"], latent_model_grid)
        shape = (latent_model_grid, latent_preds)
        ax.plot(*shape, color="k", lw=2)

        ax.axhline(0, color="k", ls="--", lw=2)
        ax.set_xlim([xlb, xub])
        ax.set_ylim([ylb, yub])
        ax.set_ylabel("measured functional score")
        ax.set_xlabel("predicted latent phenotype")
        plt.tight_layout()

        if saveas:
            fig.savefig(saveas)
        if show:
            plt.show()
        return ax

    def plot_param_hist(
        self, param, show=True, saveas=False, times_seen_threshold=0, ax=None, **kwargs
    ):
        """Plot the histogram of a parameter."""
        mut_effects_df = self.get_mutations_df()

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
        self, param, show=True, saveas=False, times_seen_threshold=0, ax=None, **kwargs
    ):
        """
        Plot the heatmap of a parameters
        associated with specific sites and substitutions.
        """
        if not param.startswith("beta") and not param.startswith("shift"):
            raise ValueError(
                "Parameter to visualize must be an existing beta, or shift parameter"
            )

        mut_effects_df = self.get_mutations_df()

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
        times_seen_threshold=0,
        agg_func="mean",
        ax=None,
        **kwargs,
    ):
        """Summarize shift parameter values by associated sites and conditions."""
        mutation_site_summary_df = self.mutation_site_summary_df(
            agg_func, times_seen_threshold=times_seen_threshold
        ).reset_index()

        if ax is None:
            fig, ax = plt.subplots(figsize=[12, 3])

        ax.axhline(0, color="k", ls="--", lw=1)

        sns.lineplot(
            data=mutation_site_summary_df,
            x="sites",
            y=f"shift_{condition}",
            color=self._data.condition_colors[condition],
            ax=ax,
            # legend=True,
            **kwargs,
        )
        color = [
            (
                self.data.condition_colors[condition]
                if s not in self.data.non_identical_sites[condition]
                else (0.0, 0.0, 0.0)
            )
            for s in mutation_site_summary_df.sites
        ]
        size = [
            0.5 if s not in self.data.non_identical_sites[condition] else 1.0
            for s in mutation_site_summary_df.sites
        ]
        sns.scatterplot(
            data=mutation_site_summary_df,
            x="sites",
            y=f"shift_{condition}",
            size=size,
            color=onp.array(color),
            ax=ax,
            legend=False,
            **kwargs,
        )

        if show:
            plt.show()
        return ax

    def mut_param_heatmap(
        self,
        mut_param="shift",
        times_seen_threshold=0,
        phenotype_as_effect=True,
        **line_and_heat_kwargs,
    ):
        """
        Wrapper method for visualizing the shift plot.
        see `multidms.plot.mut_shift_plot()` for more
        """  # noqa: D401
        possible_mut_params = set(["shift", "predicted_func_score", "beta"])
        if mut_param not in possible_mut_params:
            raise ValueError(f"invalid {mut_param=}")

        # aggregate mutation values between dataset fits
        muts_df = self.get_mutations_df(
            times_seen_threshold=times_seen_threshold,
            phenotype_as_effect=phenotype_as_effect,
            return_split=True,
        ).rename(
            columns={
                "wts": "wildtype",
                "muts": "mutant",
                "sites": "site",
            }
        )

        # drop columns which are not the mutational parameter of interest
        drop_cols = [c for c in muts_df.columns if "times_seen" in c]
        for param in possible_mut_params - set([mut_param]):
            drop_cols.extend([c for c in muts_df.columns if c.startswith(param)])
        muts_df.drop(drop_cols, axis=1, inplace=True)

        # add in the mutation annotations
        # muts_df["wildtype"], muts_df["site"], muts_df["mutant"] = zip(
        #     *muts_df.reset_index()["mutation"].map(self.data.parse_mut)
        # )

        # no longer need mutation annotation
        muts_df.reset_index(drop=True, inplace=True)

        # add conditional wildtypes
        fit = self
        reference = fit.data.reference
        conditions = fit.data.conditions
        site_map = fit.data.site_map

        wt_dict = {
            "wildtype": site_map[reference].values,
            "mutant": site_map[reference].values,
            "site": site_map[reference].index.values,
        }
        [c for c in muts_df.columns if c.startswith(mut_param)]
        for value_col in [c for c in muts_df.columns if c.startswith(mut_param)]:
            wt_dict[value_col] = 0

        # add reference wildtype values needed for lineplot and heatmap fx
        muts_df = pd.concat([muts_df, pd.DataFrame(wt_dict)])

        # add in wildtype values for each non-reference condition
        # these will be available in the tooltip
        addtl_tooltip_stats = []
        for condition in conditions:
            if condition == reference:
                continue
            addtl_tooltip_stats.append(f"wildtype_{condition}")
            muts_df[f"wildtype_{condition}"] = muts_df.site.apply(
                lambda site: site_map.loc[site, condition]
            )

        # melt conditions and stats cols, beta is already "tall"
        # note that we must rename conditions with "." in the
        # name to "_" to avoid altair errors
        # if mut_param == f"beta_{reference}":
        # muts_df_tall = muts_df.assign(condition=reference.replace(".", "_"))
        # else:
        muts_df_tall = muts_df.melt(
            id_vars=["wildtype", "site", "mutant"] + addtl_tooltip_stats,
            value_vars=[c for c in muts_df.columns if c.startswith(mut_param)],
            var_name="condition",
            value_name=mut_param,
        )
        muts_df_tall["condition"] = muts_df_tall.condition.replace(
            {
                f"{mut_param}_{condition}": condition.replace(".", "_")
                for condition in conditions
            },
        )

        # add in condition colors, rename for altair
        condition_colors = {
            con.replace(".", "_"): col for con, col in fit.data.condition_colors.items()
        }

        # rename for altair
        addtl_tooltip_stats = [v.replace(".", "_") for v in addtl_tooltip_stats]
        muts_df_tall.rename(
            {c: c.replace(".", "_") for c in muts_df_tall.columns}, axis=1, inplace=True
        )

        kwargs = {
            "data_df": muts_df_tall,
            "stat_col": mut_param,
            "addtl_tooltip_stats": addtl_tooltip_stats,
            "category_col": "condition",
            "heatmap_color_scheme": "redblue",
            "init_floor_at_zero": False,
            "categorical_wildtype": True,
            "category_colors": condition_colors,
        }

        # return multidms.plot._lineplot_and_heatmap(**kwargs, **line_and_heat_kwargs),
        return _lineplot_and_heatmap(**kwargs, **line_and_heat_kwargs)
